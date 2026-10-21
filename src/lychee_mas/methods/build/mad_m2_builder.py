"""Protocol-locked static graphs for CoT, CoT-SC, MAD and MAD-M²(S)."""

from __future__ import annotations

from typing import Any, Mapping, TypedDict

from ...core.registry import REGISTRY
from ...core.types import AgentSpec
from ..utils.mad_m2_utils import (
    extract_number,
    extract_response,
    filter_subjective_contexts,
    format_generation_prompt,
    format_mask_prompt,
    generation_seed,
    gsm8k_score,
    mask_decision,
    render_single_user_chat,
    vote_answers,
)

VARIANTS = ("cot", "cot_sc", "mad", "mad_m2s")
DEFAULT_SAMPLING = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "min_p": 0.0,
    "typical_p": 1.0,
    "repeat_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "mirostat": 0,
    "cache_prompt": False,
}


class MadM2State(TypedDict, total=False):
    question: str
    question_id: int
    base_seed: int
    gold: str
    calls: list[dict[str, Any]]
    round1: list[dict[str, Any]]
    round2: list[dict[str, Any]]
    paths: list[dict[str, Any]]
    contexts: list[dict[str, str]]
    masks: list[dict[str, Any]] | None
    initial_vote: dict[str, Any] | None
    empty_context_fallback: dict[str, Any]
    all_masked_fallback: bool
    prediction: str | float | None
    vote_counts: list[dict[str, Any]] | None
    consensus: bool | None
    tie: bool | None
    score: float


def new_mad_m2_state(question: str, question_id: int, base_seed: int, gold: str) -> MadM2State:
    """Create a fresh state for one case; lists are never shared across cases."""
    if not question or not isinstance(question_id, int) or not isinstance(base_seed, int):
        raise ValueError("question, integer question_id and integer base_seed are required")
    return {
        "question": question,
        "question_id": question_id,
        "base_seed": base_seed,
        "gold": gold,
        "calls": [],
        "round1": [],
        "round2": [],
        "paths": [],
        "contexts": [],
        "masks": None,
        "initial_vote": None,
        "empty_context_fallback": {
            "triggered": False,
            "reason": None,
            "policy": "use_base_question_cot_prompt",
            "kept_context_count": None,
            "discarded_context_indices": [],
        },
        "all_masked_fallback": False,
    }


@REGISTRY.register("graph_builder", "mad_m2")
class MadM2GraphBuilder:
    """Expand one algorithm variant into a linear LangGraph DAG."""

    name = "mad_m2"

    def __init__(
        self,
        backend: Any,
        variant: str = "mad_m2s",
        *,
        sampling: Mapping[str, Any] | None = None,
        max_tokens: int = 1024,
        stop: list[str] | None = None,
        ignore_eos: bool = False,
    ) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        if backend is None or not callable(getattr(backend, "complete", None)):
            raise TypeError("backend must provide complete(prompt, **sampling)")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.backend, self.variant, self.max_tokens = backend, variant, max_tokens
        self.stop, self.ignore_eos = list(stop or []), ignore_eos
        self.sampling = dict(DEFAULT_SAMPLING)
        if sampling is not None:
            unknown = set(sampling) - set(DEFAULT_SAMPLING)
            if unknown:
                raise ValueError(f"unknown sampling fields: {sorted(unknown)}")
            self.sampling.update(sampling)

    def _call(
        self, state: MadM2State, prompt: str, stage: str, index: int, parse: bool = True
    ) -> dict[str, Any]:
        seed = generation_seed(state["base_seed"], state["question_id"], stage, index)
        rendered = render_single_user_chat(prompt)
        result = self.backend.complete(
            rendered,
            seed=seed,
            max_tokens=self.max_tokens,
            stop=self.stop,
            ignore_eos=self.ignore_eos,
            **self.sampling,
        )
        if result.server_seed != seed:
            raise ValueError(
                f"llama-server seed mismatch at {stage}/{index}: "
                f"expected {seed}, got {result.server_seed}"
            )
        parsed, source = extract_response(result.text) if parse else (None, None)
        return {
            "stage": stage,
            "index": index,
            "seed": seed,
            "prompt": prompt,
            "rendered_prompt": rendered,
            "raw_response": result.text,
            "parsed": parsed,
            "extraction_source": source,
            "parse_failure": bool(
                parse
                and (
                    parsed is None
                    or source == "parse_failure"
                    or extract_number(parsed["answer"]) is None
                )
            ),
            "input_tokens": result.n_prompt_pos,
            "output_tokens": result.n_gen_tokens,
            "elapsed_seconds": result.latency_s,
            "stop_type": result.stop_type,
            "server_seed": result.server_seed,
        }

    def _generation_node(self, stage: str, index: int):
        def node(state: MadM2State) -> dict[str, Any]:
            contexts = state["contexts"] if stage == "round2" else None
            prompt = format_generation_prompt(state["question"], contexts)
            call = self._call(state, prompt, stage, index)
            updates: dict[str, Any] = {"calls": [*state["calls"], call]}
            if stage in ("round1", "round2"):
                updates[stage] = [*state[stage], call]
            elif stage == "cot_sc":
                updates["paths"] = [*state["paths"], call]
            return updates

        return node

    def _round1_vote(self, state: MadM2State) -> dict[str, Any]:
        contexts = [call["parsed"] for call in state["round1"]]
        consensus, winner, counts = vote_answers([item["answer"] for item in contexts])
        return {
            "contexts": contexts,
            "initial_vote": {
                "consensus": consensus,
                "winner": winner,
                "counts": [{"answer": answer, "count": count} for answer, count in counts.items()],
            },
        }

    def _mask_node(self, index: int):
        def node(state: MadM2State) -> dict[str, Any]:
            prompt = format_mask_prompt(state["question"], state["contexts"][index])
            call = self._call(state, prompt, "mask_eval", index, parse=False)
            label, keep = mask_decision(call["raw_response"], strict=False)
            call.update({"mask_label": label, "mask_keep": keep})
            mask = {"index": index, "raw_label": call["raw_response"], "label": label, "keep": keep}
            return {"calls": [*state["calls"], call], "masks": [*(state["masks"] or []), mask]}

        return node

    def _filter_node(self, state: MadM2State) -> dict[str, Any]:
        kept, fallback = filter_subjective_contexts(state["contexts"], state["masks"] or [])
        return {
            "contexts": kept,
            "empty_context_fallback": fallback,
            "all_masked_fallback": fallback["triggered"],
        }

    def _vote_node(self, state: MadM2State) -> dict[str, Any]:
        if self.variant == "cot":
            prediction = state["calls"][-1]["parsed"]["answer"]
            consensus, counts = None, None
        else:
            calls = state["paths"] if self.variant == "cot_sc" else state["round2"]
            consensus, prediction, counts = vote_answers(
                [call["parsed"]["answer"] for call in calls]
            )
        vote_counts = (
            None
            if counts is None
            else [{"answer": answer, "count": count} for answer, count in counts.items()]
        )
        maximum = max((item["count"] for item in vote_counts), default=0) if vote_counts else 0
        tie = sum(item["count"] == maximum for item in vote_counts) > 1 if vote_counts else None
        return {
            "prediction": prediction,
            "consensus": consensus,
            "vote_counts": vote_counts,
            "tie": tie,
            "score": gsm8k_score(prediction, state["gold"]),
        }

    def build(self) -> Any:
        from langgraph.graph import END, START, StateGraph  # heavy dependency is lazy

        graph = StateGraph(MadM2State)
        nodes: list[tuple[str, str, Any]] = []
        if self.variant == "cot":
            nodes.append(("cot", "agent", self._generation_node("cot", 0)))
        elif self.variant == "cot_sc":
            nodes.extend(
                (f"cot_sc_{i + 1}", "agent", self._generation_node("cot_sc", i)) for i in range(6)
            )
        else:
            nodes.extend(
                (f"agent_{i + 1}_r1", "agent", self._generation_node("round1", i)) for i in range(3)
            )
            nodes.append(("round1_vote", "vote", self._round1_vote))
            if self.variant == "mad_m2s":
                nodes.extend((f"mask_eval_{i + 1}", "mask", self._mask_node(i)) for i in range(3))
                nodes.append(("filter_fallback", "filter", self._filter_node))
            nodes.extend(
                (f"agent_{i + 1}_r2", "agent", self._generation_node("round2", i)) for i in range(3)
            )
        nodes.append(("vote", "vote", self._vote_node))
        previous: str | None = None
        for name, role, node in nodes:
            spec = AgentSpec(
                name=name, role=role, meta={"predecessors": [previous] if previous else []}
            )
            graph.add_node(name, node, metadata={"agent_spec": spec})
            graph.add_edge(previous or START, name)
            previous = name
        graph.add_edge(previous, END)
        return graph
