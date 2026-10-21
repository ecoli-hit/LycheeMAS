"""Offline and optional legacy-runner parity checks for MAD-M²."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from lychee_mas.backends.llama_cpp_backend import LlamaCppBackend, LlamaCppCompletionResult
from lychee_mas.core.registry import REGISTRY
from lychee_mas.methods.build.mad_m2_builder import new_mad_m2_state
from lychee_mas.methods.utils.mad_m2_utils import (
    extract_response,
    generation_seed,
    mask_decision,
    render_single_user_chat,
    vote_answers,
)
from lychee_mas.plugins.build import build_langgraph
from lychee_mas.plugins.prerun.graphview import extract_view

QUESTION_ID = 600
SEED = 41
QUESTION = (
    "Jamaal is at the gym. He has been using an 8-pound weight. He increases "
    "the weight that he uses by 50%. It turns out to be too heavy, so he uses "
    "a weight two pounds lighter than that. What is the weight, in pounds, "
    "that he now uses?"
)
GOLD = (
    "An increase of 50% of 8 is 1.5*8=12.\nHe switches to a "
    "12-2=<<12-2=10>>10 pound weight.\n#### 10"
)


def scripted_text(stage: str, index: int) -> str:
    if stage == "mask_eval":
        return "<label>NO</label>"  # force the auditable empty-context fallback
    answer = {"round1": [10, 12, 9]}.get(stage, [10] * 6)[index]
    return f"<think>scripted calculation {index}</think><answer>{answer}</answer>"


class ScriptedBackend:
    def __init__(self):
        self.requests: list[tuple[str, int]] = []
        self.by_seed = {
            generation_seed(SEED, QUESTION_ID, stage, index): (stage, index)
            for stage, size in (
                ("cot", 1),
                ("cot_sc", 6),
                ("round1", 3),
                ("mask_eval", 3),
                ("round2", 3),
            )
            for index in range(size)
        }

    def complete(self, prompt: str, *, seed: int, **kwargs) -> LlamaCppCompletionResult:
        self.requests.append((prompt, seed))
        stage, index = self.by_seed[seed]
        return LlamaCppCompletionResult(
            text=scripted_text(stage, index),
            n_prompt_pos=20,
            n_gen_tokens=8,
            latency_s=0.01,
            stop_type="eos",
            server_seed=seed,
            stopped=True,
            stopping_word="",
            raw_response={},
        )


@pytest.mark.parametrize(
    "variant,call_count",
    [
        ("cot", 1),
        ("cot_sc", 6),
        ("mad", 6),
        ("mad_m2s", 9),
    ],
)
def test_registered_static_graph_runs_all_variants(variant, call_count):
    assert "mad_m2" in REGISTRY.list("graph_builder")
    backend = ScriptedBackend()
    graph = build_langgraph(method="mad_m2", backend=backend, variant=variant)
    view = extract_view(graph)
    assert view.terminal == "vote"
    assert not graph.branches
    result = graph.compile().invoke(new_mad_m2_state(QUESTION, QUESTION_ID, SEED, GOLD))
    assert len(result["calls"]) == call_count
    assert result["score"] == 1.0
    assert result["prediction"] == ("10" if variant == "cot" else 10.0)
    assert all(
        item["rendered_prompt"].endswith("<|im_start|>assistant\n") for item in result["calls"]
    )
    if variant == "mad_m2s":
        assert [item["keep"] for item in result["masks"]] == [False] * 3
        assert result["empty_context_fallback"]["reason"] == (
            "all_three_subjective_contexts_masked"
        )
        assert result["all_masked_fallback"] is True
        assert result["calls"][-1]["prompt"] == result["calls"][0]["prompt"]


def test_backend_preserves_raw_completion_payload():
    seed = generation_seed(SEED, QUESTION_ID, "cot", 0)
    response = io.BytesIO(
        json.dumps(
            {
                "content": scripted_text("cot", 0),
                "tokens_evaluated": 20,
                "tokens_predicted": 8,
                "stop": True,
                "stop_type": "eos",
                "stopping_word": "",
                "generation_settings": {"seed": seed},
            }
        ).encode()
    )
    backend = LlamaCppBackend()
    prompt = render_single_user_chat("Question?")
    with patch(
        "lychee_mas.backends.llama_cpp_backend.urllib.request.urlopen", return_value=response
    ) as urlopen:
        result = backend.complete(
            prompt,
            seed=seed,
            temperature=1.0,
            top_p=1.0,
            max_tokens=1024,
            stop=[],
            ignore_eos=False,
        )
    request = urlopen.call_args.args[0]
    payload = json.loads(request.data)
    assert request.full_url.endswith("/completion")
    assert payload["prompt"] == prompt
    assert payload["seed"] == seed
    assert payload["n_predict"] == 1024
    assert payload["stop"] == [] and payload["ignore_eos"] is False
    assert (result.text, result.n_prompt_pos, result.n_gen_tokens, result.stop_type) == (
        scripted_text("cot", 0),
        20,
        8,
        "eos",
    )


def test_author_tie_and_unparseable_label_rules():
    consensus, winner, counts = vote_answers(["10", "9", "10", "9", "10", "9"])
    assert (consensus, winner, counts) == (True, 10.0, {10.0: 3, 9.0: 3})
    assert mask_decision("unparseable response") == ("none", True)


def test_backend_rejects_missing_token_counts():
    response = io.BytesIO(json.dumps({"content": "<answer>4</answer>"}).encode())
    with patch(
        "lychee_mas.backends.llama_cpp_backend.urllib.request.urlopen", return_value=response
    ):
        with pytest.raises(ValueError, match="tokens_evaluated"):
            LlamaCppBackend().complete("x", seed=1, temperature=1.0, top_p=1.0, max_tokens=16)


@pytest.mark.parametrize("variant", ["cot", "mad_m2s"])
def test_one_fixed_case_matches_legacy_runner_when_available(variant):
    """Set MAD_M2_LEGACY_ROOT to run old script and the new graph side by side.

    A self-contained scripted regression above remains runnable in upstream CI.
    """
    root_text = os.environ.get("MAD_M2_LEGACY_ROOT")
    if not root_text:
        pytest.skip("set MAD_M2_LEGACY_ROOT for external reproduction parity")
    root = Path(root_text).resolve()
    script_path = root / "scripts" / "phase_a.py"
    if not script_path.is_file():
        pytest.fail(f"legacy runner missing: {script_path}")
    sys.path.insert(0, str(root))
    try:
        spec = importlib.util.spec_from_file_location("legacy_phase_a", script_path)
        assert spec and spec.loader
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        backend = ScriptedBackend()
        old_requests: list[tuple[str, int]] = []

        def fake_generate(_url, _pid, prompt, seed, _shared, _device, parse_answer):
            old_requests.append((render_single_user_chat(prompt), seed))
            stage, index = backend.by_seed[seed]
            raw = scripted_text(stage, index)
            parsed, source = extract_response(raw) if parse_answer else (None, None)
            return {
                "seed": seed,
                "prompt": prompt,
                "rendered_prompt": render_single_user_chat(prompt),
                "raw_response": raw,
                "parsed": parsed,
                "extraction_source": source,
                "seed_matches": True,
                "input_tokens": 20,
                "output_tokens": 8,
                "elapsed_seconds": 0.01,
            }

        seeds = {
            stage: [generation_seed(SEED, QUESTION_ID, stage, i) for i in range(size)]
            for stage, size in (
                ("cot", 1),
                ("cot_sc", 6),
                ("round1", 3),
                ("mask_eval", 3),
                ("round2", 3),
            )
        }
        with patch.object(legacy, "generate", side_effect=fake_generate):
            old = legacy.run_method(
                "mad_m2_subjective" if variant == "mad_m2s" else variant,
                {"id": QUESTION_ID, "query": QUESTION, "answer": GOLD},
                seeds,
                {},
                {"llama_cpp": {"backend": "CPU"}},
                "http://127.0.0.1:8080",
                1,
            )
        graph = build_langgraph(method="mad_m2", backend=backend, variant=variant)
        new = graph.compile().invoke(new_mad_m2_state(QUESTION, QUESTION_ID, SEED, GOLD))
        assert old_requests == backend.requests  # prompt bytes and per-call seeds
        assert [(c["raw_response"], c["parsed"]) for c in old["calls"]] == [
            (c["raw_response"], c["parsed"]) for c in new["calls"]
        ]
        assert old["prediction"] == new["prediction"]
        assert old["score"] == new["score"]
        assert old["empty_context_fallback"] == new["empty_context_fallback"]
    finally:
        sys.path.remove(str(root))


@pytest.mark.parametrize("variant", ["cot", "mad_m2s"])
def test_live_one_case_matches_legacy_when_server_is_supplied(variant):
    """Optional real-model parity; never starts or modifies the model service."""
    root_text = os.environ.get("MAD_M2_LEGACY_ROOT")
    server_url = os.environ.get("MAD_M2_REAL_URL")
    if not (root_text and server_url):
        pytest.skip("set legacy root and real URL for live parity")
    root = Path(root_text).resolve()
    sys.path.insert(0, str(root))
    try:
        import yaml

        spec = importlib.util.spec_from_file_location(
            "legacy_phase_a_live", root / "scripts" / "phase_a.py"
        )
        assert spec and spec.loader
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        shared = yaml.safe_load((root / "protocol" / "shared_config.yaml").read_text())
        device = yaml.safe_load((root / "protocol" / "mac_cpu_config.yaml").read_text())
        # Local parity probe only: both runners get the same reduced output cap.
        device["llama_cpp"]["max_output_tokens"] = 256
        seeds = legacy.seed_map(shared, [QUESTION_ID])[QUESTION_ID]
        # The legacy RSS sampler watches this test process; memory is not part
        # of this parity assertion and must not be reported as model RSS.
        old = legacy.run_method(
            "mad_m2_subjective" if variant == "mad_m2s" else variant,
            {"id": QUESTION_ID, "query": QUESTION, "answer": GOLD},
            seeds,
            shared,
            device,
            server_url,
            os.getpid(),
        )
        graph = build_langgraph(
            method="mad_m2",
            variant=variant,
            backend=LlamaCppBackend(server_url),
            sampling=shared["sampling"],
            max_tokens=256,
            stop=shared["stop_conditions"]["stop_strings"],
            ignore_eos=shared["stop_conditions"]["ignore_eos"],
        )
        new = graph.compile().invoke(new_mad_m2_state(QUESTION, QUESTION_ID, SEED, GOLD))
        for old_call, new_call in zip(old["calls"], new["calls"], strict=True):
            assert old_call["rendered_prompt"] == new_call["rendered_prompt"]
            assert old_call["seed"] == new_call["seed"]
            assert old_call["raw_response"] == new_call["raw_response"]
            assert old_call["parsed"] == new_call["parsed"]
        assert old["prediction"] == new["prediction"]
        assert old["score"] == new["score"]
        assert old["empty_context_fallback"] == new["empty_context_fallback"]
    finally:
        sys.path.remove(str(root))
