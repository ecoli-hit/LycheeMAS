"""MAD-M² GSM8K protocol from the independent local reproduction.

The prompts and answer rules follow MAD-MM commit
f02069add08280b764d059a2f06ca0043aa093e2. The per-call SHA256 seed
schedule and explicit Qwen2.5 chat rendering are local reproducibility choices.
Source: https://github.com/tmlr-group/MAD-MM
Adapted from ``mad-m2-local-reproduction/src/phase_a_protocol.py``.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

COT_PROMPT = "\nPlease solve the problem step by step.\n"
DEBATE_PROMPT = (
    "\nThese are the potential solutions to the problem:\n{context}\n"
    "Use the potential solutions as additional information for the following question."
    "\n\nQuestion:{question}\nPlease think step by step and solve the problem.\n"
)
PRUNE_PROMPT = (
    "\nEvaluate the given solutions based on the question. ** Your reponse MUST end "
    "with the following format: <label>YES</label> or <label>NO</label> or "
    "<label>NOT SURE</label>. ** Return YES if the solution is completely correct, "
    "NO if any part of the solution is incorrect, and NOT SURE if you are unsure."
    "\n\n\nQuestion: {question}\n\nSolutions: {solution}\n"
)
RESPONSE_SUFFIX = (
    "\n\n### Response format (MUST be strictly followed) (DO NOT include any other "
    "formats except for the given XML format): \n<think>YOUR THINKING HERE</think>"
    "\n<answer>YOUR FINAL ANSWER ONLY, NO OTHER TEXT</answer>."
)

DEFAULT_SYSTEM_MESSAGE = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
SEED_STAGES = frozenset({"cot", "cot_sc", "round1", "mask_eval", "round2"})


def format_generation_prompt(question: str, contexts: list[dict[str, str]] | None = None) -> str:
    """Match author prompts_with_format for CoT or a nonempty debate context."""
    if contexts:
        prompt = DEBATE_PROMPT.format(
            question=question, context="\n".join(str(context) for context in contexts)
        )
    else:
        prompt = f"{question}\n{COT_PROMPT}"
    return prompt + RESPONSE_SUFFIX


def format_mask_prompt(question: str, context: dict[str, str]) -> str:
    return PRUNE_PROMPT.format(question=question, solution=context)


def render_single_user_chat(prompt: str) -> str:
    """Render the shared no-tools branch of HF and GGUF Qwen2.5 chat templates."""
    return (
        f"<|im_start|>system\n{DEFAULT_SYSTEM_MESSAGE}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def generation_seed(base_seed: int, question_id: int, stage: str, index: int) -> int:
    """Local rule: 1 + first SHA256 uint32 modulo 2**31-1."""
    if stage not in SEED_STAGES or index < 0:
        raise ValueError(f"Invalid generation stage/index: {stage}/{index}")
    key = f"{base_seed}:{question_id}:{stage}:{index}".encode("utf-8")
    return 1 + int.from_bytes(hashlib.sha256(key).digest()[:4], "big") % (2**31 - 1)


def extract_number(text: Any) -> float | None:
    """Match src/utils.py:extract_number; the last numeric substring wins."""
    matches = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?|\d+\.\d+", str(text))
    if matches:
        try:
            return float(matches[-1].replace(",", ""))
        except ValueError:
            return None
    return None


def _find_box(text: str) -> str | None:
    matches = re.findall(r"\\boxed\{(.*)\}", text)
    if not matches:
        matches = re.findall(r"\\fbox\{(.*)\}", text)
    return matches[-1] if matches else None


def extract_answer_with_source(text: str) -> tuple[str, str]:
    """Match src/qwen_math.py:extract_answer, with a logged source label."""
    matches = re.findall(r"<answer>(.*?)</answer>", text)
    if matches:
        return matches[-1], "answer_tag"
    boxed = _find_box(text)
    if boxed:
        return boxed, "boxed"
    matches = re.findall(r"####\s*(-?[0-9\.\,]+)", text)
    if matches:
        return matches[-1].replace(",", "").replace("$", ""), "gsm8k_marker"
    matches = re.findall(
        r"(?:[Tt]he(?:\s+final)?(?:\s+answer)?(?:\s+is)?:?)\s*([^\n]+)", text
    )
    if matches:
        return matches[-1], "answer_phrase"
    return text.split("\n")[-1], "last_line"


def extract_response(text: str) -> tuple[dict[str, str], str]:
    """Match author GSM8K answer processing and the model wrapper's box fallback."""
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    think = think_match.group(1).strip() if think_match else text
    answer, source = extract_answer_with_source(text)
    if not answer:
        match = re.search(r"\\boxed{(.*?)}", text, re.DOTALL)
        if match:
            answer, source = match.group(1).strip(), "wrapper_box_fallback"
        else:
            answer, source = "", "parse_failure"
    return {"think": think, "answer": answer}, source


def vote_answers(answers: list[str]) -> tuple[bool, float | None, dict[float | None, int]]:
    """Match if_reach_consensus, including first-occurrence ties and >= half."""
    if not answers:
        return False, None, {}
    normalized = [extract_number(answer) for answer in answers]
    counts: dict[float | None, int] = {}
    for answer in normalized:
        counts[answer] = counts.get(answer, 0) + 1
    winner, count = max(counts.items(), key=lambda item: item[1])
    return count >= len(answers) / 2, winner, counts


def mask_decision(raw_response: str, strict: bool = False) -> tuple[str, bool]:
    """Match subjective pruning: only a parsed NO is masked when strict=False."""
    match = re.search(r"<label>(.*?)</label>", raw_response, re.DOTALL)
    label = match.group(1).strip() if match else None
    normalized = re.sub(
        r"[\(\[\{]([A-Za-z])[\)\]\}]|([A-Za-z])[\.:\\)]", r"\1\2", str(label)
    ).lower().strip()
    if normalized == "yes":
        return normalized, True
    if normalized == "no":
        return normalized, False
    return normalized, not strict


def filter_subjective_contexts(
    contexts: list[dict[str, str]], masks: list[dict]
) -> tuple[list[dict[str, str]], dict]:
    """Keep non-NO solutions and make an all-masked fallback auditable."""
    if len(contexts) != 3 or len(masks) != 3:
        raise ValueError("subjective filtering requires three contexts and three decisions")
    kept = [context for context, mask in zip(contexts, masks) if mask["keep"]]
    fallback = not kept
    return kept, {
        "triggered": fallback,
        "reason": "all_three_subjective_contexts_masked" if fallback else None,
        "policy": "use_base_question_cot_prompt",
        "kept_context_count": len(kept),
        "discarded_context_indices": [mask["index"] for mask in masks if not mask["keep"]],
        "mask_labels": [mask["label"] for mask in masks],
    }


def gsm8k_score(prediction: Any, gold: Any) -> float:
    """Match GSM8KEval.calculate_score for valid GSM8K gold records."""
    expected, actual = extract_number(gold), extract_number(prediction)
    if actual is None:
        return 0.0
    if expected is None:
        raise ValueError("GSM8K gold answer has no number")
    return 1.0 if abs(expected - actual) <= 1e-6 else 0.0
