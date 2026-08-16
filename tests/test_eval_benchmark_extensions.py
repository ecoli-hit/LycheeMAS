import json

from lychee_mas import REGISTRY
from lychee_mas.core.types import TaskQuery
from lychee_mas.eval import benchmarks  # noqa: F401
from lychee_mas.eval.benchmarks import (
    aftraj,
    agent_collab,
    bbeh,
    choice_qa,
    common,
    gsm8k,
    hle,
    locomo10,
    mast_data,
    open_agent_traces,
    workbench,
)
from lychee_mas.eval.metrics import result_dir, score, score_details
from lychee_mas.runtime.backends.autogen_runtime import AutoGenRuntime


def test_extended_benchmarks_registered():
    names = set(REGISTRY.list("benchmark"))
    assert "human_eval" in names
    assert "gaia_validation" in names
    assert "gaia_validation_level_1" in names
    assert "aftraj_audit" in names
    assert "aftraj_audit_test" in names
    assert "agent_collab_clc" in names
    assert "mast_failure" in names
    assert "open_agent_traces" in names
    assert {"bbeh", "hle", "swe_bench_verified", "workbench"}.issubset(names)
    assert "arc_easy" in benchmarks.PREPARERS
    assert "openbookqa" in benchmarks.PREPARERS
    assert "medqa" in benchmarks.PREPARERS
    assert "locomo10" in benchmarks.PREPARERS
    assert "aftraj_audit" in benchmarks.PREPARERS
    assert "agent_collab" in benchmarks.PREPARERS
    assert "mast_failure" in benchmarks.PREPARERS
    assert "open_agent_traces" in benchmarks.PREPARERS
    assert {"bbeh", "hle", "swe_bench_verified", "workbench"}.issubset(
        benchmarks.PREPARERS
    )
    assert "aftraj" in benchmarks.SOURCE_PREPARERS
    assert "aftraj_audit" not in benchmarks.SOURCE_PREPARERS
    assert benchmarks.PREPARE_ALIASES["aftraj_audit_test"] == "aftraj"
    assert benchmarks.PREPARE_ALIASES["agent_collab_clc"] == "agent_collab"


def test_benchmark_roots_prefer_new_names_and_keep_legacy_fallback(monkeypatch):
    for name in (
        "LYCHEE_BENCHMARK_RAW_ROOT",
        "LYCHEE_BENCHMARK_PREPARED_ROOT",
        "LYCHEE_BENCHMARK_RUNS_ROOT",
        "CDM_DATA_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)

    assert common.raw_root() == "data/benchmarks/raw"
    assert common.prepared_root() == "data/benchmarks/prepared"
    assert common.runs_root() == "runs/benchmarks"

    monkeypatch.setenv("CDM_DATA_ROOT", "/legacy/raw")
    assert common.raw_root() == "/legacy/raw"
    assert common.prepared_root() == "/legacy/raw"

    monkeypatch.setenv("LYCHEE_BENCHMARK_RAW_ROOT", "/new/raw")
    monkeypatch.setenv("LYCHEE_BENCHMARK_PREPARED_ROOT", "/new/prepared")
    monkeypatch.setenv("LYCHEE_BENCHMARK_RUNS_ROOT", "/new/runs")
    assert common.raw_root() == "/new/raw"
    assert common.prepared_root() == "/new/prepared"
    assert common.runs_root() == "/new/runs"
    assert common.safe_source_id("OmniData/ARC") == "OmniData--ARC"
    assert str(common.raw_source_dir("arc_easy", "modelscope", "OmniData/ARC")) == (
        "/new/raw/arc_easy/modelscope/OmniData--ARC"
    )


def test_raw_source_match_requires_matching_provenance_and_payload(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / ".lychee_source.json").write_text(
        json.dumps(
            {
                "provider": "huggingface",
                "source_id": "org/dataset",
                "revision": "abc123",
            }
        ),
        encoding="utf-8",
    )
    assert not common.raw_source_matches(
        source,
        provider="huggingface",
        source_id="org/dataset",
        revision="abc123",
    )
    (source / "data.parquet").write_bytes(b"payload")
    assert common.raw_source_matches(
        source,
        provider="huggingface",
        source_id="org/dataset",
        revision="abc123",
    )
    assert not common.raw_source_matches(
        source,
        provider="huggingface",
        source_id="org/dataset",
        revision="different",
    )


def test_result_dir_can_separate_team_and_method(tmp_path):
    path = result_dir(
        "Qwen3.6-27B",
        "none",
        "gaia_validation",
        root=str(tmp_path),
        team="gaia",
    )

    assert path == str(tmp_path / "Qwen3.6-27B/gaia/none/gaia_validation")


def test_copy_raw_to_prepared_creates_real_directory(tmp_path):
    raw = tmp_path / "raw" / "gaia"
    raw.mkdir(parents=True)
    (raw / "metadata.parquet").write_text("fake", encoding="utf-8")
    prepared = tmp_path / "prepared" / "gaia"

    out = common.copy_raw_to_prepared(raw, prepared)

    assert out == prepared
    assert out.is_dir()
    assert (out / "metadata.parquet").read_text(encoding="utf-8") == "fake"


def test_restore_prepared_from_raw_uses_ready_candidate(tmp_path):
    bad_raw = tmp_path / "raw" / "bad"
    good_raw = tmp_path / "raw" / "good"
    bad_raw.mkdir(parents=True)
    good_raw.mkdir(parents=True)
    (good_raw / "ready.txt").write_text("ok", encoding="utf-8")
    prepared = tmp_path / "prepared" / "dataset"

    out = common.restore_prepared_from_raw(
        "toy",
        prepared,
        [
            ("huggingface", "bad/source", bad_raw),
            ("huggingface", "good/source", good_raw),
        ],
        ready=lambda path: (path / "ready.txt").is_file(),
    )

    assert out == prepared
    assert out.is_dir()
    assert (out / "ready.txt").read_text(encoding="utf-8") == "ok"


def test_gsm8k_preparation_uses_only_main_subset_and_repairs_stale_cache(
    tmp_path, monkeypatch
):
    from datasets import Dataset

    raw_root = tmp_path / "raw"
    prepared_root = tmp_path / "prepared"
    monkeypatch.setenv("LYCHEE_BENCHMARK_RAW_ROOT", str(raw_root))
    monkeypatch.setenv("LYCHEE_BENCHMARK_PREPARED_ROOT", str(prepared_root))
    monkeypatch.setenv("LYCHEE_BENCHMARK_CONVERSION_ONLY", "1")
    monkeypatch.setattr(gsm8k, "EXPECTED_SPLIT_ROWS", {"train": 2, "test": 1})

    source = common.raw_source_dir("gsm8k", "modelscope", "AI-ModelScope/gsm8k")
    main = source / "main"
    socratic = source / "socratic"
    main.mkdir(parents=True)
    socratic.mkdir(parents=True)
    Dataset.from_dict(
        {"question": ["main train 1", "main train 2"], "answer": ["#### 1", "#### 2"]}
    ).to_parquet(main / "train-00000-of-00001.parquet")
    Dataset.from_dict(
        {"question": ["main test"], "answer": ["#### 3"]}
    ).to_parquet(main / "test-00000-of-00001.parquet")
    Dataset.from_dict(
        {
            "question": ["socratic train 1", "socratic train 2"],
            "answer": ["#### 1", "#### 2"],
        }
    ).to_parquet(socratic / "train-00000-of-00001.parquet")
    Dataset.from_dict(
        {"question": ["socratic test"], "answer": ["#### 3"]}
    ).to_parquet(socratic / "test-00000-of-00001.parquet")

    stale = prepared_root / "gsm8k" / "main"
    stale.mkdir(parents=True)
    Dataset.from_dict(
        {"question": ["main test", "main test"], "answer": ["#### 3", "#### 3"]}
    ).to_parquet(stale / "test-00000-of-00001.parquet")

    result = gsm8k.prepare_gsm8k(source="modelscope")

    assert result == str(stale)
    assert list(common.load_parquet("gsm8k/main", "train")["question"]) == [
        "main train 1",
        "main train 2",
    ]
    assert list(common.load_parquet("gsm8k/main", "test")["question"]) == ["main test"]


def test_choice_qa_and_locomo_standardizers():
    medqa = choice_qa._standardize_medqa_row(
        {
            "question": "Which finding is most likely?",
            "options": {"A": "Fever", "B": "Cough", "C": "Rash", "D": "Headache"},
            "answer_idx": "C",
        }
    )
    assert medqa["answer"] == "Rash"
    assert medqa["options"][2] == "C. Rash"

    records = locomo10._load_mc10_records(
        [
            {
                "question_id": "conv-1_q1",
                "question_type": "single_hop",
                "question": "Who brought tea?",
                "answer": "Alice",
                "haystack_session_ids": ["session_1"],
                "haystack_session_datetimes": ["2024-01-01T12:00:00"],
                "haystack_sessions": [[{"role": "user", "content": "Alice brought tea."}]],
            }
        ],
        n=1,
    )
    assert records[0]["task"] == "locomo10"
    assert records[0]["gold"] == ["Alice"]
    assert "Alice brought tea" in records[0]["context"]

    original_records = locomo10._load_original_conversation_records(
        [
            {
                "sample_id": "conv-26",
                "conversation": {
                    "session_1_date_time": "1:56 pm on 8 May, 2023",
                    "session_1": [
                        {"speaker": "Caroline", "text": "I went to support group."},
                        {"speaker": "Melanie", "text": "That sounds meaningful."},
                    ],
                },
                "qa": [
                    {
                        "question": "Where did Caroline go?",
                        "answer": "support group",
                        "evidence": ["D1:1"],
                        "category": 2,
                    }
                ],
            }
        ],
        n=1,
        max_qa_per_conv=10,
    )
    assert original_records[0]["gold"] == ["support group"]
    assert "session_1" in original_records[0]["context"]
    assert "Caroline: I went to support group." in original_records[0]["context"]


def test_autogen_runtime_owns_tool_workspace_helpers(tmp_path):
    runtimes = set(REGISTRY.list("runtime"))
    assert "autogen" in runtimes
    assert "autogen_tools" not in runtimes
    source = tmp_path / "source.txt"
    source.write_text("public attachment", encoding="utf-8")
    runtime = AutoGenRuntime(work_root=tmp_path / "work")
    assert runtime.code_timeout == 60
    case_id = "c61d22de-5f6c-4958-a7f6-5e9707bd3466"
    workspace, task_text, copied, workspace_info = runtime._prepare_workspace(
        TaskQuery(
            id=case_id,
            question=f"Read this file.\n\nReferenced file path: {source}",
            context=f"Referenced file path: {source}",
        )
    )
    assert workspace.is_dir()
    assert workspace.name.startswith("ws_")
    assert len(workspace.name) == 35
    assert case_id not in str(workspace)
    assert (workspace / "source.txt").read_text(encoding="utf-8") == "public attachment"
    assert str(source) not in task_text
    assert "source.txt" in task_text
    assert copied == [str(workspace / "source.txt")]
    assert workspace_info == {
        "opaque_workspace_id": workspace.name,
        "workspace_policy": "uuid4_isolated_per_attempt_v1",
        "workspace_is_new": True,
        "preexisting_entry_count": 0,
        "attachment_filename_policy": "preserve_official_filename",
        "visible_attachment_names": ["source.txt"],
    }

    second, _, _, second_info = runtime._prepare_workspace(
        TaskQuery(
            id=case_id,
            question=f"Read this file.\n\nReferenced file path: {source}",
        )
    )
    assert second != workspace
    assert second_info["opaque_workspace_id"] != workspace_info["opaque_workspace_id"]


def test_autogen_workspace_preserves_official_gaia_attachment_name(tmp_path):
    task_id = "32102e3e-d12a-4209-9163-7b3a104efe5d"
    source = tmp_path / f"{task_id}.xlsx"
    source.write_bytes(b"official attachment")
    runtime = AutoGenRuntime(work_root=tmp_path / "work")

    workspace, task_text, copied, info = runtime._prepare_workspace(
        TaskQuery(
            id=task_id,
            question=f"Inspect it.\n\nReferenced file path: {source}",
        )
    )

    assert task_id not in workspace.name
    assert (workspace / source.name).read_bytes() == b"official attachment"
    assert source.name in task_text
    assert copied == [str(workspace / source.name)]
    assert info["attachment_filename_policy"] == "preserve_official_filename"


def test_gaia_scorer_uses_official_normalization():
    assert score("gaia", "Paris", "paris") == 1.0
    assert score("gaia", "$1,234", "1234") == 1.0


def test_bbeh_scorer_matches_official_examples():
    assert bbeh.evaluate_correctness("Ok The final answer is: \\boxed{4}.", "4")
    assert bbeh.evaluate_correctness("Ok The answer is: (A)", "a")
    assert bbeh.evaluate_correctness("Ok The answer is: **25**\nHere's why.", "25.0")
    assert not bbeh.evaluate_correctness("The final answer is: 2, 3, 5", "2,3,4")


def test_hle_multimodal_and_judge_helpers():
    image = hle._image_content({"bytes": b"png"})
    assert image == "data:image/png;base64,cG5n"
    details = hle.judge_score_details(
        {
            "extracted_final_answer": "Paris",
            "reasoning": "It matches.",
            "correct": "yes",
            "confidence": 80,
        }
    )
    assert details["score"] == 1.0
    assert details["confidence"] == 80
    assert round(hle.calibration_error([0.8], [True]), 6) == 0.2


def test_workbench_native_tool_requests_convert_to_official_actions():
    record = {
        "tool_requests": [
            {
                "tool_name": "calendar_create_event",
                "arguments": json.dumps({"title": "Review", "start": "2023-12-01"}),
            }
        ]
    }
    actions = workbench.predicted_actions(
        record, {"calendar_create_event": "calendar.create_event"}
    )
    assert actions == [
        'calendar.create_event.func(title="Review", start="2023-12-01")'
    ]


def test_human_eval_scorer_runs_check_function():
    gold = {
        "entry_point": "add",
        "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n",
    }
    pred = "```python\ndef add(a, b):\n    return a + b\n```"
    assert score("human_eval", pred, gold) == 1.0


def test_strict_mas_scorers():
    assert (
        score(
            "mas_audit",
            '{"decision": "alarm", "mistake_step": 3, "mistake_agent": "Planner"}',
            {"decision": "alarm", "mistake_step": 3, "mistake_agent": "Planner"},
        )
        == 1.0
    )
    assert (
        score(
            "mas_failure_taxonomy",
            '{"labels": ["1.1", "2.3"]}',
            {"labels": ["1.1", "2.3", "3.1"]},
        )
        > 0.0
    )
    assert (
        score(
            "mas_deviation",
            '{"is_deviation": true, "deviation_types": ["wrong_route"]}',
            {"is_deviation": True, "deviation_types": ["wrong_route"]},
        )
        == 1.0
    )
    assert (
        score(
            "mas_context_leakage",
            "Task B completed successfully without mentioning confidential token.",
            {"metric": "CLC", "expected": "Task B completed", "forbidden": ["salary=999"]},
        )
        == 1.0
    )
    assert (
        score(
            "mas_context_leakage",
            "Task B completed. salary=999",
            {"metric": "CLC", "expected": "Task B completed", "forbidden": ["salary=999"]},
        )
        == 0.0
    )
    audit_details = score_details(
        "mas_audit",
        '{"decision": "alarm", "mistake_step": 3, "mistake_agent": "Planner"}',
        {"decision": "alarm", "mistake_step": 3, "mistake_agent": "Planner"},
    )
    assert audit_details["mistake_step_correct"] is True
    collab_details = score_details(
        "mas_context_leakage",
        "Task B completed.",
        {"metric": "CLC", "expected": "Task B completed", "forbidden": ["salary=999"]},
    )
    assert collab_details["leakage_detected"] is False


def test_strict_mas_loader_helpers_are_schema_tolerant(tmp_path):
    rows = aftraj._interleave_rows(
        [{"audit_label": "unsafe", "conv_id": "u1"}],
        [{"audit_label": "safe", "conv_id": "s1"}],
    )
    assert [row["conv_id"] for row in rows] == ["u1", "s1"]
    record = aftraj._to_record(
        {
            "audit_label": "unsafe",
            "query": "Solve the task",
            "messages": [{"agent_name": "Planner", "message": "wrong step"}],
            "mistake_step": "0",
            "mistake_agent": "Planner",
        },
        task_name="aftraj_audit",
    )
    assert record["gold"]["mistake_step"] == 0
    assert "Planner" in record["question"]

    task_dir = tmp_path / "data"
    task_dir.mkdir()
    (task_dir / "TASK-DATAENG-CLC-001.json").write_text(
        json.dumps(
            {
                "task_id": "TASK-DATAENG-CLC-001",
                "task_b_description": "Review the governance policy.",
                "topology": {"type": "chain"},
                "injections": {"private_fact": "salary=999"},
                "expected_outcome": "Review complete",
            }
        ),
        encoding="utf-8",
    )
    collab_rows = agent_collab.load_source_records(metric="CLC", root=tmp_path)
    assert len(collab_rows) == 1
    collab_record = agent_collab._to_record(collab_rows[0], "CLC")
    assert collab_record["kind"] == "mas_context_leakage"
    assert "salary=999" in collab_record["gold"]["forbidden"]

    labels = mast_data.extract_taxonomy_labels({"primary": "1.1", "nested": {"secondary": "2.3"}})
    assert labels == ["1.1", "2.3"]

    run = [
        {
            "run_id": "r1",
            "event_index": 0,
            "event_type": "handoff",
            "agent_name": "Reviewer",
            "is_deviation": "False",
        },
        {
            "run_id": "r1",
            "event_index": 1,
            "event_type": "tool_call",
            "agent_name": "Executor",
            "is_deviation": "true",
            "deviation_label": "wrong_route",
        },
    ]
    assert open_agent_traces._gold(run)["deviation_types"] == ["wrong_route"]
    assert (
        open_agent_traces.deviation_score_details(
            '{"is_deviation": true, "deviation_types": ["wrong_route"]}',
            open_agent_traces._gold(run),
        )["score"]
        == 1.0
    )
