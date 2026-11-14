from lychee_mas.methods.postrun.counterfactual import (
    ArithmeticReplayTask,
    RepairProposal,
    locate_first_effective_fix,
    parse_repair,
)


def test_counterfactual_replay_finds_first_effective_step():
    task = ArithmeticReplayTask()
    spec = task.make_task("q", 4, 6, 3)
    failed = task.run(spec, {2: "CORRUPTED_OUTPUT: 999"})
    result = locate_first_effective_fix(
        task, failed, lambda _prompt, index, _trace: task.oracle_repair(spec, index)
    )
    assert not result.baseline_correct
    assert result.first_effective_step == 2
    assert result.responsible_agent == "verifier"
    assert result.tried_steps == [0, 1, 2]
    assert task.validate(result.trace)


def test_replay_preserves_later_failure_until_repaired():
    task = ArithmeticReplayTask()
    spec = task.make_task("q", 2, 3, 4)
    failed = task.run(spec, {1: "CORRUPTED_OUTPUT: 999"})
    replay = task.run(
        spec, {**failed.meta["overrides"], 0: task.oracle_repair(spec, 0).replacement}
    )
    assert not task.validate(replay)
    replay = task.run(
        spec, {**failed.meta["overrides"], 1: task.oracle_repair(spec, 1).replacement}
    )
    assert task.validate(replay)


def test_parse_repair_json_contract():
    proposal = parse_repair('{"replacement":"SOLVER_RESULT: 20","reason":"fixed","confidence":0.8}')
    assert proposal == RepairProposal("SOLVER_RESULT: 20", "fixed", 0.8)
