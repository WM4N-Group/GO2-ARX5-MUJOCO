"""Budgeted rule-policy continuations after physical candidate execution."""

from ..skills.base import SkillStatus
from .snapshot import _rng_scope


def evaluate_suffix(snapshot, *, max_control_steps=3000, max_skills=8):
    if any(type(value) is not int or value < 0 for value in (max_control_steps, max_skills)):
        raise ValueError("Suffix budgets must be nonnegative integers")
    with _rng_scope(snapshot._random_state):
        branch = snapshot.fork()
        started_at = float(branch.env.data.time)
        control_steps = 0
        records = []

        def finish(success, reason):
            success = None if success is None else bool(success)
            return {
                "oracle_task_success": success,
                "reachability": True if success is True else None,
                "reason": reason,
                "suffix_sim_time": float(branch.env.data.time - started_at),
                "suffix_control_steps": control_steps,
                "suffix_records": records,
            }

        def within_budget(_action, skill, _observation):
            nonlocal control_steps
            control_steps += 1
            return control_steps < max_control_steps or skill.status != SkillStatus.RUNNING

        while True:
            observation = branch.env.observe()
            failure = branch.safety.observation_failure(observation)
            if failure is not None:
                return finish(False, failure)
            decision = branch.replanner.decide(observation)
            if decision.terminal:
                return finish(decision.succeeded, decision.reason)
            if control_steps >= max_control_steps or len(records) >= max_skills:
                return finish(None, "suffix_budget_exhausted")
            action = decision.action
            skill = branch._create_skill(action)
            if skill is None or not skill.can_execute(observation, action):
                return finish(False, "suffix_action_rejected")
            skill.reset(action)
            before = float(branch.env.data.time)
            status, steps, interrupted = branch._execute_skill(action, skill, within_budget, None)
            records.append({
                "skill": action.skill.name, "target_pose": action.target_pose.tolist(),
                "object_id": action.object_id, "support_id": action.support_id,
                "status": status.value, "skill_steps": steps,
                "elapsed_sim_time": float(branch.env.data.time - before),
                "failure_reason": None if interrupted else skill.failure_reason,
            })
            if interrupted:
                return finish(None, "suffix_budget_exhausted")
            if status == SkillStatus.FAILED:
                return finish(False, "suffix_skill_failed")


def candidate_continuation(replay, *, max_control_steps=3000, max_skills=8):
    if any(type(value) is not int or value < 0 for value in (max_control_steps, max_skills)):
        raise ValueError("Suffix budgets must be nonnegative integers")
    transition = replay.transition
    result = {
        "schema_version": 1, "policy": "rule_oracle_stop_on_failure",
        "attempted": False, "oracle_task_success": None, "reachability": None,
        "reason": "candidate_rejected", "suffix_sim_time": 0.0,
        "suffix_control_steps": 0, "suffix_records": [],
        "budget": {"max_control_steps": max_control_steps, "max_skills": max_skills},
    }
    prefix_time = None if transition is None else float(transition.ended_at - transition.started_at)
    if transition is not None:
        if transition.interrupted:
            result["reason"] = "candidate_budget_exhausted"
        elif transition.status == SkillStatus.FAILED:
            result.update(oracle_task_success=False, reason="candidate_failed")
        elif replay.next_snapshot is not None:
            result.update(evaluate_suffix(replay.next_snapshot, max_control_steps=max_control_steps, max_skills=max_skills))
            result["attempted"] = True
        else:
            raise ValueError("A completed candidate requires a successor snapshot")
    result["candidate_sim_time"] = prefix_time
    result["total_sim_time"] = None if prefix_time is None else prefix_time + result["suffix_sim_time"]
    result["label_validity"] = {
        "oracle_task_success": result["oracle_task_success"] is not None,
        "reachability": result["reachability"] is True,
        "total_cost": result["oracle_task_success"] is not None,
    }
    result["cost_is_lower_bound"] = prefix_time is not None and result["oracle_task_success"] is None
    return result