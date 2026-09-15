# Agent Handoff

## Start Here

- Read [the conversation handoff](docs/AGENT_HANDOFF_CN.md) when starting or resuming work in this repository.
- Check the actual Git branch, worktree, machine, and environment before acting. The handoff is a dated snapshot, not proof of the current machine state.
- Follow the latest user request. Read [the next development plan](docs/RECONFIGURABLE_NAVIGATION_NEXT_PLAN_CN.md) for proposed work and [the server migration guide](docs/SERVER_MIGRATION_2X4090_CN.md) before environment setup.
- Treat dated installation and experiment sections as history. The current snapshot in the main handoff takes precedence over older status text in other documents.
- New box-skill policy bundles, state datasets, and videos are stored outside Git. Check the handoff's artifact locations and manifest hashes before attempting reproduction.

## Project Boundaries

- Executable skills are NAV, PUSH, and CLIMB. STOP is a terminal branch; JUMP is not implemented.
- The current planner is a privileged-state, rule-based Oracle. Offline MLP and a single BEV/object Transformer have been trained. The object model selected 56/68 successful test candidates versus the conditional baseline's 55/68 in one run; this small gain does not authorize replacing the planner. Ensemble, multi-step learned planning, VLM and RGB-D remain future work. See [the object-model result](docs/OBJECT_WORLD_MODEL_V2_CN.md).
- The legacy production course clears a stair entrance. The moving-box-to-platform workflow now has an explicit BoxSupportBackend on the shared executor, a scene-specific rule planner, and N1 skill-start snapshots. See [the integration record](docs/BOX_SUPPORT_EXECUTOR_CN.md). Distinguish single-actor results, composite results, and generalization claims.
- Bounded support parking, approach, and landing candidates now execute through that backend; see [the geometry pilot](docs/BOX_SUPPORT_GEOMETRY_CN.md). The planner still uses rules, and the fixed-layout pilot is not a scene-family training split or learned world model.
- Parameterized support layouts, family-grouped data, and optional budgeted rule-policy suffix evaluation are now available; see [the layout and task-label pilot](docs/BOX_SUPPORT_LAYOUTS_CN.md). A successful suffix supplies positive reachability evidence. Failure or budget exhaustion is not proof of unreachability, and the small pilot is not a trained world model.
- Large object-model datasets may use in-memory skill boundaries. `snapshot_id` identifies that boundary; it is not an archive checksum. Only records explicitly marked with `full_snapshot_available` and an archive path support direct snapshot-file replay. Preserve this distinction in data loading and reports.
- Preserve validated baseline actors and their observation, joint mapping, PD, startup, and delay contracts. Derive separate candidates when the task needs new capability; do not retrain an accepted CLIMB merely to smooth its forward-lunge motion.
- Physical PUSH requires fingertip contact and excludes chassis/leg pushing. Keep the legacy low-friction box baseline distinct from the finite-friction hybrid task; neither implies arbitrary box capability.
- The user accepts the current 29/32 composite baseline. Freeze its three selected actors and use them to advance geometric candidates, scene-family data, and the privileged-state world model. Do not make further low-level training, eliminating the three platform failures, or broad capability sweeps a prerequisite.
- Do not use kinematic object relocation as evidence of physical skill execution. Live skill switching must not reset the physical episode.

## Verification and Continuity

- Reuse the existing headless regression scripts listed in the handoff. Historical passes must not be reported as fresh results on another server.
- Keep MuJoCo and Isaac environments isolated. Do not hardcode the previous server paths into a new deployment.
- Reproduce the current box-support pipeline through [the explicit CPU launcher](mujoco/run_box_support_sequence.py). Default CPU kernel dispatch can change contact-rich rollout outcomes; record the numerical backend with the result.
- A server may run file-synchronized source over an older, dirty Git checkout. Inspect it and preserve its changes; do not use an unconditional pull or reset as a synchronization shortcut.
- Preserve uncommitted user changes, and keep secrets, environment directories, and bulk training artifacts out of Git.
- Update the handoff after material changes to validated behavior, deployment prerequisites, or the next work boundary. Distinguish verified facts, hypotheses, and planned work.