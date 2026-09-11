# Agent Handoff

## Start Here

- Read [the conversation handoff](docs/AGENT_HANDOFF_CN.md) when starting or resuming work in this repository.
- Check the actual Git branch, worktree, machine, and environment before acting. The handoff is a dated snapshot, not proof of the current machine state.
- Follow the latest user request. Read [the next development plan](docs/RECONFIGURABLE_NAVIGATION_NEXT_PLAN_CN.md) for proposed work and [the server migration guide](docs/SERVER_MIGRATION_2X4090_CN.md) before environment setup.

## Project Boundaries

- Executable skills are NAV, PUSH, and CLIMB. STOP is a terminal branch; JUMP is not implemented.
- The current planner is a privileged-state, rule-based Oracle. A learned world model, VLM proposals, and RGB-D inference are planned, not implemented.
- The accepted complex course clears a box from a stair entrance. It does not yet move a box into place as a support for climbing a higher platform.
- Preserve the validated locomotion and CLIMB actors and their observation, joint mapping, PD, startup, and delay semantics. The user accepts the CLIMB forward-lunge motion; do not retrain merely to smooth it.
- Physical PUSH requires fingertip contact and excludes chassis/leg pushing. Its current low-friction box results do not prove performance on ordinary high-friction boxes.
- Do not use kinematic object relocation as evidence of physical skill execution. Live skill switching must not reset the physical episode.

## Verification and Continuity

- Reuse the existing headless regression scripts listed in the handoff. Historical passes must not be reported as fresh results on another server.
- Keep MuJoCo and Isaac environments isolated. Do not hardcode the previous server paths into a new deployment.
- Preserve uncommitted user changes, and keep secrets, environment directories, and bulk training artifacts out of Git.
- Update the handoff after material changes to validated behavior, deployment prerequisites, or the next work boundary. Distinguish verified facts, hypotheses, and planned work.