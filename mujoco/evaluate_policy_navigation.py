"""Evaluate policy-driven NAV across randomized blocked-passage scenes."""

from __future__ import annotations

import argparse

from reconfigurable_navigation.env import BlockedPassageEnv
from reconfigurable_navigation.locomotion_runtime import LocomotionRuntime
from visualize_policy_navigation import run_navigation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--action-clip", type=float, default=20.0)
    parser.add_argument("--min-success-rate", type=float, default=0.95)
    args = parser.parse_args()
    if args.seeds <= 0:
        parser.error("--seeds must be positive")
    if args.timeout <= 0.0:
        parser.error("--timeout must be positive")
    if args.action_clip <= 0.0:
        parser.error("--action-clip must be positive")
    if not 0.0 <= args.min_success_rate <= 1.0:
        parser.error("--min-success-rate must be between 0 and 1")

    successes = 0
    for seed in range(args.seeds):
        print(f"\n=== NAV episode seed={seed} ===")
        env = BlockedPassageEnv()
        env.reset(seed=seed)
        runtime = LocomotionRuntime(env, action_clip=args.action_clip)
        successes += int(
            run_navigation(
                env,
                runtime,
                viewer=None,
                timeout=args.timeout,
                realtime=False,
            )
        )

    success_rate = successes / args.seeds
    print(
        f"\nNAV evaluation: {successes}/{args.seeds} succeeded "
        f"({success_rate:.1%}); required {args.min_success_rate:.1%}"
    )
    raise SystemExit(0 if success_rate >= args.min_success_rate else 1)


if __name__ == "__main__":
    main()