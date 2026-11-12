"""Project 4, Milestone 5: deterministic evaluation of a Franka lift policy.

Every robot plays exactly ONE episode, all starting together at t = 0, with exploration
noise OFF (the policy's mean action). Per episode it records:
  - lift success : cube centre above 10 cm for at least 1 s in a row      (original Project 4 target)
  - goal success : cube lifted (> 4 cm) AND within 5 cm of the goal, ever  (same rule as training's success_rate)
  - dropped      : the episode ended through the object_dropping termination

Run inside the box from ~/IsaacLab:
  EVAL_NAME=v5 uv run --extra isaacsim python ~/p4_eval/eval_lift.py \
      --task Custom-Lift-Franka --num_envs 2048 --checkpoint <path/to/model.pt>
"""

import json
import os
import sys

import torch

# Reuse Isaac Lab's own playback backend: argument parsing, checkpoint lookup, env creation, runner classes
from isaaclab_rl.entrypoints.backends import play_rsl_rl as P

# ----- Success definitions (metres / seconds) -----
LIFT_Z = 0.10        # "clearly lifted": cube centre above 10 cm (world z)
LIFT_HOLD_S = 1.0    # ...and kept there for at least 1 s without interruption
GOAL_LIFT_Z = 0.04   # same minimal_height as the training metric
GOAL_TOL = 0.05      # same success_threshold as the training metric (5 cm)


def main() -> None:
    # Parse the same options as `isaaclab play` (--task, --num_envs, --checkpoint, ...)
    args_cli = P._parse_args(sys.argv[1:])
    # Environment + PPO settings registered for the task (play_mode turns off training-only extras such as sensor noise)
    env_cfg, agent_cfg = P.resolve_task_config(args_cli.task, args_cli.agent, play_mode=not args_cli.train_env_cfg)

    with P.launch_simulation(env_cfg, args_cli):  # starts Isaac Sim (headless unless --viz kit is given)
        from isaaclab.utils.math import combine_frame_transforms  # goal: robot frame -> world frame

        agent_cfg = P.cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)  # apply CLI overrides to the PPO settings
        P.apply_env_overrides(args_cli, env_cfg)                       # apply --num_envs etc. to the environment
        env_cfg.seed = agent_cfg.seed                                  # same seed as training (42) -> repeatable

        # Find the checkpoint file exactly the way `isaaclab play` does
        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume_path = P._resolve_checkpoint(args_cli, agent_cfg, env_cfg, log_root_path)
        if resume_path is None:
            print("[EVAL] No checkpoint found.")
            return

        # Match observation normalization to how THIS checkpoint was trained (v5: off, v7/v8: on).
        # The run folder saved its PPO settings in params/agent.yaml when it trained.
        agent_yaml = os.path.join(os.path.dirname(resume_path), "params", "agent.yaml")
        if os.path.isfile(agent_yaml):
            norm = "obs_normalization: true" in open(agent_yaml).read()
            for part in ("actor", "critic"):
                if hasattr(agent_cfg, part) and hasattr(getattr(agent_cfg, part), "obs_normalization"):
                    getattr(agent_cfg, part).obs_normalization = norm
            print(f"[EVAL] obs_normalization from the run's agent.yaml: {norm}")
        else:
            print("[EVAL] WARNING: no params/agent.yaml next to the checkpoint; using current settings")

        # Build the environment and wrap it for rsl_rl (same as the play backend)
        env = P.create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False)
        env = P.RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        # Load the trained network; the inference policy returns the MEAN action (no exploration noise)
        runner = P.OnPolicyRunner(env, P.to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        print(f"[EVAL] Loaded checkpoint: {resume_path}")

        base = env.unwrapped                 # the raw Isaac Lab environment (scene, managers)
        robot = base.scene["robot"]          # Franka articulation
        obj = base.scene["object"]           # the cube
        n, dev = base.num_envs, base.device
        hold_steps = int(round(LIFT_HOLD_S / base.step_dt))  # 1 s / 0.02 s per step = 50 steps

        # Which terminations exist? We need "object_dropping" to count drops.
        term_names = list(base.termination_manager.active_terms)
        has_drop = "object_dropping" in term_names
        print(f"[EVAL] Terminations: {term_names}")

        obs = env.get_observations()
        max_steps = int(base.max_episode_length) + 20  # one full episode plus a small margin

        with torch.inference_mode():  # no gradients needed: faster and lighter on memory
            # ----- Per-episode counters (reset whenever a robot's episode ends) -----
            run = torch.zeros(n, dtype=torch.long, device=dev)         # current streak of steps above LIFT_Z
            best_run = torch.zeros(n, dtype=torch.long, device=dev)    # longest streak this episode
            delivered = torch.zeros(n, dtype=torch.bool, device=dev)   # ever lifted AND within 5 cm of goal
            max_z = torch.zeros(n, device=dev)                         # highest cube centre this episode
            min_dist = torch.full((n,), float("inf"), device=dev)      # closest lifted approach to the goal

            # ----- Results for each robot's FIRST episode -----
            finished = torch.zeros(n, dtype=torch.bool, device=dev)
            r_lift = torch.zeros(n, dtype=torch.bool, device=dev)
            r_goal = torch.zeros(n, dtype=torch.bool, device=dev)
            r_drop = torch.zeros(n, dtype=torch.bool, device=dev)
            r_maxz = torch.zeros(n, device=dev)
            r_mind = torch.full((n,), float("inf"), device=dev)
            r_best = torch.zeros(n, dtype=torch.long, device=dev)   # longest streak above LIFT_Z (steps)
            goal_z = []                                              # (min, max) goal height per step

            for _ in range(max_steps):
                # 1) MEASURE the current state (before stepping, so it belongs to the current episode)
                obj_pos = obj.data.root_pos_w.torch                               # cube centre, world frame
                cmd = base.command_manager.get_command("object_pose")             # goal in robot base frame
                goal_w, _ = combine_frame_transforms(
                    robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, cmd[:, :3]
                )                                                                 # goal in world frame
                dist = torch.linalg.norm(goal_w - obj_pos, dim=1)                 # cube-to-goal distance
                z = obj_pos[:, 2]                                                 # cube height (world z, as in training)
                lifted = z > GOAL_LIFT_Z
                goal_z.append((goal_w[:, 2].min().item(), goal_w[:, 2].max().item()))

                run = torch.where(z > LIFT_Z, run + 1, torch.zeros_like(run))     # extend or break the streak
                best_run = torch.maximum(best_run, run)
                delivered |= lifted & (dist < GOAL_TOL)
                max_z = torch.maximum(max_z, z)
                min_dist = torch.minimum(min_dist, torch.where(lifted, dist, torch.full_like(dist, float("inf"))))

                # 2) STEP every robot with its deterministic action
                obs, _, dones, _ = env.step(policy(obs))
                policy.reset(dones)
                done = dones.bool()
                drop_now = base.termination_manager.get_term("object_dropping").bool() if has_drop else torch.zeros_like(done)

                # 3) RECORD the first finished episode of each robot
                new = done & ~finished
                r_lift[new] = best_run[new] >= hold_steps
                r_goal[new] = delivered[new]
                r_drop[new] = drop_now[new]
                r_maxz[new] = max_z[new]
                r_mind[new] = min_dist[new]
                r_best[new] = best_run[new]
                finished |= new

                # 4) Clear the counters of every robot that just reset
                run[done] = 0
                best_run[done] = 0
                delivered[done] = False
                max_z[done] = 0.0
                min_dist[done] = float("inf")

                if bool(finished.all()):
                    break

        # ----- Summary -----
        m = int(finished.sum())
        pct = lambda t: 100.0 * float(t[finished].float().mean()) if m else 0.0
        finite = torch.isfinite(r_mind) & finished
        res = {
            "checkpoint": resume_path,
            "episodes": m,
            "lift_success_pct": round(pct(r_lift), 1),
            "goal_success_pct": round(pct(r_goal), 1),
            "drop_pct": round(pct(r_drop), 1),
            "never_lifted_pct": round(100.0 * float(((r_maxz <= GOAL_LIFT_Z) & finished).float().sum()) / max(m, 1), 1),
            "reached_10cm_pct": round(pct(r_maxz > LIFT_Z), 1),
            "median_max_height_cm": round(100.0 * float(r_maxz[finished].median()), 1),
            "median_longest_hold_above_10cm_s": round(float(r_best[finished].float().median()) * base.step_dt, 2),
            "goal_height_range_cm": [round(100.0 * min(a for a, _ in goal_z), 1), round(100.0 * max(b for _, b in goal_z), 1)],
            "median_closest_lifted_dist_cm": round(100.0 * float(r_mind[finite].median()), 1) if bool(finite.any()) else None,
        }
        print("\n================ EVALUATION (deterministic, 1 episode per robot) ================")
        for k, v in res.items():
            print(f"  {k:32s} {v}")
        print("=================================================================================\n")

        name = os.environ.get("EVAL_NAME", os.path.basename(os.path.dirname(resume_path)))
        out = os.path.expanduser(f"~/p4_eval/{name}_results.json")
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[EVAL] Saved: {out}")

        P.close_env(env)


if __name__ == "__main__":
    main()
