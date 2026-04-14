#!/usr/bin/env python3
"""팔로워 팔 단독 어드미턴스 제어 + 중력보상 테스트.

리더 팔 없이 팔로워만 구동.
손으로 팔을 밀면 순응하고, 손을 떼면 현재 위치 유지.

실행:
  ros2 launch aloha aloha_bringup.launch.py   # 터미널 1
  python3 admittance_only.py                   # 터미널 2
"""

from __future__ import annotations

import os
import signal
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gin
from run_policy import MotorConfigPaths
import numpy as np
import rclpy
from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    robot_shutdown,
    robot_startup,
)
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

from real_world.aloha.constants import DT, DT_DURATION, START_ARM_POSE, LEADER2FOLLOWER_JOINT_FN
from real_world.aloha.robot_utils import move_arms, move_grippers, torque_on

from minimalist_compliance_control.controller import ControllerConfig
from minimalist_compliance_control.utils import load_merged_motor_config
from policy.compliance import CompliancePolicy
from real_world.real_world_aloha_ros2 import RealWorldAlohaROS2

# ── 모터 로거 상수 ───────────────────────────────────────────
JOINT_NAMES = [
    "waist", "shoulder", "elbow",
    "forearm_roll", "wrist_angle", "wrist_rotate", "gripper",
]
JOINT_MOTOR_TYPE = {
    "waist":        "XM540-W270",
    "shoulder":     "XM540-W270",
    "elbow":        "XM540-W270",
    "forearm_roll": "XM540-W270",
    "wrist_angle":  "XM540-W270",
    "wrist_rotate": "XM430-W350",
    "gripper":      "XM430-W350",
}
MOTOR_KT = {
    "XM540-W270": 2.4091,   # N·m / A
    "XM430-W350": 1.793,    # N·m / A
}
LOG_INTERVAL = 1.0   # 초
LOG_SAVE_DIR = "logs"


# ── EE 로그 버퍼 ─────────────────────────────────────────────
# { site_name: { "x_ref": [], "x_obs": [], "time": [] } }
_ee_log: dict = {}


def _init_ee_log(site_names: list[str]) -> None:
    """site별 EE 로그 버퍼 초기화."""
    global _ee_log
    _ee_log = {
        site: {"x_ref": [], "x_obs": [], "time": []}
        for site in site_names
    }


def _append_ee_log(
    site_names: list[str],
    state_ref,
    x_obs: np.ndarray,
    current_time: float,
) -> None:
    """매 스텝 EE 목표/현재 위치를 버퍼에 저장."""
    if state_ref is None:
        return
    for idx, site in enumerate(site_names):
        if site not in _ee_log:
            continue
        _ee_log[site]["x_ref"].append(
            np.asarray(state_ref.x_ref[idx], dtype=np.float32).copy()
        )
        _ee_log[site]["x_obs"].append(
            np.asarray(x_obs[idx], dtype=np.float32).copy()
        )
        _ee_log[site]["time"].append(float(current_time))


def save_ee_log() -> None:
    """버퍼에 저장된 EE 위치/방향 데이터를 그래프로 저장."""
    os.makedirs(LOG_SAVE_DIR, exist_ok=True)

    POS_LABELS = ["x (m)", "y (m)", "z (m)"]
    ORI_LABELS = ["rx (rad)", "ry (rad)", "rz (rad)"]

    for site, data in _ee_log.items():
        if not data["time"]:
            print(f"[save_ee_log] {site}: 데이터 없음, 스킵")
            continue

        t     = np.array(data["time"])
        t     = t - t[0]
        x_ref = np.array(data["x_ref"])   # (T, 6)
        x_obs = np.array(data["x_obs"])   # (T, 6)

        site_tag = site.replace("/", "_")

        # ── 그래프 1: EE 위치 (x, y, z) ─────────────────────────
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        fig.suptitle(f"{site} — EE Position (m)", fontsize=13)
        for j, ax in enumerate(axes):
            ax.plot(t, x_ref[:, j], label="x_ref (target)",
                    linewidth=1.2, color="tab:blue")
            ax.plot(t, x_obs[:, j], label="x_obs (actual)",
                    linewidth=1.2, color="tab:orange", linestyle="--")
            ax.set_ylabel(POS_LABELS[j])
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(True)
        axes[-1].set_xlabel("time (s)")
        plt.tight_layout()
        path1 = os.path.join(LOG_SAVE_DIR, f"{site_tag}_ee_position.png")
        fig.savefig(path1, dpi=120)
        plt.close(fig)
        print(f"[save_ee_log] 저장: {path1}")

        # ── 그래프 2: EE 방향 (rx, ry, rz) ─────────────────────
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        fig.suptitle(f"{site} — EE Orientation (rad)", fontsize=13)
        for j, ax in enumerate(axes):
            ax.plot(t, x_ref[:, j + 3], label="x_ref (target)",
                    linewidth=1.2, color="tab:blue")
            ax.plot(t, x_obs[:, j + 3], label="x_obs (actual)",
                    linewidth=1.2, color="tab:orange", linestyle="--")
            ax.set_ylabel(ORI_LABELS[j])
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(True)
        axes[-1].set_xlabel("time (s)")
        plt.tight_layout()
        path2 = os.path.join(LOG_SAVE_DIR, f"{site_tag}_ee_orientation.png")
        fig.savefig(path2, dpi=120)
        plt.close(fig)
        print(f"[save_ee_log] 저장: {path2}")

        # ── 그래프 3: 위치 오차 (pos error norm) ─────────────────
        pos_err = np.linalg.norm(x_ref[:, :3] - x_obs[:, :3], axis=1)
        fig, ax = plt.subplots(figsize=(12, 4))
        fig.suptitle(f"{site} — Position Error (m)", fontsize=13)
        ax.plot(t, pos_err, linewidth=1.2, color="tab:red")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("error (m)")
        ax.grid(True)
        plt.tight_layout()
        path3 = os.path.join(LOG_SAVE_DIR, f"{site_tag}_ee_pos_error.png")
        fig.savefig(path3, dpi=120)
        plt.close(fig)
        print(f"[save_ee_log] 저장: {path3}")


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_repo_root(), path)


# def log_motor_state(
#     follower_bot_left:  InterbotixManipulatorXS,
#     follower_bot_right: InterbotixManipulatorXS,
# ) -> None:
#     """팔로워 양팔의 전류(mA)와 토크(N·m)를 출력."""
#     print("\n" + "═" * 65)
#     print(f"  [{time.strftime('%H:%M:%S')}] 팔로워 모터 상태")
#     print("═" * 65)

#     for side, bot in [("LEFT ", follower_bot_left), ("RIGHT", follower_bot_right)]:
#         with bot.core.js_mutex:
#             js = bot.core.joint_states

#         if js is None:
#             print(f"  [{side}] joint_states 없음")
#             continue

#         effort = np.array(js.effort, dtype=np.float32)

#         print(f"\n  [{side} follower]")
#         print(f"  {'관절':<14} {'모터':<14} {'전류(mA)':>10} {'전류(A)':>8} {'토크(N·m)':>10}")
#         print(f"  {'-'*58}")

#         for i, joint in enumerate(JOINT_NAMES):
#             if i >= len(effort):
#                 break
#             motor_type = JOINT_MOTOR_TYPE[joint]
#             kt         = MOTOR_KT[motor_type]
#             cur_mA     = float(effort[i])
#             cur_A      = cur_mA / 1000.0
#             torque     = kt * cur_A
#             print(
#                 f"  {joint:<14} {motor_type:<14}"
#                 f" {cur_mA:>+10.1f}"
#                 f" {cur_A:>+8.4f}"
#                 f" {torque:>+10.4f}"
#             )
#     print()


# ──────────────────────────────────────────────────────────────
# 팔로워 팔 초기화
# ──────────────────────────────────────────────────────────────

def setup_follower_for_admittance(
    bot_left:  InterbotixManipulatorXS,
    bot_right: InterbotixManipulatorXS,
) -> None:
    """팔로워 팔을 어드미턴스 제어용으로 설정하고 START_ARM_POSE로 이동."""

    for bot in [bot_left, bot_right]:
        bot.core.robot_reboot_motors("single", "gripper", True)
        bot.core.robot_set_operating_modes(
             "group",  "arm",     "position"
        )
        bot.core.robot_set_operating_modes(
            "single", "gripper", "current_based_position"
        )
        bot.core.robot_set_motor_registers(
            "single", "gripper", "current_limit", 400
        )
        torque_on(bot)

    print("[admittance] 팔로워 설정 완료. START_ARM_POSE로 이동 중...")
    move_arms(
        [bot_left, bot_right],
        [START_ARM_POSE[:6]] * 2,
        moving_time=4.0,
    )
    from real_world.aloha.constants import FOLLOWER_GRIPPER_JOINT_CLOSE
    move_grippers(
        [bot_left, bot_right],
        [FOLLOWER_GRIPPER_JOINT_CLOSE] * 2,
        moving_time=0.5,
    )
    print("[admittance] 준비 완료. 팔을 밀어보세요. (Ctrl+C 로 종료)")


# ──────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────

def main() -> None:

    gin_path = _resolve("config/aloha.gin")
    print(f"[admittance] gin 로드: {gin_path}")
    gin.parse_config_file(gin_path, skip_unknown=True)

    motor_cfg_paths = MotorConfigPaths()
    if motor_cfg_paths.default_config_path is None:
        motor_cfg_paths.default_config_path = "descriptions/default.yml"
        motor_cfg_paths.robot_config_path   = "descriptions/aloha/robot.yml"
        motor_cfg_paths.motor_config_path   = "descriptions/aloha/motors.yml"

    merged_config = load_merged_motor_config(
        _resolve(str(motor_cfg_paths.default_config_path)),
        _resolve(str(motor_cfg_paths.robot_config_path)),
        _resolve(str(motor_cfg_paths.motor_config_path))
        if motor_cfg_paths.motor_config_path else None,
    )

    xml_path = _resolve(str(ControllerConfig().xml_path))
    print(f"Robot xml 경로: {xml_path}")

    # ── ROS2 초기화 ───────────────────────────────────────────
    rclpy.init()
    node = create_interbotix_global_node("aloha")

    follower_bot_left = InterbotixManipulatorXS(
        robot_model="vx300s", robot_name="follower_left",
        node=node, iterative_update_fk=False,
    )
    follower_bot_right = InterbotixManipulatorXS(
        robot_model="vx300s", robot_name="follower_right",
        node=node, iterative_update_fk=False,
    )

    def _shutdown(sig, frame):
        print("\n[admittance] 종료 중...")
        sys.exit(0)
    signal.signal(signal.SIGINT, _shutdown)

    robot_startup(node)

    setup_follower_for_admittance(follower_bot_left, follower_bot_right)

    # ── ROS2 백엔드 ───────────────────────────────────────────
    sim = RealWorldAlohaROS2(
        node=node,
        follower_bot_left=follower_bot_left,
        follower_bot_right=follower_bot_right,
        xml_path=xml_path,
        merged_config=merged_config,
        control_dt=DT,
    )

    # ── CompliancePolicy 생성 ─────────────────────────────────
    init_obs = sim.get_observation()
    policy = CompliancePolicy(
        name="compliance",
        robot="aloha",
        init_motor_pos=np.asarray(init_obs.motor_pos, dtype=np.float32),
        start_keyboard_listener=False,
        show_help=False,
    )
    policy.is_prepared  = True
    policy.prep_duration = 0.0

    # ── EE 로그 버퍼 초기화 ──────────────────────────────────
    site_names = list(policy.controller.config.site_names)
    _init_ee_log(site_names)

    print("[admittance] 어드미턴스 루프 시작")
    next_tick = time.monotonic()
    last_log  = time.monotonic()
    loop_start = time.monotonic()

    try:
        while rclpy.ok():
            obs    = sim.get_observation()
            action = policy.step(obs, sim)

            mixed_action = obs.motor_pos.copy()
            mixed_action[0:6]  = action[0:6]    # 왼팔 어드미턴스 제어
            mixed_action[7:13] = action[7:13]   # 오른팔 어드미턴스 제어
            # [6], [13]은 obs.motor_pos 그대로 → 현재 그리퍼 위치 유지
            sim.set_motor_target(mixed_action)


            # ── 매 스텝: EE 로그 버퍼에 저장 ────────────────
            state_ref  = policy.controller._last_state
            x_obs_ee   = policy.controller.get_x_obs()
            current_time = time.monotonic() - loop_start
            _append_ee_log(site_names, state_ref, x_obs_ee, current_time)

            # ── 1초마다 콘솔 출력 ─────────────────────────────
            now = time.monotonic()
            if now - last_log >= LOG_INTERVAL:
                last_log = now

                # log_motor_state(follower_bot_left, follower_bot_right)
                # ── DEBUG: policy 내부 상태 확인 ──────────────────────
                last_state = policy.controller._last_state
                if last_state is not None:
                    print(f"[DBG policy] x_ref[0][:3]={np.array(last_state.x_ref[0])[:3]}")
                    print(f"[DBG policy] x_ref[1][:3]={np.array(last_state.x_ref[1])[:3]}")
                print(f"[DBG policy] pose_command=\n{policy.pose_command}")
                print(f"[DBG policy] base_pose_command=\n{policy.base_pose_command}")
                # ──────────────────────────────────────────────────────
                
            next_tick += DT
            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_tick = time.monotonic()

    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        print("[admittance] 정리 중...")
        # ── EE 그래프 저장 ────────────────────────────────────
        try:
            save_ee_log()
        except Exception as e:
            print(f"[save_ee_log 오류] {e}")
            import traceback
            traceback.print_exc()

        policy.close()
        sim.close()
        robot_shutdown(node)
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
