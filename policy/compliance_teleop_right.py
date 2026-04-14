#!/usr/bin/env python3
"""어드미턴스 제어가 적용된 ALOHA 원격조종 — 오른팔 단독 테스트.

compliance_teleop.py와 동일하지만 오른쪽 팔(follower_right / leader_right)만 제어.
왼팔(follower_left)은 START_ARM_POSE에 고정.

실행:
  ros2 launch aloha aloha_bringup.launch.py   # 터미널 1
  python3 compliance_teleop_right.py           # 터미널 2

흐름:
  오른쪽 리더 FK → xdes[1] (오른팔 목표 EE 위치)
      ↓
  CompliancePolicy (어드미턴스 + 외력 추정)
      ↓
  오른쪽 팔로워 position 명령  (왼팔은 현재 위치 유지)
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from functools import partial
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gin
import numpy as np
import rclpy
from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    get_interbotix_global_node,
    robot_shutdown,
    robot_startup,
)
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

from aloha.constants import (
    DT,
    DT_DURATION,
    FOLLOWER_GRIPPER_JOINT_CLOSE,
    LEADER2FOLLOWER_JOINT_FN,
    LEADER_GRIPPER_CLOSE_THRESH,
    LEADER_GRIPPER_JOINT_MID,
    START_ARM_POSE,
)
from aloha.robot_utils import (
    disable_gravity_compensation,
    enable_gravity_compensation,
    get_arm_gripper_positions,
    move_arms,
    move_grippers,
    torque_off,
    torque_on,
)

# MCC 임포트
from minimalist_compliance_control.controller import ControllerConfig
from minimalist_compliance_control.utils import load_merged_motor_config
from policy.compliance import CompliancePolicy
from real_world.real_world_aloha_ros2 import RealWorldAlohaROS2, JOINT_ORDER


# ──────────────────────────────────────────────────────────────
# EE 로그 설정
# ──────────────────────────────────────────────────────────────

LOG_SAVE_DIR = "logs"

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


# ──────────────────────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────────────────────

def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_repo_root(), path)


# ──────────────────────────────────────────────────────────────
# 초기화
# ──────────────────────────────────────────────────────────────

def opening_ceremony(
    follower_bot_left:  InterbotixManipulatorXS,
    follower_bot_right: InterbotixManipulatorXS,
    leader_bot_right:   InterbotixManipulatorXS,
) -> None:
    """팔을 시작 위치로 이동. 왼팔 팔로워는 position 고정, 오른쪽만 텔레옵."""
    # ── 팔로워 설정 ──────────────────────────────────────────────
    for bot in [follower_bot_left, follower_bot_right]:
        bot.core.robot_reboot_motors("single", "gripper", True)
        bot.core.robot_set_operating_modes("group",  "arm",     "position")
        bot.core.robot_set_operating_modes("single", "gripper", "current_based_position")
        bot.core.robot_set_motor_registers("single", "gripper", "current_limit", 300)

    # ── 오른쪽 리더 설정 ─────────────────────────────────────────
    leader_bot_right.core.robot_set_operating_modes("group",  "arm",     "position")
    leader_bot_right.core.robot_set_operating_modes("single", "gripper", "position")

    # 토크 ON
    for bot in [follower_bot_left, follower_bot_right, leader_bot_right]:
        torque_on(bot)

    # START_ARM_POSE로 이동
    start_arm = START_ARM_POSE[:6]
    move_arms(
        [follower_bot_left, follower_bot_right, leader_bot_right],
        [start_arm] * 3,
        moving_time=4.0,
    )
    move_grippers(
        [follower_bot_left, follower_bot_right],
        [FOLLOWER_GRIPPER_JOINT_CLOSE] * 2,
        moving_time=0.5,
    )
    move_grippers(
        [leader_bot_right],
        [LEADER_GRIPPER_JOINT_MID],
        moving_time=0.5,
    )


def press_to_start(
    leader_bot_right: InterbotixManipulatorXS,
    use_gravity_comp: bool,
) -> None:
    """오른쪽 리더 그리퍼를 쥐면 텔레옵 시작."""
    leader_bot_right.core.robot_torque_enable("single", "gripper", False)
    print("[compliance_teleop_right] 오른쪽 리더 그리퍼를 쥐면 시작합니다...")

    pressed = False
    while rclpy.ok() and not pressed:
        pressed = get_arm_gripper_positions(leader_bot_right) < LEADER_GRIPPER_CLOSE_THRESH
        get_interbotix_global_node().get_clock().sleep_for(DT_DURATION)

    if use_gravity_comp:
        enable_gravity_compensation(leader_bot_right)
    else:
        torque_off(leader_bot_right)

    print("[compliance_teleop_right] 시작!")


# ──────────────────────────────────────────────────────────────
# 오른쪽 리더 팔 → xdes 계산
# ──────────────────────────────────────────────────────────────

def get_right_leader_xdes(
    leader_bot_right:  InterbotixManipulatorXS,
    compliance_policy: CompliancePolicy,
    sim: Any,
) -> np.ndarray:
    """오른쪽 리더 관절각도 → 팔로워 MuJoCo FK → 오른팔 EE 목표 위치.

    Returns:
        xdes: (2, 6) array.
              xdes[0] (왼팔) = 현재 _last_state.x_ref[0] 그대로 유지
              xdes[1] (오른팔) = 리더 FK 결과
    """
    # 이전 x_ref를 기본값으로 사용 (왼팔 고정 유지)
    if compliance_policy.controller._last_state is not None:
        xdes = np.asarray(
            compliance_policy.controller._last_state.x_ref, dtype=np.float32
        ).copy()
    else:
        xdes = np.zeros((2, 6), dtype=np.float32)

    # 오른쪽 리더 관절각도 읽기 (thread-safe)
    with leader_bot_right.core.js_mutex:
        right_js = leader_bot_right.core.joint_states

    if right_js is None:
        return xdes  # joint_states 미수신 → 이전 값 유지

    right_pos = np.array(right_js.position[:6], dtype=np.float32)

    # sim.motor_ordering 형식으로 오른팔 관절각도 구성
    # 왼팔은 0으로 채워도 되나, FK에서 오른팔만 관심 있으므로
    # 현재 팔로워 왼팔 위치로 채워 MuJoCo 상태를 유지
    n = len(sim.motor_ordering)
    leader_motor_pos = np.zeros(n, dtype=np.float32)
    for i, name in enumerate(sim.motor_ordering):
        if name.startswith("right_"):
            joint = name[len("right_"):]
            if joint in JOINT_ORDER:
                idx = JOINT_ORDER.index(joint)
                if idx < len(right_pos):
                    leader_motor_pos[i] = right_pos[idx]

    # 팔로워 MuJoCo qpos 형식으로 변환 후 FK 계산
    leader_qpos = sim.get_qpos(leader_motor_pos)
    compliance_policy.controller.sync_qpos(leader_qpos)
    fk_result = compliance_policy.controller.get_x_obs()  # (2, 6)

    # 오른팔 EE만 업데이트 (index 1)
    xdes[1] = fk_result[1].copy()

    return xdes


# ──────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────

def signal_handler(sig, frame, leader_bot_right):
    print("\n[compliance_teleop_right] Ctrl+C 감지 → 종료")
    disable_gravity_compensation(leader_bot_right)
    sys.exit(0)


def main(args: dict) -> None:
    gin_file         = args.get("gin_file", "config/aloha.gin")
    use_gravity_comp = args.get("gravity_compensation", False)

    # ── Gin 설정 로드 ──────────────────────────────────────────
    gin_path = _resolve(gin_file)
    print(f"[compliance_teleop_right] gin 로드: {gin_path}")
    gin.parse_config_file(gin_path, skip_unknown=True)

    from run_policy import MotorConfigPaths
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
    # 오른쪽 리더만 사용
    leader_bot_right = InterbotixManipulatorXS(
        robot_model="wx250s", robot_name="leader_right",
        node=node, iterative_update_fk=False,
    )

    signal.signal(
        signal.SIGINT,
        partial(signal_handler, leader_bot_right=leader_bot_right),
    )

    robot_startup(node)
    disable_gravity_compensation(leader_bot_right)

    # ── 초기 자세 이동 ────────────────────────────────────────
    opening_ceremony(follower_bot_left, follower_bot_right, leader_bot_right)
    press_to_start(leader_bot_right, use_gravity_comp)

    # ── ROS2 백엔드 생성 ──────────────────────────────────────
    # RealWorldAlohaROS2는 양팔 팔로워가 필요하므로 둘 다 전달
    sim = RealWorldAlohaROS2(
        node=node,
        follower_bot_left=follower_bot_left,
        follower_bot_right=follower_bot_right,
        xml_path=xml_path,
        merged_config=merged_config,
        control_dt=DT,
    )

    # ── CompliancePolicy 생성 ─────────────────────────────────
    init_obs = sim.get_observation(retries=-1)
    policy = CompliancePolicy(
        name="compliance",
        robot="aloha",
        init_motor_pos=np.asarray(init_obs.motor_pos, dtype=np.float32),
    )

    # ── EE 로그 버퍼 초기화 ──────────────────────────────────
    site_names = list(policy.controller.config.site_names)
    _init_ee_log(site_names)

    # ── 텔레옵 루프 ───────────────────────────────────────────
    print("[compliance_teleop_right] 어드미턴스 텔레옵 루프 시작 (오른팔 단독)")

    next_tick  = time.monotonic()
    loop_start = time.monotonic()

    try:
        while rclpy.ok():

            # 1. 오른쪽 리더 → 오른팔 xdes 계산
            #    xdes[0] (왼팔) = 이전 x_ref 유지 (움직이지 않음)
            #    xdes[1] (오른팔) = 리더 FK 결과
            xdes = get_right_leader_xdes(leader_bot_right, policy, sim)

            # pose_command 업데이트: 오른팔(index 1)만 갱신
            if policy._alignment_applied:
                policy.pose_command[1]      = xdes[1].copy()
                policy.base_pose_command[1] = xdes[1].copy()
                # 왼팔(index 0)은 alignment 시 설정된 초기값 유지

            # 2. 팔로워 상태 읽기
            obs = sim.get_observation()

            # 3. 어드미턴스 제어 스텝
            action = policy.step(obs, sim)

            # 4. 팔로워 팔 명령
            #    왼팔(0~6): obs.motor_pos 그대로 유지 (고정)
            #    오른팔(7~12): 어드미턴스 제어 적용
            #    오른팔 그리퍼(13): 리더 추종
            mixed_action = obs.motor_pos.copy()       # 왼팔 고정 기본값
            mixed_action[7:13] = action[7:13]         # 오른팔 어드미턴스
            mixed_action[13]   = LEADER2FOLLOWER_JOINT_FN(
                leader_bot_right.core.joint_states.position[6]
            )
            sim.set_motor_target(mixed_action)

            # 4-1. EE 로그 기록
            _append_ee_log(
                site_names,
                policy.controller._last_state,
                policy.controller.get_x_obs(),
                time.monotonic() - loop_start,
            )

            # 5. 주기 맞추기 (50Hz)
            next_tick += DT
            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_tick = time.monotonic()

    except KeyboardInterrupt:
        pass
    finally:
        print("[compliance_teleop_right] 종료 중...")
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
    parser = argparse.ArgumentParser(description="어드미턴스 제어 텔레옵 — 오른팔 단독")
    parser.add_argument(
        "--gin-file",
        type=str,
        default="config/aloha.gin",
        help="gin 설정 파일 경로",
    )
    parser.add_argument(
        "-g", "--gravity-compensation",
        action="store_true",
        help="오른쪽 리더 팔 중력 보상 활성화",
    )
    args = parser.parse_args()
    main({
        "gin_file":             args.gin_file,
        "gravity_compensation": args.gravity_compensation,
    })
