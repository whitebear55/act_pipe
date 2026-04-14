#!/usr/bin/env python3
"""어드미턴스 제어가 적용된 ALOHA 원격조종.

dual_side_teleop.py + MCC CompliancePolicy를 통합.

실행:
  ros2 launch aloha aloha_bringup.launch.py   # 터미널 1 (기존과 동일)
  python3 compliance_teleop.py                 # 터미널 2 (이 파일)

흐름:
  리더 팔 FK → xdes (목표 EE 위치)
      ↓
  CompliancePolicy (어드미턴스 + 외력 추정)
      ↓
  팔로워 팔 position 명령
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
    leader_bot_left:    InterbotixManipulatorXS,
    leader_bot_right:   InterbotixManipulatorXS,
) -> None:
    """모든 팔을 시작 위치로 이동."""
    # ── 팔로워: current_based_position 모드 (effort = 전류[mA]) ──
    # 어드미턴스 제어를 위해 현재 전류를 읽어야 하므로
    # arm도 current_based_position으로 설정
    for bot in [follower_bot_left, follower_bot_right]:
        bot.core.robot_reboot_motors("single", "gripper", True)
        bot.core.robot_set_operating_modes("group",  "arm",     "position")
        bot.core.robot_set_operating_modes("single", "gripper", "current_based_position")
        bot.core.robot_set_motor_registers("single", "gripper", "current_limit", 300)

    # ── 리더: position 모드 ──
    for bot in [leader_bot_left, leader_bot_right]:
        bot.core.robot_set_operating_modes("group",  "arm",     "position")
        bot.core.robot_set_operating_modes("single", "gripper", "position")

    # 토크 ON
    for bot in [follower_bot_left, follower_bot_right,
                leader_bot_left,  leader_bot_right]:
        torque_on(bot)

    # START_ARM_POSE로 이동
    start_arm = START_ARM_POSE[:6]
    move_arms(
        [follower_bot_left, follower_bot_right, leader_bot_left, leader_bot_right],
        [start_arm] * 4,
        moving_time=4.0,
    )
    move_grippers(
        [leader_bot_left, follower_bot_left, leader_bot_right, follower_bot_right],
        [LEADER_GRIPPER_JOINT_MID, FOLLOWER_GRIPPER_JOINT_CLOSE] * 2,
        moving_time=0.5,
    )


def press_to_start(
    leader_bot_left:  InterbotixManipulatorXS,
    leader_bot_right: InterbotixManipulatorXS,
    use_gravity_comp: bool,
) -> None:
    """그리퍼를 쥐면 텔레옵 시작."""
    leader_bot_left.core.robot_torque_enable("single", "gripper", False)
    leader_bot_right.core.robot_torque_enable("single", "gripper", False)
    print("[compliance_teleop] 리더 그리퍼를 쥐면 시작합니다...")

    pressed = False
    while rclpy.ok() and not pressed:
        pressed = (
            get_arm_gripper_positions(leader_bot_left)  < LEADER_GRIPPER_CLOSE_THRESH
            and get_arm_gripper_positions(leader_bot_right) < LEADER_GRIPPER_CLOSE_THRESH
        )
        get_interbotix_global_node().get_clock().sleep_for(DT_DURATION)

    if use_gravity_comp:
        enable_gravity_compensation(leader_bot_left)
        enable_gravity_compensation(leader_bot_right)
    else:
        torque_off(leader_bot_left)
        torque_off(leader_bot_right)

    print("[compliance_teleop] 시작!")


# ──────────────────────────────────────────────────────────────
# 리더 팔 → xdes 계산
# ──────────────────────────────────────────────────────────────

def get_leader_xdes(
    leader_bot_left:  InterbotixManipulatorXS,
    leader_bot_right: InterbotixManipulatorXS,
    compliance_policy: CompliancePolicy,
    sim: Any,
) -> np.ndarray:
    """리더 관절각도 → 팔로워 MuJoCo FK → EE 목표 위치.

    WX250s(리더)와 VX300s(팔로워)는 기구학이 다르므로,
    리더 관절각도를 팔로워 MuJoCo 모델에 입력하여 EE 위치를 계산.

    흐름:
      리더 joint_states (6DOF)
          ↓
      sim.motor_ordering 형식으로 변환
          ↓
      sim.get_qpos() → MuJoCo qpos
          ↓
      controller.sync_qpos() + get_x_obs()  ← 팔로워 FK 사용
          ↓
      xdes: (2, 6)  [left, right] × [x, y, z, rx, ry, rz]

    ※ controller.sync_qpos()는 이후 policy.step() 내부의
      controller.step(qpos=follower_qpos)에서 팔로워 qpos로
      덮어쓰이므로 안전하게 호출 가능.

    Returns:
        xdes: (2, 6) array [left_xdes, right_xdes]
              각 행: [x, y, z, rx, ry, rz] (rotation vector)
    """
    # 1. 리더 팔 관절각도 읽기 (thread-safe)
    with leader_bot_left.core.js_mutex:
        left_js = leader_bot_left.core.joint_states
    with leader_bot_right.core.js_mutex:
        right_js = leader_bot_right.core.joint_states

    if left_js is None or right_js is None:
        # joint_states 미수신 → 이전 x_ref 유지
        if compliance_policy.controller._last_state is not None:
            return np.asarray(
                compliance_policy.controller._last_state.x_ref, dtype=np.float32
            ).copy()
        return np.zeros((2, 6), dtype=np.float32)

    left_pos  = np.array(left_js.position[:6],  dtype=np.float32)
    right_pos = np.array(right_js.position[:6], dtype=np.float32)

    # 2. sim.motor_ordering 형식으로 leader 관절각도 구성
    #    리더(WX250s)와 팔로워(VX300s)는 관절각도가 직접 매핑되므로
    #    팔로워 motor_ordering 기준으로 구성
    n = len(sim.motor_ordering)
    leader_motor_pos = np.zeros(n, dtype=np.float32)
    for i, name in enumerate(sim.motor_ordering):
        if name.startswith("left_"):
            joint = name[len("left_"):]
            if joint in JOINT_ORDER:
                idx = JOINT_ORDER.index(joint)
                if idx < len(left_pos):
                    leader_motor_pos[i] = left_pos[idx]
        elif name.startswith("right_"):
            joint = name[len("right_"):]
            if joint in JOINT_ORDER:
                idx = JOINT_ORDER.index(joint)
                if idx < len(right_pos):
                    leader_motor_pos[i] = right_pos[idx]

    # 3. 팔로워 MuJoCo qpos 형식으로 변환
    leader_qpos = sim.get_qpos(leader_motor_pos)

    # 4. 팔로워 MuJoCo 모델에 리더 관절각도를 입력하여 FK 계산
    compliance_policy.controller.sync_qpos(leader_qpos)
    xdes = compliance_policy.controller.get_x_obs().copy()  # (2, 6)

    return xdes


# ──────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────

def signal_handler(sig, frame, leader_bot_left, leader_bot_right):
    print("\n[compliance_teleop] Ctrl+C 감지 → 종료")
    disable_gravity_compensation(leader_bot_left)
    disable_gravity_compensation(leader_bot_right)
    sys.exit(0)


def main(args: dict) -> None:
    gin_file       = args.get("gin_file", "config/aloha.gin")
    use_gravity_comp = args.get("gravity_compensation", False)

    # ── Gin 설정 로드 ─────────────────────────────────────────-
    gin.parse_config_file(gin_file, skip_unknown=True)

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
    leader_bot_left = InterbotixManipulatorXS(
        robot_model="wx250s", robot_name="leader_left",
        node=node, iterative_update_fk=False,
    )
    leader_bot_right = InterbotixManipulatorXS(
        robot_model="wx250s", robot_name="leader_right",
        node=node, iterative_update_fk=False,
    )

    signal.signal(
        signal.SIGINT,
        partial(signal_handler,
                leader_bot_left=leader_bot_left,
                leader_bot_right=leader_bot_right),
    )

    robot_startup(node)
    disable_gravity_compensation(leader_bot_left)
    disable_gravity_compensation(leader_bot_right)

    # ── 초기 자세 이동 ────────────────────────────────────────
    opening_ceremony(
        follower_bot_left, follower_bot_right,
        leader_bot_left,   leader_bot_right,
    )
    press_to_start(leader_bot_left, leader_bot_right, use_gravity_comp)

    # ── ROS2 백엔드 생성 ──────────────────────────────────────
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
    print("[compliance_teleop] 어드미턴스 텔레옵 루프 시작")

    next_tick  = time.monotonic()
    loop_start = time.monotonic()

    try:
        while rclpy.ok():

            # 1. 리더 팔 → xdes 계산 후 policy에 주입
            xdes = get_leader_xdes(leader_bot_left, leader_bot_right, policy, sim)
            # pose_command를 리더 팔 위치로 업데이트
            # (base_pose_command는 alignment 후 고정, pose_command만 매 루프 갱신)
            if policy._alignment_applied:
                policy.pose_command = xdes.copy()
                policy.base_pose_command = xdes.copy()

            # 2. 팔로워 상태 읽기
            obs = sim.get_observation()

            # 3. 어드미턴스 제어 스텝 (motor position 반환)
            action = policy.step(obs, sim)

            # 4. 팔로워 팔 명령
            # action[6/13]은 MuJoCo 내부 좌표(~0.022)로 모터 단위와 다름 → 그리퍼만 오버라이드
            mixed_action = action.copy()
            mixed_action[6]  = LEADER2FOLLOWER_JOINT_FN(
                leader_bot_left.core.joint_states.position[6]
            )
            mixed_action[13] = LEADER2FOLLOWER_JOINT_FN(
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
                next_tick = time.monotonic()  # 지연 시 리셋

    except KeyboardInterrupt:
        pass
    finally:
        print("[compliance_teleop] 종료 중...")
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
    parser = argparse.ArgumentParser(description="어드미턴스 제어 텔레옵")
    parser.add_argument(
        "--gin-file",
        type=str,
        default="config/aloha.gin",
        help="gin 설정 파일 경로",
    )
    parser.add_argument(
        "-g", "--gravity-compensation",
        action="store_true",
        help="리더 팔 중력 보상 활성화",
    )
    args = parser.parse_args()
    main({
        "gin_file":             args.gin_file,
        "gravity_compensation": args.gravity_compensation,
    })
