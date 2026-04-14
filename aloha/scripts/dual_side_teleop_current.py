#!/usr/bin/env python3

import argparse
import os
import signal
from functools import partial

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aloha.constants import (
    DT_DURATION,
    FOLLOWER_GRIPPER_JOINT_CLOSE,
    LEADER2FOLLOWER_JOINT_FN,
    LEADER_GRIPPER_CLOSE_THRESH,
    LEADER_GRIPPER_JOINT_MID,
    START_ARM_POSE,
)
import numpy as np
import time
from aloha.robot_utils import (
    enable_gravity_compensation,
    disable_gravity_compensation,
    get_arm_gripper_positions,
    move_arms,
    move_grippers,
    torque_off,
    torque_on,
)
from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    get_interbotix_global_node,
    robot_shutdown,
    robot_startup,
)
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand
import rclpy

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


# ── 로그 버퍼 ────────────────────────────────────────────────
# 구조: { "left": { "current": [], "torque": [], "pos": [], "time": [] }, "right": ... }
_log: dict = {
    side: {
        "current": [],   # (T, 7) mA
        "torque":  [],   # (T, 7) N·m
        "pos":     [],   # (T, 7) rad
        "time":    [],   # (T,)
    }
    for side in ("left", "right")
}


def _read_and_append_log(
    follower_bot_left:  InterbotixManipulatorXS,
    follower_bot_right: InterbotixManipulatorXS,
) -> None:
    """양팔 joint_states에서 전류/토크/pos를 읽어 버퍼에 저장."""
    now = time.monotonic()

    for side, bot in [("left", follower_bot_left), ("right", follower_bot_right)]:
        with bot.core.js_mutex:
            js = bot.core.joint_states
        if js is None:
            continue

        effort   = np.array(js.effort,   dtype=np.float32)
        position = np.array(js.position, dtype=np.float32)

        n = len(JOINT_NAMES)
        cur_mA  = np.zeros(n, dtype=np.float32)
        tor_Nm  = np.zeros(n, dtype=np.float32)
        pos_rad = np.zeros(n, dtype=np.float32)

        for i, joint in enumerate(JOINT_NAMES):
            if i >= len(effort):
                break
            kt          = MOTOR_KT[JOINT_MOTOR_TYPE[joint]]
            cur_mA[i]   = float(effort[i])
            tor_Nm[i]   = kt * cur_mA[i] / 1000.0
            pos_rad[i]  = float(position[i]) if i < len(position) else 0.0

        _log[side]["current"].append(cur_mA.copy())
        _log[side]["torque"].append(tor_Nm.copy())
        _log[side]["pos"].append(pos_rad.copy())
        _log[side]["time"].append(now)


def log_motor_state(
    follower_bot_left:  InterbotixManipulatorXS,
    follower_bot_right: InterbotixManipulatorXS,
) -> None:
    """양팔 전류/토크를 콘솔에 출력하고 버퍼에도 저장."""
    _read_and_append_log(follower_bot_left, follower_bot_right)

    print("\n" + "═" * 65)
    print(f"  [{time.strftime('%H:%M:%S')}] 팔로워 모터 상태")
    print("═" * 65)

    for side, bot in [("LEFT ", follower_bot_left), ("RIGHT", follower_bot_right)]:
        with bot.core.js_mutex:
            js = bot.core.joint_states
        if js is None:
            print(f"  [{side}] joint_states 없음")
            continue

        effort   = np.array(js.effort,   dtype=np.float32)
        position = np.array(js.position, dtype=np.float32)

        print(f"\n  [{side} follower]")
        print(f"  {'관절':<14} {'모터':<14} {'전류(mA)':>10} {'전류(A)':>8} "
              f"{'토크(N·m)':>10} {'pos(rad)':>10}")
        print(f"  {'-'*70}")

        for i, joint in enumerate(JOINT_NAMES):
            if i >= len(effort):
                break
            kt     = MOTOR_KT[JOINT_MOTOR_TYPE[joint]]
            cur_mA = float(effort[i])
            cur_A  = cur_mA / 1000.0
            torque = kt * cur_A
            pos    = float(position[i]) if i < len(position) else 0.0
            print(
                f"  {joint:<14} {JOINT_MOTOR_TYPE[joint]:<14}"
                f" {cur_mA:>+10.1f}"
                f" {cur_A:>+8.4f}"
                f" {torque:>+10.4f}"
                f" {pos:>+10.4f}"
            )
    print()


def save_log() -> None:
    """버퍼에 저장된 데이터를 그래프 3종으로 저장."""
    os.makedirs(LOG_SAVE_DIR, exist_ok=True)

    for side, data in _log.items():
        if not data["time"]:
            print(f"[save_log] {side}: 데이터 없음, 스킵")
            continue

        t       = np.array(data["time"])
        t       = t - t[0]
        current = np.array(data["current"])  # (T, 7)
        torque  = np.array(data["torque"])   # (T, 7)
        pos     = np.array(data["pos"])      # (T, 7)
        n_joints = len(JOINT_NAMES)          # 7

        rows = (n_joints + 1) // 2  # 4행

        # ── 그래프 1: 전류 (mA) ───────────────────────────────────
        fig, axes = plt.subplots(rows, 2, figsize=(14, 3 * rows))
        fig.suptitle(f"{side} follower — 전류 (mA)", fontsize=13)
        for j, ax in enumerate(axes.flat):
            if j >= n_joints:
                ax.set_visible(False)
                continue
            ax.plot(t, current[:, j], linewidth=1.0, color="tab:blue")
            ax.axhline(0, color="gray", linewidth=0.5)
            ax.set_title(JOINT_NAMES[j])
            ax.set_xlabel("time (s)")
            ax.set_ylabel("current (mA)")
            ax.grid(True)
        plt.tight_layout()
        path = os.path.join(LOG_SAVE_DIR, f"{side}_1_current_mA.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"[save_log] 저장: {path}")

        # ── 그래프 2: 토크 (N·m) ─────────────────────────────────
        fig, axes = plt.subplots(rows, 2, figsize=(14, 3 * rows))
        fig.suptitle(f"{side} follower — 토크 추정 (N·m)", fontsize=13)
        for j, ax in enumerate(axes.flat):
            if j >= n_joints:
                ax.set_visible(False)
                continue
            ax.plot(t, torque[:, j], linewidth=1.0, color="tab:orange")
            ax.axhline(0, color="gray", linewidth=0.5)
            ax.set_title(JOINT_NAMES[j])
            ax.set_xlabel("time (s)")
            ax.set_ylabel("torque (N·m)")
            ax.grid(True)
        plt.tight_layout()
        path = os.path.join(LOG_SAVE_DIR, f"{side}_2_torque_Nm.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"[save_log] 저장: {path}")

        # ── 그래프 3: 관절 위치 (rad) ────────────────────────────
        fig, axes = plt.subplots(rows, 2, figsize=(14, 3 * rows))
        fig.suptitle(f"{side} follower — 관절 위치 (rad)", fontsize=13)
        for j, ax in enumerate(axes.flat):
            if j >= n_joints:
                ax.set_visible(False)
                continue
            ax.plot(t, pos[:, j], linewidth=1.0, color="tab:purple")
            ax.axhline(0, color="gray", linewidth=0.5)
            ax.set_title(JOINT_NAMES[j])
            ax.set_xlabel("time (s)")
            ax.set_ylabel("position (rad)")
            ax.grid(True)
        plt.tight_layout()
        path = os.path.join(LOG_SAVE_DIR, f"{side}_3_pos_rad.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"[save_log] 저장: {path}")


def opening_ceremony(
    leader_bot_left: InterbotixManipulatorXS,
    leader_bot_right: InterbotixManipulatorXS,
    follower_bot_left: InterbotixManipulatorXS,
    follower_bot_right: InterbotixManipulatorXS,
) -> None:
    """Move all 4 robots to a pose where it is easy to start demonstration."""
    follower_bot_left.core.robot_reboot_motors('single', 'gripper', True)
    follower_bot_left.core.robot_set_operating_modes('group', 'arm', 'position')
    # follower_bot_left.core.robot_set_operating_modes('group', 'arm', 'current_based_position')
    follower_bot_left.core.robot_set_operating_modes('single', 'gripper', 'current_based_position')

    leader_bot_left.core.robot_set_operating_modes('group', 'arm', 'position')
    leader_bot_left.core.robot_set_operating_modes('single', 'gripper', 'position')
    follower_bot_left.core.robot_set_motor_registers('single', 'gripper', 'current_limit', 300)
    follower_bot_left.core.robot_set_motor_registers('group', 'arm', 'current_limit', 2300)

    follower_bot_right.core.robot_reboot_motors('single', 'gripper', True)
    # follower_bot_right.core.robot_set_operating_modes('group', 'arm', 'position')
    follower_bot_right.core.robot_set_operating_modes('group', 'arm', 'current_based_position')
    follower_bot_right.core.robot_set_operating_modes(
        'single', 'gripper', 'current_based_position'
    )
    leader_bot_right.core.robot_set_operating_modes('group', 'arm', 'position')
    leader_bot_right.core.robot_set_operating_modes('single', 'gripper', 'position')
    follower_bot_right.core.robot_set_motor_registers('single', 'gripper', 'current_limit', 300)
    follower_bot_right.core.robot_set_motor_registers('group', 'arm', 'current_limit',2300)

    torque_on(follower_bot_left)
    torque_on(leader_bot_left)
    torque_on(follower_bot_right)
    torque_on(leader_bot_right)

    start_arm_qpos = START_ARM_POSE[:6]
    move_arms(
        [leader_bot_left, follower_bot_left, leader_bot_right, follower_bot_right],
        [start_arm_qpos] * 4,
        moving_time=4.0,
    )
    move_grippers(
        [leader_bot_left, follower_bot_left, leader_bot_right, follower_bot_right],
        [LEADER_GRIPPER_JOINT_MID, FOLLOWER_GRIPPER_JOINT_CLOSE] * 2,
        moving_time=0.5
    )


def press_to_start(
    leader_bot_left: InterbotixManipulatorXS,
    leader_bot_right: InterbotixManipulatorXS,
    gravity_compensation: bool,
) -> None:
    leader_bot_left.core.robot_torque_enable('single', 'gripper', False)
    leader_bot_right.core.robot_torque_enable('single', 'gripper', False)
    print('Close the grippers to start')
    pressed = False
    while rclpy.ok() and not pressed:
        pressed = (
            (get_arm_gripper_positions(leader_bot_left) < LEADER_GRIPPER_CLOSE_THRESH) and
            (get_arm_gripper_positions(leader_bot_right) < LEADER_GRIPPER_CLOSE_THRESH)
        )
        get_interbotix_global_node().get_clock().sleep_for(DT_DURATION)
    if gravity_compensation:
        enable_gravity_compensation(leader_bot_left)
        enable_gravity_compensation(leader_bot_right)
    else:
        torque_off(leader_bot_left)
        torque_off(leader_bot_right)
    print('Started!')


def signal_handler(sig, frame, leader_bot_left, leader_bot_right,
                   follower_bot_left, follower_bot_right):
    print('\n[종료] Ctrl+C 감지 → 로그 저장 중...')
    disable_gravity_compensation(leader_bot_left)
    disable_gravity_compensation(leader_bot_right)
    save_log()
    exit(0)


def main(args: dict) -> None:
    gravity_compensation = args.get('gravity_compensation', False)
    last_log_time = 0.0

    node = create_interbotix_global_node('aloha')
    follower_bot_left = InterbotixManipulatorXS(
        robot_model='vx300s',
        robot_name='follower_left',
        node=node,
        iterative_update_fk=False,
    )
    follower_bot_right = InterbotixManipulatorXS(
        robot_model='vx300s',
        robot_name='follower_right',
        node=node,
        iterative_update_fk=False,
    )
    leader_bot_left = InterbotixManipulatorXS(
        robot_model='wx250s',
        robot_name='leader_left',
        node=node,
        iterative_update_fk=False,
    )
    leader_bot_right = InterbotixManipulatorXS(
        robot_model='wx250s',
        robot_name='leader_right',
        node=node,
        iterative_update_fk=False,
    )

    signal.signal(
        signal.SIGINT,
        partial(
            signal_handler,
            leader_bot_left=leader_bot_left,
            leader_bot_right=leader_bot_right,
            follower_bot_left=follower_bot_left,
            follower_bot_right=follower_bot_right,
        )
    )

    robot_startup(node)

    disable_gravity_compensation(leader_bot_left)
    disable_gravity_compensation(leader_bot_right)

    opening_ceremony(
        leader_bot_left,
        leader_bot_right,
        follower_bot_left,
        follower_bot_right,
    )

    press_to_start(leader_bot_left, leader_bot_right, gravity_compensation)

    # ── 텔레오퍼레이션 루프 ──────────────────────────────────────
    gripper_left_command  = JointSingleCommand(name='gripper')
    gripper_right_command = JointSingleCommand(name='gripper')

    while rclpy.ok():
        # 관절 위치 동기화
        leader_left_state_joints  = leader_bot_left.core.joint_states.position[:6]
        leader_right_state_joints = leader_bot_right.core.joint_states.position[:6]
        follower_bot_left.arm.set_joint_positions(leader_left_state_joints,  blocking=False)
        follower_bot_right.arm.set_joint_positions(leader_right_state_joints, blocking=False)

        # 그리퍼 위치 동기화
        gripper_left_command.cmd = LEADER2FOLLOWER_JOINT_FN(
            leader_bot_left.core.joint_states.position[6]
        )
        gripper_right_command.cmd = LEADER2FOLLOWER_JOINT_FN(
            leader_bot_right.core.joint_states.position[6]
        )
        follower_bot_left.gripper.core.pub_single.publish(gripper_left_command)
        follower_bot_right.gripper.core.pub_single.publish(gripper_right_command)

        # ── 매 루프: 버퍼에 저장 (고밀도) ──────────────────────
        _read_and_append_log(follower_bot_left, follower_bot_right)

        # ── 1초마다: 콘솔 출력 ──────────────────────────────────
        now = time.monotonic()
        if now - last_log_time >= LOG_INTERVAL:
            log_motor_state(follower_bot_left, follower_bot_right)
            last_log_time = now

        get_interbotix_global_node().get_clock().sleep_for(DT_DURATION)

    # 정상 종료 시 저장
    save_log()
    robot_shutdown(node)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-g', '--gravity_compensation',
        action='store_true',
        help='If set, gravity compensation will be enabled for the leader robots when teleop starts.',
    )
    main(vars(parser.parse_args()))
