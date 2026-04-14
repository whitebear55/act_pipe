"""ROS2 기반 ALOHA 백엔드.

Interbotix SDK(ROS2)를 이용하여 모터 상태를 읽고 명령을 전송.
MCC의 BaseSim 인터페이스를 구현하여 run_policy.py와 호환.

핵심 흐름:
  ROS2 /follower_*/joint_states → Obs 생성 → CompliancePolicy → set_joint_positions
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, Optional

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from sim.base_sim import BaseSim, Obs


# ── 모터 상수 (default.yml 기반) ──────────────────────────
# kt: effective torque constant = kt_winding × gear_ratio × η
# effort 단위: Interbotix current_based_position 모드 → mA
MOTOR_KT: dict[str, float] = {
    "XM540-W270": 2.4091,   # N·m / A (stall_torque / stall_current)
    "XM430-W350": 1.793,    # N·m / A
}

JOINT_ORDER = [
    "waist", "shoulder", "elbow",
    "forearm_roll", "wrist_angle", "wrist_rotate",
]
JOINT_ORDER_WITH_GRIPPER = JOINT_ORDER + ["left_finger"]
# gain_backdrive: 백드라이브 시 토크 보정 계수 (toddlerbot default 기준)
GAIN_BACKDRIVE = 1.0


@dataclass

class RealWorldAlohaROS2(BaseSim):
    """Interbotix SDK(ROS2)를 이용한 ALOHA 실제 환경 백엔드.

    BaseSim 인터페이스를 구현하여 compliance policy와 연결.
    """

    def __init__(
        self,
        node: Any,                         # InterbotixRobotNode
        follower_bot_left: Any,            # InterbotixManipulatorXS
        follower_bot_right: Any,           # InterbotixManipulatorXS
        xml_path: str,
        merged_config: dict[str, Any],
        control_dt: float = 0.02,
    ) -> None:
        self.name = "real_world"
        self.control_dt = float(control_dt)
        self.xml_path = str(xml_path)
        self._node = node
        self._bot_left  = follower_bot_left
        self._bot_right = follower_bot_right

        # ── 설정에서 모터 정보 파싱 ──────────────────────────
        motors_cfg = merged_config.get("motors", {})
        print(f"모터 정보 : {motors_cfg}")
        actuators_cfg = merged_config.get("actuators", {})
        print(f"actuators 정보 : {actuators_cfg}")

        # XML에서 joint 순서 결정
        xml_root = ET.parse(xml_path).getroot()
        print(f"xml_root : {xml_root}")
        


        self.motor_ordering: list[str] = [
            name for name in motors_cfg
            if "zero_pos" in motors_cfg[name]   # motors.yml 항목만 (zero_pos 있는 것)
            and xml_root.find(
                f".//joint[@name='{name.replace(chr(95), chr(47), 1)}']"
            ) is not None
        ]
        print(f"self.motor_ordering : {self.motor_ordering}")

        if not self.motor_ordering:
            raise ValueError(f"XML에서 motor joint를 찾을 수 없음: {xml_path}")

        n = len(self.motor_ordering)
        # print(f"찾은 모터 이름 : {name}")
        self.motor_kt    = np.ones(n, dtype=np.float32)
        self.motor_groups = np.array(
            [motors_cfg[name].get("group", "arm") for name in self.motor_ordering],
            dtype=object,
        )
        self.drive_state = np.ones(n, dtype=np.float32)

        for i, name in enumerate(self.motor_ordering):
            # motors.yml에는 motor 키 없음 → robot.yml에서 같은 joint 이름으로 찾기
            slash_name = name.replace("_", "/", 1)   # left_waist → left/waist
            motor_type = motors_cfg.get(slash_name, {}).get("motor", "")
            kt = MOTOR_KT.get(motor_type, 1.0)       # effective kt 사용
            self.motor_kt[i] = kt
            print(f"motor의 Torque Constant : {self.motor_kt[i]}")

        # ── MuJoCo qpos 매핑 (RealWorldDynamixel과 동일 로직) ──
        # admittance control을 위한 arm 부분 -> 12개의 모터
        # Mujoco에서 계산하기 위해서 aloha_mobile.xml에는 16개의 상태가 필요 
        # 이 둘을 연결하는 함수
        self._build_qpos_map(xml_root, xml_path)

        # 첫 메시지 수신 대기
        print("joint_states 수신 대기 중...")
        timeout = 10.0
        t0 = time.monotonic()
        import rclpy
        while (self._bot_left.core.joint_states is None or self._bot_right.core.joint_states is None):
            rclpy.spin_once(node, timeout_sec=0.05)
            if time.monotonic() - t0 > timeout:
                raise RuntimeError("joint_states 수신 타임아웃")
        print("joint_states 수신 시작.")

        # ── [추가] 전류 로그 버퍼 ────────────────────────────────
        # motor_ordering 순서대로 14개 모터의 전류(mA)와 토크(N·m) 기록
        import os, matplotlib
        matplotlib.use("Agg")
        self._current_log: list[np.ndarray] = []   # (T, n) — mA
        self._torque_log:  list[np.ndarray] = []   # (T, n) — N·m
        self._time_log:    list[float]       = []   # (T,)
        self._log_save_dir = "logs"
        os.makedirs(self._log_save_dir, exist_ok=True)


    # ── qpos 매핑 ────────────────────────────────────────────
    # ROS2(실제 모터) 와 Mujoco사이의 모터 사이의 매핑
    def _build_qpos_map(self, xml_root: ET.Element, xml_path: str) -> None:
        model = mujoco.MjModel.from_xml_path(xml_path)
        self._qpos_dim = int(model.nq)
        self._qvel_dim = int(model.nv)
        self._qpos_base = np.zeros(self._qpos_dim, dtype=np.float32)

        if int(model.nkey) > 0:
            self._qpos_base = np.array(model.key_qpos[0], dtype=np.float32)

        motor_index = {name: i for i, name in enumerate(self.motor_ordering)}
        n = len(self.motor_ordering)

        motor_to_dof = np.zeros((self._qpos_dim, n), dtype=np.float32)

        # 매핑된 qpos 인덱스를 명시적으로 추적
        self._mapped_qpos_indices: list[int] = []

        for name in self.motor_ordering:
            xml_joint_name = name.replace("_", "/", 1)
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, xml_joint_name)
            if jid < 0:
                continue
            qpos_adr = int(model.jnt_qposadr[jid])
            motor_to_dof[qpos_adr, motor_index[name]] = 1.0
            self._mapped_qpos_indices.append(qpos_adr)   # ← 추가

        self._motor_to_dof = motor_to_dof
        # TODO 여기까지 함 밑에는 왜 없어진거지?
        self._mapped_qpos_set = set(self._mapped_qpos_indices)  # ← 추가


    def get_qpos(self, motor_pos: np.ndarray) -> np.ndarray:
        qpos = self._qpos_base.copy()
        mapped = self._motor_to_dof @ motor_pos

        # v != 0 조건 대신 명시적 인덱스 사용
        for i in self._mapped_qpos_indices:
            qpos[i] = mapped[i]   # 0이든 아니든 항상 덮어씀

        return qpos


    def get_qvel(self, motor_vel: np.ndarray) -> np.ndarray:
        qvel = np.zeros(self._qvel_dim, dtype=np.float32)
        mapped = self._motor_to_dof @ motor_vel

        for i in self._mapped_qpos_indices:
            qvel[i] = mapped[i]   # qvel은 qpos_adr와 동일 인덱스 사용

        return qvel

    # ── 토크 추정 ─────────────────────────────────────────────

    def _effort_to_motor_tor(
        self,
        effort: np.ndarray,
        velocity: np.ndarray,
        eps_vel: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """effort(mA) → motor_tor(N·m) 변환.

        current_based_position 모드: effort = Present_Current [mA]
        """

        # TODO 토크 추정
        # mA → A
        i_est = effort / 1000.0

        tau_est = self.motor_kt * i_est

        # drive_state: forward(+1) / backward(-1)
        prev = self.drive_state.copy()
        drive_state = np.where(
            np.abs(velocity) > eps_vel,
            np.sign(tau_est * velocity),
            prev,
        ).astype(np.float32)
        self.drive_state = drive_state

        gain_back = np.where(
            self.motor_groups == "arm",
            GAIN_BACKDRIVE,
            1.0,
        ).astype(np.float32)
        
        gain = np.where(drive_state > 0, 1.0, gain_back).astype(np.float32)
        motor_tor = gain * tau_est

        return i_est, drive_state, motor_tor

    def _read_joint_states(self) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                          np.ndarray, np.ndarray, np.ndarray]:
        """js_mutex로 thread-safe하게 양팔 joint_states 읽기.

        Returns:
            left_pos, left_vel, left_eff,
            right_pos, right_vel, right_eff  (각각 numpy array)
        """
        with self._bot_left.core.js_mutex:
            ljs = self._bot_left.core.joint_states

        with self._bot_right.core.js_mutex:
            rjs = self._bot_right.core.joint_states

        if ljs is None or rjs is None:
            raise RuntimeError("joint_states 미수신")

        return (
            np.array(ljs.position, dtype=np.float32),
            np.array(ljs.velocity, dtype=np.float32),
            np.array(ljs.effort,   dtype=np.float32),
            np.array(rjs.position, dtype=np.float32),
            np.array(rjs.velocity, dtype=np.float32),
            np.array(rjs.effort,   dtype=np.float32),
        )
    
    # ── BaseSim 인터페이스 ────────────────────────────────────


    def get_observation(self, retries: int = 0) -> Obs:
        """현재 팔로워 팔 상태를 Obs로 변환.

        구조:
          1. Interbotix 내부 joint_states를 js_mutex로 안전하게 읽기
          2. motor_ordering 순서에 맞게 재배열
          3. 토크 추정 → Obs 생성
        """
        left_pos, left_vel, left_eff, \
        right_pos, right_vel, right_eff = self._read_joint_states()

        # ── motor_ordering 순서로 재배열 ─────────────────────
        # vx300s joint_states 순서:
        #   position[0..5] = JOINT_ORDER (waist~wrist_rotate)
        #   position[6]    = gripper (left_finger)
        n = len(self.motor_ordering)
        motor_pos = np.zeros(n, dtype=np.float32)
        motor_vel = np.zeros(n, dtype=np.float32)
        motor_eff = np.zeros(n, dtype=np.float32)

        for i, name in enumerate(self.motor_ordering):
            # motors.yml: "right_waist", "left_waist" 형식
            if name.startswith("right_"):
                joint     = name[len("right_"):]    # "right_waist" → "waist"
                side_pos  = right_pos
                side_vel  = right_vel
                side_eff  = right_eff
            else:
                joint     = name[len("left_"):]     # "left_waist" → "waist"
                side_pos  = left_pos
                side_vel  = left_vel
                side_eff  = left_eff

            if joint in JOINT_ORDER_WITH_GRIPPER:
                idx = JOINT_ORDER_WITH_GRIPPER.index(joint)
                if idx < len(side_pos):
                    motor_pos[i] = side_pos[idx]
                    motor_vel[i] = side_vel[idx] if idx < len(side_vel) else 0.0
                    motor_eff[i] = side_eff[idx] if idx < len(side_eff) else 0.0

        # ── 토크 추정 ─────────────────────────────────────────
        i_est, drive_state, motor_tor = self._effort_to_motor_tor(
            motor_eff, motor_vel
        )

        # ── [추가] 전류/토크 로그 저장 ───────────────────────────
        self._current_log.append(motor_eff.copy())          # mA
        self._torque_log.append(motor_tor.copy())           # N·m
        self._time_log.append(float(time.monotonic()))

        return Obs(
            ang_vel    = None,
            time       = float(time.monotonic()),
            motor_pos  = motor_pos,
            motor_vel  = motor_vel,
            motor_tor  = motor_tor,
            qpos       = self.get_qpos(motor_pos),
            qvel       = self.get_qvel(motor_vel),
            rot        = None,
            motor_cur  = i_est,
            motor_drive= drive_state,
            motor_pwm  = np.zeros_like(motor_pos),
            motor_vin  = np.full_like(motor_pos, 12.0),
            motor_temp = np.zeros_like(motor_pos),
        )

    def set_motor_target(self, motor_angles: np.ndarray) -> None:
        """어드미턴스 제어 출력을 팔로워 팔에 전송."""
        motor_angles = np.asarray(motor_angles, dtype=np.float32).reshape(-1)

        # motor_ordering 순서에서 좌/우팔 분리
        left_pos  = np.zeros(6, dtype=np.float32)
        right_pos = np.zeros(6, dtype=np.float32)
        left_grip  = None
        right_grip = None

        for i, name in enumerate(self.motor_ordering):
            if i >= len(motor_angles):
                break
            if name.startswith("right_"):
                joint = name.replace("right_", "")
                if joint in JOINT_ORDER:
                    right_pos[JOINT_ORDER.index(joint)] = motor_angles[i]
                elif joint == "left_finger":
                    right_grip = motor_angles[i]

            elif name.startswith("left_"):
                joint = name.replace("left_", "")
                if joint in JOINT_ORDER:
                    left_pos[JOINT_ORDER.index(joint)] = motor_angles[i]
                elif joint == "left_finger":
                    left_grip = motor_angles[i]

        # 팔 위치 명령
        self._bot_left.arm.set_joint_positions(left_pos.tolist(),  blocking=False)
        self._bot_right.arm.set_joint_positions(right_pos.tolist(), blocking=False)

        # 그리퍼 명령 (있을 경우)
        if left_grip is not None:
            from interbotix_xs_msgs.msg import JointSingleCommand
            cmd = JointSingleCommand(name="gripper")
            cmd.cmd = float(left_grip)
            self._bot_left.gripper.core.pub_single.publish(cmd)
        if right_grip is not None:
            from interbotix_xs_msgs.msg import JointSingleCommand
            cmd = JointSingleCommand(name="gripper")
            cmd.cmd = float(right_grip)
            self._bot_right.gripper.core.pub_single.publish(cmd)

    def step(self) -> None:
        pass

    def sync(self) -> bool:
        return True
    
    def save_log(self) -> None:
        """전류(mA)와 토크(N·m) 그래프를 저장."""
        import os
        import matplotlib.pyplot as plt

        if not self._time_log:
            print("[save_log] 데이터 없음, 스킵")
            return

        t        = np.array(self._time_log)
        t        = t - t[0]
        current  = np.array(self._current_log)  # (T, n)
        torque   = np.array(self._torque_log)   # (T, n)

        # motor_ordering에서 좌/우 분리
        left_idx  = [i for i, n in enumerate(self.motor_ordering)
                     if n.startswith("left_")]
        right_idx = [i for i, n in enumerate(self.motor_ordering)
                     if n.startswith("right_")]

        joint_labels = ["waist", "shoulder", "elbow",
                        "forearm_roll", "wrist_angle", "wrist_rotate", "gripper"]

        for side, indices in [("left", left_idx), ("right", right_idx)]:
            if not indices:
                continue

            cur = current[:, indices]   # (T, 7)
            tor = torque[:, indices]    # (T, 7)
            n_joints = len(indices)
            labels = joint_labels[:n_joints]

            # ── 그래프 1: 전류 (mA) ───────────────────────────────
            fig, axes = plt.subplots(
                (n_joints + 1) // 2, 2,
                figsize=(14, 3 * ((n_joints + 1) // 2))
            )
            fig.suptitle(f"{side} — 전류 (mA)", fontsize=13)
            for j, ax in enumerate(axes.flat):
                if j >= n_joints:
                    ax.set_visible(False)
                    continue
                ax.plot(t, cur[:, j], linewidth=1.2, color="tab:blue")
                ax.axhline(0, color="gray", linewidth=0.5)
                ax.set_title(labels[j])
                ax.set_xlabel("time (s)")
                ax.set_ylabel("current (mA)")
                ax.grid(True)
            plt.tight_layout()
            path = os.path.join(self._log_save_dir, f"{side}_current_mA.png")
            fig.savefig(path, dpi=120)
            plt.close(fig)
            print(f"[save_log] 저장: {path}")

            # ── 그래프 2: 토크 (N·m) ─────────────────────────────
            fig, axes = plt.subplots(
                (n_joints + 1) // 2, 2,
                figsize=(14, 3 * ((n_joints + 1) // 2))
            )
            fig.suptitle(f"{side} — 토크 추정 (N·m)", fontsize=13)
            for j, ax in enumerate(axes.flat):
                if j >= n_joints:
                    ax.set_visible(False)
                    continue
                ax.plot(t, tor[:, j], linewidth=1.2, color="tab:orange")
                ax.axhline(0, color="gray", linewidth=0.5)
                ax.set_title(labels[j])
                ax.set_xlabel("time (s)")
                ax.set_ylabel("torque (N·m)")
                ax.grid(True)
            plt.tight_layout()
            path = os.path.join(self._log_save_dir, f"{side}_torque_Nm.png")
            fig.savefig(path, dpi=120)
            plt.close(fig)
            print(f"[save_log] 저장: {path}")


    def close(self) -> None:
        self.save_log()   # ← 추가
        pass
