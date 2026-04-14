"""Compliance affordance policy built on top of the base compliance controller.

This keeps old policy orchestration style while using local VLM + minimalist
compliance modules.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import joblib
import mujoco
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R

from policy.compliance import CompliancePolicy
from real_world.camera import Camera
from vlm.affordance.affordance_predictor import AffordancePredictor
from vlm.affordance.compliance_predictor import CompliancePredictor
from vlm.affordance.plan_ee_pose import plan_end_effector_poses
from vlm.depth.depth_utils import depth_to_xyzmap

# ── Task 테이블 정의 ───────────────────────────────────────────────────────────
# 새 task 추가 시 아래 테이블만 수정하면 됩니다.

TASK_OBJECT_MAP: Dict[str, str] = {
    "wiping":  "black ink",
    "drawing": "star",
    "picking": "red box",
}

TASK_TOOL_MAP: Dict[str, str] = {
    "wiping":  "eraser",
    "drawing": "pen",
    "picking": "gripper",
}

# status("wiping") → plan task명("wipe") 변환용
TASK_PLAN_MAP: Dict[str, str] = {
    "wiping":  "wipe",
    "drawing": "draw",
    "picking": "pick",
}

# robot별 task별 기본 site 이름
TASK_DEFAULT_SITE_MAP: Dict[str, Dict[str, List[str]]] = {
    "leap": {
        "wiping":  ["mf_tip"],
        "drawing": ["rf_tip", "if_tip"],
        "picking": ["mf_tip"],
    },
    "toddlerbot": {
        "wiping":  ["left_hand_center"],
        "drawing": ["right_hand_center"],
        "picking": ["left_hand_center"],
    },
    "fr": {
        "wiping":  ["attachment_site"],
        "drawing": ["attachment_site"],
        "picking": ["attachment_site"],
    },
}
TASK_DEFAULT_DESCRIPTION = {
    "wiping":  lambda obj: f"wipe up the {obj} on the whiteboard with an eraser.",
    "drawing": lambda obj: f"draw the {obj} on the whiteboard using the pen.",
    "picking": lambda obj: f"pick up the {obj}.",
}

# replay를 지원하는 task 집합
# picking처럼 매번 위치가 달라지는 task는 여기에 추가하지 않습니다.
REPLAY_SUPPORTED_TASKS: frozenset = frozenset({"wiping", "drawing"})

# 모든 유효 task 집합 (TASK_OBJECT_MAP 기준으로 자동 생성)
ACTIVE_TASKS: frozenset = frozenset(TASK_OBJECT_MAP.keys())

# task 완료 시 동작 설정
# loop=True: 완료 후 재예측 요청 (wiping의 close-loop)
# on_complete_action: step()의 action에 적용할 값 (None이면 없음)
TASK_COMPLETION_CONFIG: Dict[str, Dict[str, Any]] = {
    "wiping": {
        "loop":             True,
        "on_complete_action": None,
    },
    "drawing": {
        "loop":             False,
        "on_complete_action": None,
    },
    "picking": {
        "loop":             False,
        "on_complete_action": {"index": -1, "value": 0.0},  # 그리퍼 닫기
    },
}

# task 진입 시 초기화할 task 전용 상태변수
TASK_RESET_STATE: Dict[str, Dict[str, Any]] = {
    # picking 전용 상태 없음 (pick_arrived, pick_target_3d 제거됨)
}

# loop task에서 재예측 전 대기 시간(초)
WIPE_PAUSE_DURATION = 2.0

class ComplianceVLMPolicy(CompliancePolicy):
    """Guides compliance references via affordance-predicted trajectories."""

    def __init__(
        self,
        name: str,
        robot: str,
        init_motor_pos: npt.ArrayLike,
        replay: str = "",
        site_names: str = "",
        object: str = "black ink. whiteboard. vase",
        record_video: bool = False,
        image_height: int = 480,
        image_width: int = 640,
        predictor_model: str = "gemini-2.5-flash",
        predictor_provider: str = "gemini",
    ) -> None:
        if robot == "leap":
            gin_config_name = "leap_vlm.gin"
        elif robot == "toddlerbot":
            gin_config_name = "toddlerbot_vlm.gin"
        elif robot == "fr":
            gin_config_name = "fr_vlm.gin"
        else:
            raise ValueError(f"Unsupported robot: {robot}")

        super().__init__(
            name=name,
            robot=robot,
            init_motor_pos=init_motor_pos,
            config_name=gin_config_name,
            show_help=False,
        )

        model = self.controller.wrench_sim.model 
        self.head_name = str(self.compliance_cfg.head_name).strip() # camera 위치
        self.head_site_id = -1
        self.head_body_id = -1
        if self.head_name:
            self.head_site_id = int(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.head_name)
            )
        if self.head_name:
            self.head_body_id = int(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.head_name)
            )

        self.image_height = int(image_height)
        self.image_width = int(image_width)

        self.target_site_names: List[str] = []
        site_names_str = str(site_names).strip()
        self.site_names_fixed = len(site_names_str) > 0
        if self.site_names_fixed:
            self.target_site_names = [
                s.strip() for s in site_names_str.split(",") if s.strip()
            ]
        if not self.target_site_names:
            # 기본 task="wiping"으로 초기 site 결정
            self.target_site_names = self._default_site_names_for_mode("wiping")

        # ref_motor_pos dimension check
        cfg_ref_motor_pos = np.asarray(
            self.compliance_cfg.ref_motor_pos, dtype=np.float32
        ).reshape(-1)
        if cfg_ref_motor_pos.size > 0:
            if cfg_ref_motor_pos.shape[0] != self.default_motor_pos.shape[0]:
                raise ValueError(
                    "ComplianceConfig.ref_motor_pos has wrong size: "
                    f"expected {self.default_motor_pos.shape[0]}, "
                    f"got {cfg_ref_motor_pos.shape[0]}."
                )
            self.ref_motor_pos = cfg_ref_motor_pos.copy()
        
        self.neck_pitch_idx: Optional[int] = None
        if self.robot == "toddlerbot":
            motor_ordering: List[str] = [
                str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i))
                for i in range(int(model.nu))
            ]
            self.neck_pitch_idx = (
                motor_ordering.index("neck_pitch_act")
                if "neck_pitch_act" in motor_ordering
                else None
            )

        # task direction -> 누르는 방향 : Normal
        # 표면을 따라 움직이는 방향 : Tangent
        self.kp_pos_normal = float(self.compliance_cfg.kp_pos_normal)
        self.kp_pos_tangent = float(self.compliance_cfg.kp_pos_tangent)
        self.kp_rot_normal = float(self.compliance_cfg.kp_rot_normal) # 손목을 비틀지 마라
        self.kp_rot_tangent = float(self.compliance_cfg.kp_rot_tangent) # 표면에 수평하게 손목을 유지 
        self.fixed_contact_force = float(self.compliance_cfg.fixed_contact_force) # 물체에 접촉한 후 유지해야할 일정한 힘의 크기
        
        self.rest_pose_command = np.asarray(
            self.base_pose_command, dtype=np.float32
        ).copy()
        if not self._has_initial_pose_override:
            self._set_rest_pose_from_ref_motor_pos()
        else:
            # initial_pose가 있으면 그대로 rest_pose로 사용
            self.rest_pose_command = np.asarray(self.base_pose_command, dtype=np.float32).copy()
            print(f"DEBUG: Computed Rest Pose = {self.rest_pose_command}")
        # self._set_rest_pose_from_ref_motor_pos()

        print(f"DEBUG: Computed Rest Pose = {self.rest_pose_command}") # ref_motor_pos로 계산된 값

        # Task 상태 initialization
        self.status = "waiting"
        self.target_object_label = str(object)

        # task별 object label (CLI --object 인자로 wiping 기본값 설정)
        self.task_object_map: Dict[str, str] = {
            task: TASK_OBJECT_MAP[task] for task in TASK_OBJECT_MAP
        }
        self.task_object_map["wiping"] = str(object)  # CLI 인자 반영
        self.tool = TASK_TOOL_MAP.get("wiping", "eraser")

        self.trajectory_plans: Dict[str, Tuple[np.ndarray, ...]] = {}
        self.traj_start_time: Optional[float] = None

        self.wipe_pause_duration  = WIPE_PAUSE_DURATION
        self.wipe_pause_end_time: Optional[float] = None
        self.prediction_requested  = False
        self.prediction_counter    = 0
        self.replay                = str(replay).strip()
        self.fixed_trajectory_active = False
        self.replay_task: Optional[str] = None
        self.replay_contact_points_camera:  Dict[str, np.ndarray] = {}
        self.replay_contact_normals_camera: Dict[str, np.ndarray] = {}
        self.replay_unavailable_reported = False

        # 작업 완료 Flag
        self._pending_complete_action: Optional[Dict[str, Any]] = None

        # Replay 시스템
        self.predictor: Optional[AffordancePredictor] = None
        self._load_replay_trajectory()
        self._activate_replay_task_if_available()

        # VLM 엔진 및 카메라 초기화 
        self.predictor = AffordancePredictor(
            model=str(predictor_model),
            provider=str(predictor_provider),
            server_ip="127.0.0.1"
        )


        self.pick_cp = CompliancePredictor(
            provider=str(predictor_provider),
            model=str(predictor_model),
        )

        self.left_camera: Optional[Camera] = None
        self.right_camera: Optional[Camera] = None
        try:
            self.left_camera = Camera("left")
            self.right_camera = Camera("right")
        except Exception as exc:
            self.left_camera = None
            self.right_camera = None
            print(f"[ComplianceVLM] camera stream disabled: {exc}")

        # 키보드를 이용하여 VLM에게 명령을 지시
        self.teleop.set_command_bindings(
            {"w": "wiping", "d": "drawing", "p": "picking"},
            help_labels={"w": "wipe", "d": "draw", "p" : "pick"},
            enable_default_controls=False,
        )
        self.teleop.print_help(prefix="[ComplianceVLM]")

        self.debug_output_dir = tempfile.TemporaryDirectory(prefix="compliance_vlm_")
        self.prediction_executor = ThreadPoolExecutor(max_workers=1) # VLM을 별도의 Thread에서 구동
        self.prediction_future: Optional[Future] = None # VLM의 비동기 처리를 위한 자료향

        self.record_video = bool(record_video)
        self.video_logging_active = False
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.video_temp_dir: Optional[tempfile.TemporaryDirectory] = None
        self.video_path: Optional[Path] = None
        self.video_fps: Optional[float] = None
        self.last_left_frame: Optional[np.ndarray] = None
        self.last_right_frame: Optional[np.ndarray] = None
        self.video_frame_timestamps: List[float] = []

        self.set_stiffness(
            pos_stiffness=[400.0, 400.0, 400.0], rot_stiffness=[40.0, 40.0, 40.0]
        )

        
    
    def reset(self) -> None:
        self.traj_start_time = None
        self.trajectory_plans = {}
        self.wipe_pause_end_time = None
        self.prediction_requested = False
        if self.prediction_future is not None:
            self.prediction_future.cancel()
            self.prediction_future = None
        self.status = "waiting"
        self.fixed_trajectory_active = False
        self._activate_replay_task_if_available() # 이전에 저장된 성공 궤적이 있다면 다시 로드할 준비

    # 명령어 표준화(-> 작업에 mapping)
    def _normalize_task_label(self, value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return None
        normalized = value.strip().lower()
        if normalized in ("wipe", "wiping"):
            return "wipe"
        if normalized in ("draw", "drawing"):
            return "draw"
        return None

    def _set_rest_pose_from_ref_motor_pos(self) -> None:
        """Initialize rest pose from the compliance reference mapping."""
        comp_ref = self.controller.compliance_ref
        if comp_ref is None:
            return

        rest_pose = np.asarray(
            comp_ref.get_x_ref_from_motor_pos(
                np.asarray(self.ref_motor_pos, dtype=np.float32)
            ),
            dtype=np.float32,
        )
        self.rest_pose_command = rest_pose # 아무일도 안할 때 돌아올 휴식 위치
        self.pose_command[:, :] = self.rest_pose_command # 실제 외력에 반응하면서 움직이는 실제 경로
        self.base_pose_command[:, :] = self.rest_pose_command # 이상적인 경로(외력이 없을때)

    def _activate_replay_task_if_available(self) -> None:
        if not self.replay_task:
            return
        if self.replay_task == "wipe":
            self.set_mode(True)
            print("[ComplianceVLM] replay task detected: wipe (auto-selected).")
        elif self.replay_task == "draw":
            self.set_mode(False)
            print("[ComplianceVLM] replay task detected: draw (auto-selected).")

    # 저장된 로봇 부위 확인 : 리플레이 파일(.lz4)안에 어떤 로봇 부위(site)의 움직임 정보가 들어가있는지 알 수 있는 함수
    def _replay_site_names(self) -> List[str]:
        return [str(x) for x in self.replay_contact_points_camera.keys()]

    def _default_site_names_for_mode(self, task: str) -> List[str]:
        """task 문자열 기반으로 기본 site names 반환."""
        robot_map = TASK_DEFAULT_SITE_MAP.get(self.robot, {})
        if robot_map:
            return robot_map.get(task, ["attachment_site"])
        # 정의되지 않은 robot은 첫 번째 wrench site 사용
        return [self.wrench_site_names[0]] if self.wrench_site_names else []

    # 과거의 성공 기억 불러오기
    def _load_replay_trajectory(self) -> None:
        self.replay_task = None
        self.replay_contact_points_camera = {}
        self.replay_contact_normals_camera = {}
        path = self.get_fixed_trajectory_path()
        if path is None:
            if self.replay:
                print(
                    f"[ComplianceVLM] replay trajectory not found under: {self.replay}"
                )
            return
        try:
            payload = joblib.load(path)
            if not isinstance(payload, dict):
                raise ValueError("replay payload must be a dict.")
            task = self._normalize_task_label(payload.get("task"))
            if task is None:
                raise ValueError("replay payload missing/invalid task.")
            contact_points = payload.get("contact_pos_camera") # 카메라 시점에서 물체와 접촉했던 3D 좌표
            contact_normals = payload.get("contact_normals_camera") # 접촉 시점에서 어느 방향으로 힘을 가했었는지에 대한 방향 데이터
            if isinstance(contact_points, dict) and isinstance(contact_normals, dict):
                self.replay_contact_points_camera = {
                    str(k): np.asarray(v, dtype=np.float32)
                    for k, v in contact_points.items()
                }
                self.replay_contact_normals_camera = {
                    str(k): np.asarray(v, dtype=np.float32)
                    for k, v in contact_normals.items()
                }
            if (
                not self.replay_contact_points_camera
                or not self.replay_contact_normals_camera
            ):
                raise ValueError(
                    "replay payload must contain contact_pos_camera and contact_normals_camera."
                )
            self.replay_task = task
            print(
                f"[ComplianceVLM] loaded replay trajectory for task '{task}' from {path}"
            )
        except Exception as exc:
            print(f"[ComplianceVLM] failed to load replay trajectory: {exc}")
            self.replay_task = None
            self.replay_contact_points_camera = {}
            self.replay_contact_normals_camera = {}

    # 리플레이 가능 여부 판별
    def can_use_replay_for_current_mode(self) -> bool:
        REPLAY_SUPPORTED_TASKS = frozenset({"wiping", "drawing"})
        if self.status not in REPLAY_SUPPORTED_TASKS:
            return False
        if self.replay_task is None:
            return False
        current_task = "wipe" if self.status == "wiping" else "draw"
        replay_sites = set(self._replay_site_names())
        if not replay_sites:
            return False
        if current_task != self.replay_task:
            return False
        return any(
            site_name in replay_sites and site_name in self.wrench_site_names
            for site_name in self.target_site_names
        )

    def set_mode(
        self,
        task :str,
        object_label: Optional[str] = None,
        site_names: Optional[List[str]] = None,
    ) -> None:
        
        if task not in ACTIVE_TASKS:
            print(f"[ComplianceVLM] 알 수 없는 task: {task}")
            return

        # 이미 같은 task 진행 중이면 무시
        if self.status == task and self.status != "waiting":
            return

        # ── status / tool 설정 ──────────────────────────────
        self.status = task
        self.tool   = TASK_TOOL_MAP.get(task, "gripper")

        # ── object label 설정 ───────────────────────────────
        resolved_label = (
            object_label
            if object_label is not None
            else self.task_object_map.get(task)
        )
        if resolved_label is not None:
            self.target_object_label = str(resolved_label)

        # ── site names 설정 ─────────────────────────────────
        if site_names is not None and len(site_names) > 0:
            self.target_site_names = [str(x) for x in site_names]
        elif not self.site_names_fixed:
            self.target_site_names = self._default_site_names_for_mode(task)

        # ── predictor default_task 설정 ─────────────────────
        if self.predictor is not None:
            desc_fn = TASK_DEFAULT_DESCRIPTION.get(task)
            if desc_fn is not None:
                self.predictor.default_task = desc_fn(self.target_object_label)

        # ── 궤적 초기화 ─────────────────────────────────────
        self.trajectory_plans        = {}
        self.traj_start_time         = None
        self.wrench_command[:, :]    = 0.0
        self.fixed_trajectory_active = False
        self.prediction_requested    = not self.can_use_replay_for_current_mode()

        # 저장된 파일 찾기
    def get_fixed_trajectory_path(self) -> Optional[Path]:
        if self.replay:
            replay_path = Path(self.replay).expanduser()
            if replay_path.suffix == ".lz4":
                return replay_path if replay_path.exists() else None
            path = replay_path / "trajectory.lz4"
            return path if path.exists() else None

        return None

    # 과거 데이터를 현재 좌표로 번역하기 (카메라 기준 좌표 -> 로봇 월드 좌표로 실시간 변환하여 궤적 생성)
    def prepare_fixed_plan(self) -> None:
        if not self.can_use_replay_for_current_mode():
            return
        try: # 유효한 site 필터링
            valid_sites = [
                site_name
                for site_name in self.target_site_names
                if site_name in self.wrench_site_names
                and site_name in self.replay_contact_points_camera
                and site_name in self.replay_contact_normals_camera
            ]
            if not valid_sites:
                self.fixed_trajectory_active = False
                return

            # Replan from replayed camera-space contacts so execution uses current
            # head pose and current compliance parameters.
            head_pos, head_quat = self.get_head_pose() #현재 로봇 머리의 정확한 위치와 각도를 가져옴(mjModel.forward()이용)
            
# TODO pose_cur, contact_points, contact_nnormals -> 이미지 속 힌트를 로봇의 실제 팔 움직임으로 바꾸기 위해 필요한것들
            pose_cur = { # 시작점
                site_name: np.asarray(
                    self.pose_command[self.wrench_site_names.index(site_name)],
                    dtype=np.float32,
                ) 
                for site_name in valid_sites
            }
            contact_points = { # 카메라(헤드) 기준 접촉 지점
                site_name: np.asarray(
                    self.replay_contact_points_camera[site_name], dtype=np.float32
                )
                for site_name in valid_sites
            }
            contact_normals = { # 카메라(헤드) 기준 누르는 방향
                site_name: np.asarray(
                    self.replay_contact_normals_camera[site_name], dtype=np.float32
                )
                for site_name in valid_sites
            }
            # 현재의 카메라 위치 + 현재의 댐핑/강성 값 + 현재의 로봇 EE위치를 모두 조합하여 지금 상황에 맞는 이동궤적 생성
            self.trajectory_plans = plan_end_effector_poses(
                contact_points_camera=contact_points,
                contact_normals_camera=contact_normals,
                head_position_world=np.asarray(head_pos, dtype=np.float32),
                head_quaternion_world_wxyz=np.asarray(head_quat, dtype=np.float32),
                tangent_pos_stiffness=float(self.kp_pos_tangent),
                normal_pos_stiffness=float(self.kp_pos_normal),
                tangent_rot_stiffness=float(self.kp_rot_tangent),
                normal_rot_stiffness=float(self.kp_rot_normal),
                contact_force=float(self.fixed_contact_force),
                pose_cur=pose_cur,
                output_dir=None,
                traj_dt=float(self.control_dt),
                traj_v_max_contact=0.02,
                traj_v_max_free=0.1,
                tool=self.tool,
                robot_name=self.robot,
                task=self.replay_task,
                mass=float(self.mass),
                inertia_diag=np.asarray(self.inertia_diag, dtype=np.float32),
            )

            self.traj_start_time = None
            self.fixed_trajectory_active = bool(self.trajectory_plans)
            if self.fixed_trajectory_active:
                print(
                    "[ComplianceVLM] replay trajectory prepared for sites: "
                    f"{list(self.trajectory_plans.keys())}"
                )
        except Exception as exc:
            print(f"[ComplianceVLM] failed to load fixed trajectory: {exc}")
            self.fixed_trajectory_active = False

    # 예측 데이터 저장소 만들기(임시 디렉토리 안에 VLM이 예측한 결과물을 저장할 폴더를 생성)
    def get_prediction_output_dir(self, prediction_idx: int) -> Optional[str]:
        if self.debug_output_dir is None:
            return None
        base = Path(self.debug_output_dir.name)
        out_dir = base / f"prediction_{prediction_idx}"
        out_dir.mkdir(parents=True, exist_ok=True)
        return str(out_dir)

    def check_mode_command(self) -> None:
        cmd = self.teleop.poll_command()
        if cmd is None:
            return

        if cmd in ACTIVE_TASKS:
            if self.status != cmd:
                self.set_mode(cmd)
                print(f"[ComplianceVLM] {cmd} 모드 진입")

    #이미지 규격 통일(VLM이 이해할 수 있는 규격으로) 
    def _to_hwc_u8(self, image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)
        if arr.ndim != 3:
            arr = np.zeros(
                (int(self.image_height), int(self.image_width), 3),
                dtype=np.uint8,
            )
        if arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        if arr.dtype != np.uint8:
            max_v = float(arr.max()) if arr.size else 0.0
            if max_v <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
        return arr

    def get_head_pose(self) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Return configured head pose as world position and scalar-first quaternion."""
        data = self.controller.wrench_sim.data
        
        if self.head_body_id >= 0: # Body인 경우
            pos = np.asarray(data.xpos[self.head_body_id], dtype=np.float32)
            quat = np.asarray(data.xquat[self.head_body_id], dtype=np.float32)
            return pos.astype(np.float32), np.asarray(quat, dtype=np.float32)
        if self.head_site_id >= 0: # Site인 경우
            pos = np.asarray(data.site_xpos[self.head_site_id], dtype=np.float32)
            quat = R.from_matrix(
                np.asarray(data.site_xmat[self.head_site_id], dtype=np.float32).reshape(
                    3, 3
                )
            ).as_quat(scalar_first=True)
            return pos.astype(np.float32), np.asarray(quat, dtype=np.float32)
        raise ValueError(
            f"Head pose source '{self.head_name}' not found as site/body. "
            "Configure ComplianceConfig.head_name in *_vlm.gin."
        )

    def _get_stereo_images(self,obs: Any = None) -> tuple[np.ndarray, np.ndarray]:
        if obs is not None:
                # hasattr 대신 dict나 object 속성을 직접 체크
                left = getattr(obs, "left_image", None)
                right = getattr(obs, "right_image", None)
                
                if left is not None:
                    # print("DEBUG: Simulation Images Found!") # 주석 해제하여 확인
                    return self._to_hwc_u8(left), self._to_hwc_u8(right if right is not None else left)
        
        if self.left_camera is not None:
            try:
                self.last_left_frame = self.left_camera.get_frame()
                print(f"left camera get!")
            except Exception:
                pass
        if self.right_camera is not None:
            try:
                self.last_right_frame = self.right_camera.get_frame()
                print(f"right camera get!")
            except Exception:
                pass

        left = self.last_left_frame
        right = self.last_right_frame if self.last_right_frame is not None else left

        if left is None:
            left = np.zeros(
                (int(self.image_height), int(self.image_width), 3), dtype=np.uint8
            )
            right = left
        elif right is None:
            right = left

        return self._to_hwc_u8(np.asarray(left)), self._to_hwc_u8(np.asarray(right))

# VLM(Gemini)를 통해 실시간으로 물체의 위치를 파악하고 새로운 경로를 생성하는 과정을 담당
    def run_prediction_pipeline(
        self,
        head_pos: np.ndarray,
        head_quat: np.ndarray,
        left_image: np.ndarray,
        right_image: np.ndarray,
        output_dir: Optional[str],
        pose_cur_by_site: Dict[str, np.ndarray],
        site_names: List[str],
        task: str,
        object_label: str,
    ) -> Optional[Dict[str, Tuple[np.ndarray, ...]]]:
        
        if self.predictor is None:
            return None
        
        is_wiping = task == "wiping"
        print(f"is_wiping : {is_wiping}")
        # VLM이 카메라image와 명령어를 VLM에게 전달 -> 목표 라벨에 대한 2D/3D 어포던스(점과 방향)을 예측
        # 3D 좌표값을 찍기위해서는 "stereo matching"방법이 필요 -> 하나의 스테레오 카메라내부의 2개의 렌즈에서 오는 Image를 뜻함
        prediction = self.predictor.predict(
            left_image=left_image,
            right_image=right_image,
            robot_name=self.robot,
            site_names=site_names,
            task=is_wiping,
            output_dir=output_dir,
            object_label=object_label,
        )
        if prediction is None:
            return None

        # 데이터 검증
        if not (
            isinstance(prediction, tuple)
            and len(prediction) == 2
            and isinstance(prediction[0], dict)
            and isinstance(prediction[1], dict)
        ):
            print(
                "[ComplianceVLM] prediction failed: expected (contact_points_dict, "
                f"contact_normals_dict), got {type(prediction).__name__}"
            )
            return None
        raw_points, raw_normals = prediction
        valid_sites = [
            s
            for s in site_names
            if s in raw_points and s in raw_normals and s in pose_cur_by_site
        ]
        contact_points = {
            site_name: np.asarray(raw_points[site_name], dtype=np.float32)
            for site_name in valid_sites
        }
        contact_normals = {
            site_name: np.asarray(raw_normals[site_name], dtype=np.float32)
            for site_name in valid_sites
        }

        if not contact_points:
            return None
        
        # 경로 생성
        return plan_end_effector_poses(
            contact_points_camera=contact_points,
            contact_normals_camera=contact_normals,
            head_position_world=np.asarray(head_pos, dtype=np.float32),
            head_quaternion_world_wxyz=np.asarray(head_quat, dtype=np.float32),
            tangent_pos_stiffness=float(self.kp_pos_tangent),
            normal_pos_stiffness=float(self.kp_pos_normal),
            tangent_rot_stiffness=float(self.kp_rot_tangent),
            normal_rot_stiffness=float(self.kp_rot_normal),
            contact_force=float(self.fixed_contact_force),
            pose_cur=pose_cur_by_site,
            output_dir=output_dir,
            traj_dt=float(self.control_dt),
            traj_v_max_contact=0.02,
            traj_v_max_free=0.1,
            tool=TASK_TOOL_MAP.get(task, "gripper"),
            robot_name=self.robot,
            task=TASK_PLAN_MAP.get(task, task),
            mass=float(self.mass),
            inertia_diag=np.asarray(self.inertia_diag, dtype=np.float32),
        )

    # 예측이 필요할 때를 판단해서 예측을 시키게 하는 함수
    def maybe_start_prediction(self, obs: Any, has_fixed_trajectory: bool) -> None:
        if self.status == "waiting": 
            return
        if self.prediction_future is not None:
            return
        if self.predictor is None:
            return
        if self.status not in ACTIVE_TASKS:   # ← 정의되지 않은 task 방어
            return
        # 모든 task 공통 진입 조건
        if not self.prediction_requested and self.trajectory_plans:
            return
        if has_fixed_trajectory and not self.trajectory_plans:
            return

        left_image, right_image = self._get_stereo_images(obs) # Img Get
        head_pos, head_quat = self.get_head_pose() # 눈의 위치 GET
        output_dir = self.get_prediction_output_dir(self.prediction_counter) 
        site_names = list(self.target_site_names) 
        object_label = str(self.target_object_label)

        pose_cur = {  # 손의 위치를 파악
            site_name: np.asarray(
                self.pose_command[self.wrench_site_names.index(site_name)],
                dtype=np.float32,
            )
            for site_name in site_names
            if site_name in self.wrench_site_names
        }
        # submit : 로봇의 메인 제어 loop에서 VLM의 연산을 다른 스레드에 보내는 부분
        # ThreadPoolExecutor를 활용하여 사전에 work thread를 미리 생성하여 대기하고 있을 때, submit으로 호출될 때, 
        # run_prediction_pipeline함수와 인자의 재료들을 한번에 묶어서 '참조'방식으로 전달(작업이 완료되었는지는 Polling방식으로 확인)
        self.prediction_future = self.prediction_executor.submit(
            self.run_prediction_pipeline,
            head_pos,
            head_quat,
            left_image,
            right_image,
            output_dir,
            pose_cur,
            site_names,
            self.status,
            object_label,
        )
        self.prediction_counter += 1
        self.prediction_requested = False


    def request_prediction_after_completion(self) -> None: # 다음 작업 예약 (닦기 시, VLM에게 재질문을 하여 닦는 작업을 한번 더 할 수 있음)
        if self.status == "wiping":
            self.prediction_requested = True

    def _consume_prediction(self, obs_time: Optional[float],sim: Any = None) -> None: # 결과 확인 및 궤적 교체 -> VLM답변 -> 로봇의 실제 궤적
        
        # Poliing방식
        if self.prediction_future is None:
            return
        if not self.prediction_future.done():
            return
        
        try:
            result = self.prediction_future.result() # VLM의 결과(점과 법선벡터)로 가게 하는 궤적을 GET
        
        except Exception as exc:
            print(f"[ComplianceVLM] prediction failed: {exc}")
            result = None
        
        finally:
            self.prediction_future = None

        if result is None:
            if self.status == "wiping":
                now = float(obs_time) if obs_time is not None else 0.0
                self.wipe_pause_end_time = now + self.wipe_pause_duration
            elif self.status == "picking":
                print("[Pick] 궤적 생성 실패, 대기 복귀")
                self.status = "waiting"
            return

        self.trajectory_plans = result
        self.traj_start_time = None

    def _apply_trajectory(self, now: float) -> None:
        if not self.trajectory_plans:
            return
        if self.traj_start_time is None:
            self.traj_start_time = float(now)

        elapsed = max(0.0, float(now) - float(self.traj_start_time)) # 시간계산(시작점붙 지금까지 흐른 시간 -> 이 시간을 기준으로 스케줄표의 몇 번째 줄을 읽어야할 지 결정)
        indices: Dict[str, Tuple[int, int]] = {}

        for site_name in self.target_site_names:
            if site_name not in self.wrench_site_names:
                continue
            plan = self.trajectory_plans.get(site_name)
            if plan is None:
                continue

            (
                time_samples,
                _,
                ee_pos,
                ee_ori,
                pos_stiffness,
                rot_stiffness,
                pos_damping,
                rot_damping,
                command_forces,
            ) = plan

            # 스케쥴표 조회 
            idx = np.searchsorted(time_samples, elapsed, side="right") - 1
            idx = int(np.clip(idx, 0, len(time_samples) - 1))
            indices[site_name] = (idx, len(time_samples))

            # 찾은 idx를 기준으로 그 시점에 맞는 desired값 추출
            site_idx = self.wrench_site_names.index(site_name)
            self.pose_command[site_idx, 0:3] = np.asarray(ee_pos[idx], dtype=np.float32)
            self.pose_command[site_idx, 3:6] = np.asarray(ee_ori[idx], dtype=np.float32)
            self.pos_stiffness[site_idx] = np.asarray(
                pos_stiffness[idx], dtype=np.float32
            ).reshape(-1)
            self.rot_stiffness[site_idx] = np.asarray(
                rot_stiffness[idx], dtype=np.float32
            ).reshape(-1)
            self.pos_damping[site_idx] = np.asarray(
                pos_damping[idx], dtype=np.float32
            ).reshape(-1)
            self.rot_damping[site_idx] = np.asarray(
                rot_damping[idx], dtype=np.float32
            ).reshape(-1)
            # 로봇이 물리적으로 가해야하는 실제 힘
            self.wrench_command[site_idx, 0:3] = np.asarray(
                command_forces[idx], dtype=np.float32
            )

        # Base CompliancePolicy.step() rebuilds pose_command from base_pose_command
        # each cycle; keep base_pose in sync with replay references.
        self.base_pose_command = np.asarray(self.pose_command, dtype=np.float32).copy()

        if indices and all(idx >= length - 1 for idx, length in indices.values()):
            cfg = TASK_COMPLETION_CONFIG.get(self.status, {})

            if cfg.get("loop") and not self.fixed_trajectory_active:
                # wiping VLM 실시간 모드: 일시 정지 후 재예측
                self.wipe_pause_end_time = float(now) + self.wipe_pause_duration
            
            else:
                # 그 외 모든 task: waiting으로 복귀
                self.status = "waiting"
                if self.fixed_trajectory_active:
                    self.wipe_pause_end_time = None
                    self.fixed_trajectory_active = False

            # task별 완료 액션 플래그 설정
            if cfg.get("on_complete_action") is not None:
                self._pending_complete_action = cfg["on_complete_action"]

            self.trajectory_plans = {}
            self.traj_start_time  = None

    def start_video_logging(self) -> None:
        if self.video_logging_active or not self.record_video:
            return
        self.video_temp_dir = tempfile.TemporaryDirectory(
            prefix="compliance_vlm_video_"
        )
        self.video_path = Path(self.video_temp_dir.name) / "camera.mp4"
        self.video_fps = max(1.0, 1.0 / max(self.control_dt, 1e-3))
        self.video_logging_active = True

    def ensure_video_writer(self, frame: np.ndarray) -> bool:
        if self.video_writer is not None:
            return True
        if self.video_path is None:
            return False
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(self.video_path),
            fourcc,
            float(self.video_fps if self.video_fps is not None else 30.0),
            (int(frame.shape[1]), int(frame.shape[0])),
        )
        if not writer.isOpened():
            writer.release()
            self.video_logging_active = False
            return False
        self.video_writer = writer
        return True

    def log_camera_frame(
        self, timestamp_s: float, left: np.ndarray, right: np.ndarray
    ) -> None:
        if not self.video_logging_active:
            return
        frame = np.hstack([left, right])
        if not self.ensure_video_writer(frame):
            return
        assert self.video_writer is not None
        self.video_writer.write(frame)
        self.video_frame_timestamps.append(float(timestamp_s))

    def export_camera_video(self, exp_dir: Path) -> None:
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        if self.video_path is None or not self.video_path.exists():
            return
        exp_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.video_path, exp_dir / "camera.mp4")

        timestamp_path = exp_dir / "camera_timestamps.json"
        with timestamp_path.open("w", encoding="utf-8") as f:
            json.dump(self.video_frame_timestamps, f, indent=2)

    def discard_video_recording(self) -> None:
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        if self.video_temp_dir is not None:
            self.video_temp_dir.cleanup()
            self.video_temp_dir = None
        self.video_path = None

    def step(
        self,
        obs: Any,
        sim: Any,
    ) -> npt.NDArray[np.float32]:
        # 부모 클래스 호출(compliancePolicy)을 통해 기본적인 어드미턴스 제어 수식을 먼저 실행
        # TODO : 기본 Control loop를 먼저 돌림 -> action(관절 토크, 목표 관절 각도)

        action = np.asarray(super().step(obs, sim), dtype=np.float32)
        

        self.check_mode_command()
        

        # 로봇이 초기자세를 잡는 중이라면, VLM로직을 건너 뛰고, 기본 동작만 수행함
        if not bool(getattr(self, "is_prepared", False)):
            return np.asarray(action, dtype=np.float32)

        prep_duration = float(getattr(self, "prep_duration", 0.0))
        if float(obs.time) < prep_duration:
            return np.asarray(action, dtype=np.float32)

        # 목을 고정시켜 카메라가 흔들리지 않게 고정!
        if self.neck_pitch_idx is not None:
            action[self.neck_pitch_idx] = np.float32(
                self.ref_motor_pos[self.neck_pitch_idx]
            )

        # 한번 닦기를 완료했으면, 다 닦였나?를 확인하기 위해 VLM을 호출하여 다시 물어볼 준비
        if self.status == "waiting":
            self.wipe_pause_end_time = None
            return np.asarray(action, dtype=np.float32)

        if self.status != "wiping":
            self.wipe_pause_end_time = None

        if self.status == "wiping" and self.wipe_pause_end_time is not None:
            if float(obs.time) >= float(self.wipe_pause_end_time):
                self.wipe_pause_end_time = None
                self.trajectory_plans = {}
                self.traj_start_time = None
                self.request_prediction_after_completion()
            else:
                self._consume_prediction(float(obs.time))
                return np.asarray(action, dtype=np.float32)


        has_fixed_trajectory = self.can_use_replay_for_current_mode() # 녹화 데이터 사용가능한 상황인지 파악
        if self.replay_task is not None and not has_fixed_trajectory:
            if not self.replay_unavailable_reported:
                current_task = (
                    "wipe"
                    if self.status == "wiping"
                    else ("draw" if self.status == "drawing" else "none")
                )
                print(
                    "[ComplianceVLM] replay not active for current state: "
                    f"current_task={current_task}, "
                    f"replay_task={self.replay_task}, "
                    f"target_sites={self.target_site_names}, "
                    f"replay_sites={self._replay_site_names()}"
                )
                self.replay_unavailable_reported = True
        elif has_fixed_trajectory:
            self.replay_unavailable_reported = False
        
        # 경로 준비
        if has_fixed_trajectory and not self.trajectory_plans: # 리플레이 모드인데 경로 계획이 empty라면,
            self.prepare_fixed_plan() # .lz4파일에서 데이터를 읽어와 경로를 저장

        self.maybe_start_prediction(obs, has_fixed_trajectory) # VLM에게 예측을 시킬지 판단
        self._consume_prediction(float(obs.time),sim) # VLM이 답변을 다 했는지 확인 및 답변이 왔다면 새 경로로 경로 계확
        self._apply_trajectory(float(obs.time)) 

        left, right = self._get_stereo_images(obs)
        if self.record_video:
            self.start_video_logging()
            self.log_camera_frame(float(obs.time), left, right)

        if getattr(self, "_pending_complete_action", None) is not None:
            act = self._pending_complete_action
            action[act["index"]] = np.float32(act["value"])
            self._pending_complete_action = None
            
        return np.asarray(action, dtype=np.float32)

    def close(self, exp_folder_path: str = "") -> None:
        if self.prediction_future is not None:
            self.prediction_future.cancel()
            self.prediction_future = None
        self.prediction_executor.shutdown(wait=False)

        if self.left_camera is not None:
            try:
                self.left_camera.close()
            except Exception:
                pass
            self.left_camera = None
        if self.right_camera is not None:
            try:
                self.right_camera.close()
            except Exception:
                pass
            self.right_camera = None

        if self.debug_output_dir is not None and exp_folder_path:
            exp_dir = Path(exp_folder_path)
            exp_dir.mkdir(parents=True, exist_ok=True)
            src = Path(self.debug_output_dir.name)
            if src.exists():
                for item in src.iterdir():
                    dst = exp_dir / item.name
                    if dst.exists():
                        if dst.is_dir():
                            shutil.rmtree(dst)
                        else:
                            dst.unlink()
                    if item.is_dir():
                        shutil.copytree(item, dst)
                    else:
                        shutil.copy2(item, dst)

        if exp_folder_path:
            self.export_camera_video(Path(exp_folder_path))
        self.discard_video_recording()

        if self.debug_output_dir is not None:
            self.debug_output_dir.cleanup()
            self.debug_output_dir = None

        super().close(exp_folder_path)
