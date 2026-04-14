#!/usr/bin/env python3

from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    robot_shutdown,
    robot_startup,
)
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

def main():
    # 1. ROS 2 통신 노드 생성 (필수)
    node = create_interbotix_global_node('aloha')

    # 2. 로봇 객체 생성
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
    
    # 3. 통신 시작 (필수)
    robot_startup(node)

    bots_to_read = [follower_bot_left, follower_bot_right]

    # 4. 토크 끄기 (수동 정렬을 위해 모터 힘 빼기)
    print("모든 팔의 토크를 해제합니다...")
    for bot in bots_to_read:
        # Interbotix API를 이용해 직접 토크 해제
        bot.core.robot_torque_enable("group", "all", False)

    # 5. 수동 정렬 대기 (핵심 포인트!)
    print("\n로봇의 토크가 풀렸습니다. 손으로 움직일 수 있습니다.")
    input("👉 로봇 관절을 0점(정렬 마크)에 완벽히 맞춘 뒤 [Enter] 키를 누르세요...")

    # 6. 엔코더 값 읽기 (추출하신 로직)
    print("\n[현재 위치 엔코더 Raw Ticks]")
    for bot in bots_to_read:
        name = bot.core.robot_name
        try:
            raw_ticks = bot.core.robot_get_motor_registers("group", "all", "Present_Position")
            print(f"{name}: {raw_ticks}")
        except Exception as e:
            print(f"{name} 값을 읽는데 실패했습니다: {e}")

    # 7. 안전하게 종료
    robot_shutdown(node)

if __name__ == '__main__':
    main()
