src/aloha폴더는 모바일 알로하를 이용하여 ACT알고리즘을 활용해 실제로 모방학습을 진행했던 코드들이 있음. dual_side_teleop.py파일을 보면 현재는 리더암(widowX 250s)의 관절각도를 읽어서 그대로 팔로워(viper 300s)암의 관절각도로 매핑하는 방법을 사용했어
그러나, 어드미턴스 제어를 위한 함수(minimalist_compliance_control/controller_ref.py의 integrate_commands)를 보면 다음 시점의 목표 위치를 계산하여 이를 추종하는 흐름으로 어드미턴스 제어가 진행돼
따라서, 해당 프로젝트에서는 dual_side_teleop코드와 같이 joint-joint형식으로 하면 안돼
 매 time step마다 리더암의 관절각도를 바탕으로 팔로워암의 xml에 넣어서 무조코의 물리엔진 함수를 통해 EE의 pos값을 계산하고, 이를 추종하기 위해서 IK solver를 사용하면 돼
 
 여기서, 리더암의 xml을 이용하지 않고 팔로워암의 xml로 계산하는 이유는 mobile aloha특성 상 리더암과 팔로워 암의 관절각도가 그대로 매핑이 되기 때문이야
 다만, 그리퍼의 관절 각도 범위는 다르기 때문에 그리퍼 열고 닫는 것은 항상 constant.py에 정의된 변환 함수를 사용해야해
 
 본 논문에서는 ROS를 사용하지 않고, 바로 다이나믹셀모터의 sdk를 이용해서 진행했지만 나는 ROS2를 이용하고, interbortix의 라이브러리도 이용해서 프로젝트를 구현중이야
 
 실제 로봇팔 제어를 위해 current_based_position 모드로 제어하지 말고 position 모드로 제어하도록 해(진동 억제를 위해)
