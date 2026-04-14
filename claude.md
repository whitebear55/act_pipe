이 프로젝트의 목적은 어드미턴스 제어기를 모방학습과 결합하여 닦기와 같은 접촉이 빈번한 task를 수행하는 것이야
모든 코드의 흐름은 폴더내부에 있는 minimalist compliance control논문을 바탕으로 구현
사용자의 허락 없이 새로운 파일을 추가하지 마
코드에서 에러가 발생 시 이를 정리하여 /log폴더를 만들어서 기록

프로젝트 수행 절차는 아래와 같아

1. 사용하는 로봇에 대한 정보를 담고있는 xml,yml,gin파일 작성
@description_claude.md 참조

2. 시뮬레이션(MuJoCo)에서 어드미턴스 제어가 적용되는지 확인(run_policy --robot aloha --sim mujoco --policy compliance --vis none)

3. 실제 로봇에서 어드미턴스 제어 적용
@real_world_claude.md 참조

4. 모방학습을 위한 데이터 수집 시 원격조종의 제어기로 어드미턴스 제어기 결합
@admittance_teleop_claude.md 참조
 
 5. 시연 데이터를 수집하기 위해 데이터 저장
 원격조종을 통해 시연데이터를 수집할 때는, x_des(사용자가 명령을 한 EE pos)와 x_ref(사용자가 명령한 EE pos를 추종하지만 어드미턴스 제어기를 통해 실제 EE가 위치한 pos)를 저장해야함
 
 6. 시연데이터를 바탕으로 diffusion방법으로 AI모델을 학습
 
 7. 학습된 AI모델 추론 테스트
 테스트를 진행할 때도 마찬가지로 AI모델이 내놓은 x_des값을 추종하되 실제 추종할 때는 어드미턴스 제어기를 통과시켜서 제어를 진행해야함(해당 부분의 코드를 작성할 때는 policy/compliance_dp.py)의 흐름과 비슷하게 작성)
 
