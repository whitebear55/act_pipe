# test_cam_side_launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterFile


def generate_launch_description():

    cam_side_name_arg = DeclareLaunchArgument(
        'cam_side_name',
        default_value='cam_side',
        description='Name of the side camera',
    )

    cam_side_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        namespace=LaunchConfiguration('cam_side_name'),
        name='camera',
        parameters=[
            ParameterFile(
                param_file=PathJoinSubstitution([
                    FindPackageShare('usb_cam'),
                    'config',
                    'params_1.yaml',
                ]),
                allow_substs=True,
            )
        ],
        output='screen',
    )

    return LaunchDescription([
        cam_side_name_arg,
        cam_side_node,
    ])
