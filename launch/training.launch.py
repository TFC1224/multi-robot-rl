"""Launch the MAPPO trainer (no ROS 2 daemon required)."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    scenario_arg = DeclareLaunchArgument(
        'scenario', default_value='multi_1',
        description='One of multi_1 / multi_2 / multi_3.')
    steps_arg = DeclareLaunchArgument(
        'total_timesteps', default_value='200000',
        description='Total environment timesteps to train.')
    save_dir_arg = DeclareLaunchArgument(
        'save_dir', default_value='models',
        description='Directory to store checkpoints.')
    device_arg = DeclareLaunchArgument(
        'device', default_value='auto',
        description='cuda / cpu / auto')

    trainer_cmd = ExecuteProcess(
        cmd=[
            'python3',
            '-m',
            'multi_robot_rl.scripts.train_mappo',
            '--scenario', LaunchConfiguration('scenario'),
            '--total_timesteps', LaunchConfiguration('total_timesteps'),
            '--save_dir', LaunchConfiguration('save_dir'),
            '--device', LaunchConfiguration('device'),
        ],
        name='mappo_trainer',
        output='screen',
    )

    return LaunchDescription([
        scenario_arg,
        steps_arg,
        save_dir_arg,
        device_arg,
        trainer_cmd,
    ])
