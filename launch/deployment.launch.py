"""Launch 3 distributed agent nodes (pure CTDE execution)."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    model_path_arg = DeclareLaunchArgument(
        'model_path', default_value='models/mappo_actor.pth',
        description='Path to the trained DistributedActor weights.')
    n_agents_arg = DeclareLaunchArgument(
        'n_agents', default_value='3',
        description='Number of robots to launch (default 3).')
    device_arg = DeclareLaunchArgument(
        'device', default_value='cpu',
        description='Inference device for the actor.')

    n_agents = int(LaunchConfiguration('n_agents').perform(None))
    nodes = []
    for i in range(n_agents):
        nodes.append(Node(
            package='multi_robot_rl',
            executable='agent_node',
            name=f'agent_node_{i + 1}',
            output='screen',
            parameters=[{
                'robot_id': i,
                'model_path': LaunchConfiguration('model_path'),
                'device': LaunchConfiguration('device'),
            }],
        ))

    return LaunchDescription([
        model_path_arg,
        n_agents_arg,
        device_arg,
        *nodes,
    ])
