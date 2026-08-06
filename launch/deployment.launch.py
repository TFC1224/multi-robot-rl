"""Launch N distributed agent nodes (pure CTDE execution).

Uses OpaqueFunction to properly resolve LaunchConfiguration arguments at
runtime, avoiding type errors with ``.perform(None)``.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_nodes(context, *args, **kwargs):
    """Build agent Node list dynamically from launch arguments."""
    n_agents = int(LaunchConfiguration('n_agents').perform(context))
    model_path = LaunchConfiguration('model_path').perform(context)
    device = LaunchConfiguration('device').perform(context)

    nodes = []
    for i in range(n_agents):
        nodes.append(Node(
            package='multi_robot_rl',
            executable='agent_node',
            name=f'agent_node_{i + 1}',
            output='screen',
            parameters=[{
                'robot_id': i,
                'model_path': model_path,
                'device': device,
            }],
        ))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'model_path', default_value='models/mappo_actor.pth',
            description='Path to the trained DistributedActor weights.'),
        DeclareLaunchArgument(
            'n_agents', default_value='3',
            description='Number of robots to launch (default 3).'),
        DeclareLaunchArgument(
            'device', default_value='cpu',
            description='Inference device for the actor.'),
        OpaqueFunction(function=_launch_nodes),
    ])
