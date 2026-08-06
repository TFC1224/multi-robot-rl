"""ROS 2 distributed actor node.

Each ROSLanders runs one instance of this node. All instances share the
*same* ``DistributedActor`` weights — pure CTDE means execution needs no
inter-robot communication. The node subscribes to its local map and
pose, builds the per-agent observation, runs the actor, and publishes a
``PoseStamped`` goal that the existing ``move_base`` action server
consumes.
"""

from __future__ import annotations

import sys
from typing import Optional

import numpy as np
import rclpy
import torch
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node

from .frontier_detector import detect_frontiers
from .multi_agent_env import LOCAL_MAP_SIZE, MAX_FRONTIERS, N_AGENTS
from .networks import DistributedActor
from .observation import build_observation, extract_local_map


def _to_numpy(msg: OccupancyGrid) -> np.ndarray:
    """Convert an OccupancyGrid message to a (H, W) uint8 numpy array."""
    H, W = msg.info.height, msg.info.width
    data = np.asarray(msg.data, dtype=np.int8).reshape(H, W)
    out = np.full((H, W), 2, dtype=np.uint8)
    out[data == 0] = 0
    out[data >= 50] = 1
    return out


def _pose_to_rc(msg: OccupancyGrid, pose: PoseStamped,
                resolution: float) -> tuple[int, int]:
    info = msg.info
    x = pose.pose.position.x - info.origin.position.x
    y = pose.pose.position.y - info.origin.position.y
    col = int(x / max(resolution, 1e-3))
    row = int(y / max(resolution, 1e-3))
    return row, col


class AgentNode(Node):
    """One ROS 2 node per robot. Loads the shared actor weights."""

    def __init__(self, robot_id: int, model_path: str, device: str = 'cpu'):
        super().__init__(f'agent_node_{robot_id}')

        # ---- model ----
        self.robot_id = robot_id
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model = DistributedActor(
            n_actions=MAX_FRONTIERS, n_teammates=N_AGENTS - 1,
        ).to(self.device)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        self.get_logger().info(
            f'Loaded actor from {model_path} (robot_id={robot_id})')

        # ---- state ----
        self.local_map_msg: Optional[OccupancyGrid] = None
        self.shared_map_msg: Optional[OccupancyGrid] = None
        self.local_pose: Optional[PoseStamped] = None
        self.teammate_poses: list[Optional[PoseStamped]] = [None] * (N_AGENTS - 1)
        self.timer = self.create_timer(2.0, self._decision_callback)

        # ---- I/O ----
        self.local_map_sub = self.create_subscription(
            OccupancyGrid, f'/robot_{robot_id + 1}/map',
            self._local_map_cb, 10)
        self.shared_map_sub = self.create_subscription(
            OccupancyGrid, '/shared_map', self._shared_map_cb, 10)
        self.local_pose_sub = self.create_subscription(
            PoseStamped, f'/robot_{robot_id + 1}/amcl_pose',
            self._local_pose_cb, 10)
        # Subscribe to all other robots' amcl_pose topics
        for j in range(N_AGENTS):
            if j == robot_id:
                continue
            self.create_subscription(
                PoseStamped, f'/robot_{j + 1}/amcl_pose',
                lambda msg, j=j: self._teammate_pose_cb(msg, j), 10)
        self.goal_pub = self.create_publisher(
            PoseStamped, f'/robot_{robot_id + 1}/move_base_simple/goal', 10)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _local_map_cb(self, msg: OccupancyGrid) -> None:
        self.local_map_msg = msg

    def _shared_map_cb(self, msg: OccupancyGrid) -> None:
        self.shared_map_msg = msg

    def _local_pose_cb(self, msg: PoseStamped) -> None:
        self.local_pose = msg

    def _teammate_pose_cb(self, msg: PoseStamped, j: int) -> None:
        # Re-index to [0..N_AGENTS-1] excluding self
        slots = [k for k in range(N_AGENTS) if k != self.robot_id]
        if j in slots:
            self.teammate_poses[slots.index(j)] = msg

    def _decision_callback(self) -> None:
        if self.shared_map_msg is None or self.local_pose is None:
            return

        with torch.no_grad():
            obs = self._build_observation()
            tensors = self._obs_to_tensors(obs)
            logits = self.model(**tensors)
            action = int(torch.argmax(logits, dim=-1).item())

        shared = _to_numpy(self.shared_map_msg)
        r, c, _ = self._extract_pose_rc()
        fronts, n_valid = detect_frontiers(
            shared, (r, c), max_n=MAX_FRONTIERS)
        if action >= n_valid:
            return
        # De-normalize frontier target
        rr = int(round(fronts[action, 0] * (shared.shape[0] - 1)))
        cc = int(round(fronts[action, 1] * (shared.shape[1] - 1)))
        goal = PoseStamped()
        goal.header.frame_id = 'world'
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = float(
            self.shared_map_msg.info.origin.position.x
            + cc * self.shared_map_msg.info.resolution)
        goal.pose.position.y = float(
            self.shared_map_msg.info.origin.position.y
            + rr * self.shared_map_msg.info.resolution)
        goal.pose.orientation.w = 1.0
        self.goal_pub.publish(goal)
        self.get_logger().info(
            f'Robot {self.robot_id + 1} -> frontier {action} '
            f'({goal.pose.position.x:.2f}, {goal.pose.position.y:.2f})')

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_pose_rc(self) -> tuple[int, int, float]:
        pose = self.local_pose.pose
        info = self.shared_map_msg.info
        x = pose.position.x - info.origin.position.x
        y = pose.position.y - info.origin.position.y
        col = int(x / max(info.resolution, 1e-3))
        row = int(y / max(info.resolution, 1e-3))
        # Compute yaw from quaternion
        q = pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        theta = float(np.arctan2(siny_cosp, cosy_cosp))
        return row, col, theta

    def _build_observation(self) -> dict[str, np.ndarray]:
        """Build observation using the shared observation module."""
        shared = _to_numpy(self.shared_map_msg)
        H, W = shared.shape
        r, c, theta = self._extract_pose_rc()

        local_map = extract_local_map(shared, (r, c), local_map_size=LOCAL_MAP_SIZE)

        # Resolve teammate poses (with missing-data tolerance)
        teammate_poses = []
        for msg in self.teammate_poses:
            if msg is None:
                teammate_poses.append(None)
                continue
            pose = msg.pose
            info = self.shared_map_msg.info
            tx = pose.position.x - info.origin.position.x
            ty = pose.position.y - info.origin.position.y
            tr = int(ty / max(info.resolution, 1e-3))
            tc = int(tx / max(info.resolution, 1e-3))
            q = pose.orientation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            t_theta = float(np.arctan2(siny_cosp, cosy_cosp))
            teammate_poses.append((tr, tc, t_theta))

        return build_observation(
            local_map, (r, c, theta), (H, W),
            teammate_poses=teammate_poses,
            shared_occupancy=shared,
            n_agents=N_AGENTS,
            max_frontiers=MAX_FRONTIERS,
        )

    def _obs_to_tensors(self, obs: dict) -> dict[str, torch.Tensor]:
        return {
            'local_map': torch.from_numpy(obs['local_map']).unsqueeze(0).to(self.device),
            'own_pose': torch.from_numpy(obs['own_pose']).unsqueeze(0).to(self.device),
            'teammates': torch.from_numpy(obs['teammates']).unsqueeze(0).to(self.device),
            'frontiers': torch.from_numpy(obs['frontiers']).unsqueeze(0).to(self.device),
            'n_frontiers': torch.tensor(
                [obs['n_frontiers']], dtype=torch.long, device=self.device),
        }


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--robot_id', type=int, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    args, ros_args = parser.parse_known_args(argv if argv is not None else sys.argv[1:])

    rclpy.init(args=ros_args)
    node = AgentNode(args.robot_id, args.model_path, device=args.device)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
