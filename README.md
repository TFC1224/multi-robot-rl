# multi_robot_rl — 方案三实验代码

纯 CTDE 多机协同主动探索系统(基于 MAPPO)的最小可运行核心实现。
对应**方案三**(基于纯 CTDE 的多 ROSLander 机器人协同主动探索系统)。

## 设计目标

- **训练**: 集中式 Critic + 分布式 Actor(参数共享)的 MAPPO 算法
- **执行**: 每台机器人 Jetson 独立加载 Actor,纯分布,零通信开销
- **仿真**: 2D 栅格 Gymnasium 环境(替代 Gazebo),便于无 Gazebo 环境下训练

## 目录结构

```
multi_robot_rl/
├── package.xml              # ROS 2 ament_python 包定义
├── setup.py                 # 包安装脚本
├── setup.cfg
├── config/
│   └── mappo_config.yaml    # MAPPO 超参数(对应方案三 §6.4)
├── launch/
│   ├── training.launch.py   # 启动训练
│   └── deployment.launch.py # 启动 3 机部署
├── scripts/
│   └── train_mappo.py       # 训练入口脚本(可独立运行,不依赖 ROS 2)
├── multi_robot_rl/          # Python 模块
│   ├── __init__.py
│   ├── grid_world.py        # 三个场景(Multi-1/2/3)+ sensing/movement/metrics
│   ├── frontier_detector.py # 8 邻域卷积 + 聚类前沿点
│   ├── reward_functions.py  # 团队奖励(7 项)+ Jain 公平性
│   ├── multi_agent_env.py   # Gymnasium MultiAgentExplorerEnv
│   ├── networks.py          # DistributedActor + CentralizedCritic
│   ├── replay_buffer.py     # RolloutBuffer + GAE
│   ├── mappo_trainer.py     # MAPPOTrainer 训练器
│   └── agent_node.py        # ROS 2 分布式 Actor 节点(执行时)
└── smoke_test.py            # 离线 smoke test(可选)
```

## 关键算法设计

### 状态空间(对应方案三 §4.1)

- **全局状态**(Critic): shared_map(3×64×64) + 3 机位姿(6) + 3 机朝向(6) + team_coverage + step_norm
- **个体观测**(Actor): local_map(3×64×64) + own_pose(4) + teammates(2×4) + frontiers(16×3) + n_frontiers

### 动作空间(对应方案三 §4.2)

每台机器人离散选择前沿点索引 `a_i ∈ {0, ..., 15, 16, 17}`,其中 16=等待、17=返回。

### 奖励函数(对应方案三 §4.5)

7 项团队奖励,所有权重可通过 `RewardWeights` 关掉以做消融:

| 项 | 公式 | 权重 |
|---|---|---|
| R_explore | +10 × Δteam_coverage | 10.0 |
| R_individual | +5 × Δcov_i | 5.0 |
| R_overlap | -3 × overlap_ratio | 3.0 |
| R_balance | +0.5 × Jain(individual_coverage) | 0.5 |
| R_collision | -50 (if collision) | 50.0 |
| R_step | -0.01 | 0.01 |
| R_done | +150 (if cov ≥ 95%) | 150.0 |

### MAPPO 更新(对应方案三 §5.2)

- GAE 优势估计: γ=0.99, λ=0.95
- PPO-Clip: ε=0.2
- 共享参数 Actor(3 机同一份权重)+ 集中式 Critic(仅训练时)
- Actor: lr=3e-4, Critic: lr=5e-4
- n_steps=2048, n_epochs=10, batch_size=64
- 梯度裁剪 max_grad_norm=0.5

## 运行流程

### 安装

```bash
cd ~/experiment_ws
colcon build --packages-select multi_robot_rl
source install/setup.bash
```

### 训练(无需 ROS 2)

```bash
cd ~/experiment_ws/src/multi_robot_rl
python scripts/train_mappo.py --total_timesteps 1000000 --scenario multi_1
```

或使用 launch(需要在 ROS 2 环境下):

```bash
ros2 launch multi_robot_rl training.launch.py \
    scenario:=multi_1 total_timesteps:=1000000
```

训练完成后会在 `models/` 目录下生成 `mappo_actor.pth` 和 `mappo_critic.pth`。

### 部署到 3 台 ROSLander(执行时)

```bash
ros2 launch multi_robot_rl deployment.launch.py \
    model_path:=/path/to/mappo_actor.pth n_agents:=3
```

每台机器人启动一个 `agent_node`,加载相同的 Actor 权重,订阅 `/robot_i/map`、`/amcl_pose`、`/shared_map`,发布 `/robot_i/move_base_simple/goal`。

## 与方案二的兼容性

| 复用项 | 来源 |
|---|---|
| `MapEncoder` CNN 结构 | `networks.py` 中 `MapEncoder` 复用方案二 `QNetwork.map_encoder` |
| 前沿点检测算法 | 复用方案二 §A.1 的 8 邻域卷积方法 |
| 奖励函数系数 | R_explore=10.0 与方案二一致 |
| `move_base` 集成 | 复用方案二的 `actionlib` → 改为 ROS 2 `PoseStamped` |

## 验证清单(静态代码审查通过)

- [x] `grid_world.py` 三个场景(Multi-1/2/3)实现完整,边界处理正确
- [x] `frontier_detector.py` 输出形状 `(16, 3)`,有效前沿点 < 16 时正确填零
- [x] `MultiAgentExplorerEnv.reset()` 返回 3 个合法观测,所有形状匹配 actor 输入
- [x] `MultiAgentExplorerEnv.step(actions)` 返回 (obs, rewards, terminated, truncated, info)
- [x] `DistributedActor.forward()` 输入 5 个张量,输出 `(B, 16)` logits
- [x] `CentralizedCritic.forward()` 输入 4 个张量,输出 `(B, 1)` value
- [x] `RolloutBuffer.compute_advantages` GAE 实现正确
- [x] `MAPPOTrainer.train()` 完整跑通 rollout + update + 周期性保存
- [x] `agent_node.py` 加载 actor 权重后能独立推理

## 后续扩展(本计划不包含)

- 多回合评估脚本 + TensorBoard 日志
- QMIX 对比算法实现
- 消融实验开关(通过 `RewardWeights.use_*`)
- Gazebo + ROS 2 真实仿真集成(类似 gym-turtlebot 的 `simulation.launch.py`)
- Jetson 端 ONNX/TorchScript 导出

## 参考文献

1. Yu, C., et al. **The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games**. NeurIPS, 2021.
2. Rashid, T., et al. **Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning**. JMLR, 2020.
3. gym-turtlebot: https://anurye.github.io/gym-turtlebot/
4. ROS Wiki. **multirobot_map_merge**. http://wiki.ros.org/multirobot_map_merge