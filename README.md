# multi-robot-rl — 基于 MAPPO 的多机器人协同前沿探索

3 台差速驱动机器人以团队形式执行基于前沿(frontier)的主动 SLAM 探索。
算法在自定义 2D 栅格 Gymnasium 环境下训练,采用 MAPPO (CTDE) 范式,
部署于 Jetson Orin NX 上作为三个独立的 Actor 节点运行。

---

## 1. 研究背景与问题定义

### 1.1 背景

自主探索是机器人学的基础问题之一。传统方法(如 tuples、BFS)在高维状态空间下
难以有效处理,在大面积场景(≥50 m²)中单机探索的时间开销尤为突出。
为解决这一问题,多机器人协同探索成为必然选择——多个机器人并行覆盖未知区域,
可显著缩短任务完成时间。

本项目以 **Ruan 等人(2021)** 提出的端到端深度强化学习探索框架为基础,
将多智能体协同引入未知室内场景的主动探索任务,
利用 MAPPO 算法实现集中式训练、分布式执行的协同探索策略。

### 1.2 任务定义

**任务**: 3 台差速驱动机器人在未知室内场景中协同完成基于前沿的主动探索。

**关键约束**: 决策时零通信(zero communication at decision time),即策略推理
仅依赖各机器人自身的局部观测,不依赖跨机器人通信。

### 1.3 评价指标

| 指标 | 目标值 | 定义 |
|---|---|---|
| 团队覆盖率 | ≥ 95% | 任意机器人观测到的空闲格子之并集 |
| 任务完成时间 | ≤ 90 s | 首次 `step()` 至 `terminated` 的时间 |
| 重复覆盖率 | ≤ 8% | 被 ≥2 台机器人观测到的格子 / 被 ≥1 台机器人观测到的总格子 |
| 碰撞次数 | 0 | 任意机器人对之间欧氏距离 ≤ 2 格 |
| Jain 公平性 | ≥ 0.85 | `J = (Σcov_i)² / (n·Σcov_i²)` |

---

## 2. 方法基础: CTDE 范式

### 2.1 背景与动机

单机 PPO 直接应用于多机场景时,面临**环境非平稳**(non-stationarity)问题——
同一状态下各智能体的邻居行为不断变化,导致值函数估计不稳定。

**CTDE (Centralized Training, Decentralized Execution)** 范式通过以下方式解决:

- **集中式 Critic**: 训练时使用全局状态 `s_global`,为每个智能体提供
  准确的值函数估计,消除非平稳性。
- **分布式 Actor**: 执行时各机器人仅依赖自身局部观测 `o_i`,
  不需要 Critic、不需要跨机器人通信。
- **零通信执行**: 协同行为从团队奖励信号和集中式 Critic 的梯度中自然涌现,
  无需显式的机器人间通信协议。

### 2.2 理论基础

本项目以以下工作为基础:

- **Yu et al. (NeurIPS 2021)**: MAPPO——PPO 在合作式多智能体环境中的有效应用,
  证明了在 CTDE 范式下 PPO 可以达到甚至超越专用多智能体算法的性能。
- **Ruan et al. (2021)**: 端到端 DRL 探索框架,利用内在动机驱动未知环境中的自主导航,
  为本项目的前沿检测和奖励设计提供了方法基础。
- **Yamauchi (1998)**: Frontier-based exploration 的经典范式,
  定义了前沿点的概念——已知空闲区域与未知区域之间的边界。

---

## 3. 核心架构: MAPPO 算法

### 3.1 模块组成

| 模块 | 文件 | 职责 |
|---|---|---|
| `DistributedActor` | `multi_robot_rl/networks.py` | 共享参数策略网络,输入局部观测,输出动作 logits |
| `CentralizedCritic` | `multi_robot_rl/networks.py` | 集中式值函数 `V(s_global)`,仅训练时使用 |
| `RolloutBuffer` | `multi_robot_rl/replay_buffer.py` | On-policy Rollout 缓冲,含 GAE 计算 |
| `MAPPOTrainer` | `multi_robot_rl/mappo_trainer.py` | 训练循环: 收集 → GAE → PPO 裁剪 → 保存 |

### 3.2 状态空间

**个体观测**(Actor 输入,5 项):

- `local_map`: 3 × 64 × 64 — 局部栅格地图(FREE / OCCUPIED / UNKNOWN)
- `own_pose`: 4 维 — `[x, y, cos θ, sin θ]`
- `teammates`: 2 × 4 维 — 其他两台机器人的位姿
- `frontiers`: 16 × 3 维 — 前沿候选槽位,`(row, col, cluster_size)`
- `n_frontiers`: 标量 — 有效前沿数量,取值范围 [0, 16]

**全局状态**(Critic 输入,4 项):

- `shared_map`: 3 × 64 × 64 — 融合后的共享地图
- `robot_positions`: 6 维 — 每台机器人的 (x, y) 坐标
- `robot_oris`: 6 维 — 每台机器人的 (cos θ, sin θ)
- `team_stats`: 2 维 — 当前覆盖率 + 归一化步数

### 3.3 动作空间

离散动作索引 `a_i ∈ {0..17}`:

| 动作 ID | 含义 |
|---|---|
| 0..15 | 前往前沿槽位 `a_i` |
| 16 | 原地等待(hold) |
| 17 | 返回初始位置 |

### 3.4 奖励函数

7 项团队奖励,各项系数可通过 `RewardWeights` 独立开关:

| 奖励项 | 公式 | 系数 |
|---|---|---|
| `R_explore` | +10 × Δteam_coverage | 10.0 |
| `R_individual` | +5 × Δcov_i (各机) | 5.0 |
| `R_overlap` | −3 × overlap_ratio | 3.0 |
| `R_balance` | +0.5 × Jain(individual_coverage) | 0.5 |
| `R_collision` | −50 若发生碰撞 | 50.0 |
| `R_step` | −0.01 | 0.01 |
| `R_done` | +150 若覆盖率 ≥ 95% | 150.0 |

其中 `overlap_ratio` 为本步新探索格子中被其他机器人已探索过的比例,
`Δcov_i` 为各机器人的覆盖率增量。

### 3.5 参数共享

**训练时技巧**: 所有 3 个智能体运行在同一 Python 进程中,
共享同一份 `DistributedActor.state_dict()`。
每次梯度更新同时更新所有智能体的策略,样本效率提升 3 倍。
无需显式角色标签,各机器人通过自身 teammate slice 的差异自然涌现角色分化。

**部署时**: 每台 Jetson Orin NX **独立从磁盘加载同一个 `mappo_actor.pth` 权重文件**
(离线文件拷贝),无任何状态同步协议,各机启动后权重完全一致。

---

## 4. 实验环境与配置

### 4.1 训练环境: 自定义 2D 栅格 Gymnasium

代码: `multi_robot_rl/grid_world.py`、`multi_robot_rl/multi_agent_env.py`

| 组件 | 规格 |
|---|---|
| 地图表示 | 2D `uint8` 栅格 FREE / OCCUPIED / UNKNOWN,0.1 m/格 |
| 感知模型 | 圆形 LIDAR,R = 4 m,360° FoV,被 OCCUPIED 遮挡 |
| 地图融合 | 模拟 `multirobot_map_merge`: 各机器人 FREE / OCCUPIED 观测取并集为共享栅格 |
| 碰撞检测 | 两两欧氏距离 ≤ 2 格 |
| 运动模型 | 每步最多移动 5 格朝向目标,碰障碍沿主轴滑行 |

Gazebo + ROS 2 仿真在 RTX 4090 上约 14 K 环境步/小时,
即 1 M 步需要约 70 小时; 2D 栅格环境快约 30 倍,训练在此进行。

### 4.2 三个场景(由 `scenario` 参数选择)

| 场景 | 布局 | 面积 | 房间数 | 障碍物数 |
|---|---|---|---|---|
| `multi_1` | 双房间公寓 | 35 m² | 2 | 4 个箱子 |
| `multi_2` | L 形含壁龛 | 50 m² | 1 + 1 | 6 个箱子 |
| `multi_3` | 多房间 + 走廊 | 80 m² | 4 | 8 个箱子 |

### 4.3 Gazebo + ROS 2 部署环境栈

代码: `launch/simulation.launch.py`、`launch/spawn_robots.launch.py`
模型: `multi_robot_rl/urdf/roslander.urdf.xacro`
世界文件: `multi_robot_rl/worlds/multi_room.world`
依赖: ROS 2 Humble + `ros_gz_sim`(Gazebo Sim / Fortress)

**启动顺序**:

```
gz_sim                          ── ros_gz_sim (加载 multi_room.world)
clock bridge                    ── /clock → ROS 时间
3× robot_state_publisher        ── 每机器人一个 namespace
3× spawn_entity                 ── ros_gz_sim,每 namespace 一个 URDF
3× ros_gz_bridge                ── cmd_vel, odom, scan, imu,
                                   joint_states ↔ Gazebo 主题
3× map_merge_node               ── off-the-shelf multirobot_map_merge,
                                   聚合 /robot_<i>/map → /merged_map
3× agent_node                   ── 本包,加载 Actor 权重,
                                   从 /merged_map 裁剪 64×64 窗口
```

> `shared_map_publisher.py` **不是一个独立节点**,而是作为 ROS 组件运行在
> `agent_node` **内部**: 每个 agent 订阅 `/merged_map` 并切片为自身的
> `local_map` 张量,策略不直接看到 `/merged_map`。

### 4.4 MAPPO 超参数 (`config/mappo_config.yaml`)

| 参数 | 值 | 说明 |
|---|---|---|
| `total_timesteps` | 1,000,000 | 总环境步数 |
| `n_steps` | 2,048 | 每次 Rollout 的步数 |
| `n_epochs` | 10 | 每次 Rollout 的 PPO epoch 数 |
| `batch_size` | 64 | SGD 小批量大小 |
| `gamma` | 0.99 | 折扣因子 |
| `gae_lambda` | 0.95 | GAE 参数 |
| `clip_range` | 0.2 | PPO ε 裁剪范围 |
| `actor_lr` | 3e-4 | Actor Adam 学习率 |
| `critic_lr` | 5e-4 | Critic Adam 学习率 |
| `entropy_coef` | 0.01 | 熵正则化系数 |
| `value_loss_coef` | 0.5 | Critic 损失权重 |
| `max_grad_norm` | 0.5 | 梯度裁剪范数 |
| `save_freq` | 50,000 | Checkpoint 保存间隔 |
| `log_freq` | 2,048 | 日志输出间隔 |
| `seed` | 0 | 随机种子 |

### 4.5 环境超参数 (`EnvConfig`)

| 参数 | 默认值 | 说明 |
|---|---|---|
| `scenario` | `multi_1` | 场景选择器 |
| `target_coverage` | 0.95 | 终止覆盖率阈值 |
| `max_steps_per_episode` | 300 | 每回合最大步数 |

### 4.6 硬件配置

| 用途 | 配置 |
|---|---|
| 训练服务器 | Ubuntu 20.04, RTX 4090 (24 GB), PyTorch 2.1.0 + CUDA 11.8 |
| 部署平台 | 3 × Jetson Orin NX (每机器人一台), CUDA 11.4, JetPack 6.0 |

### 4.7 日志与收敛

每 `log_freq` 步输出:

- `sps` — 环境步/秒
- `return` — 最近 10 回合的平均团队回报
- `coverage` — 最近 10 回合的平均团队覆盖率
- `actor_loss` / `critic_loss` / `entropy`

收敛标准: 连续 ≥ 100 个 Rollout 窗口 `coverage ≥ 0.95`。

---

## 5. 部署与仿真集成

### 5.1 安装与构建

```bash
cd ~/experiment_ws

# 安装 Python 依赖
cd src/multi_robot_rl
pip install -r requirements.txt
cd ~/experiment_ws

# 构建 ROS 2 包
colcon build --packages-select multi_robot_rl
source install/setup.bash
```

### 5.2 训练流程(2D 栅格环境)

**JSON 配置模式**(推荐):

```bash
# 新训练运行
python -m multi_robot_rl.train_entry --fresh --tag baseline

# 从 checkpoint 恢复
python -m multi_robot_rl.train_entry --resume runs/multi_1/20260806-baseline

# 按 tag 恢复
python -m multi_robot_rl.train_entry --resume baseline

# CLI 覆盖配置参数
python -m multi_robot_rl.train_entry --tag experiment --total_timesteps 500000 --n_agents 4

# 启动 TensorBoard 服务
python -m multi_robot_rl.train_entry --tag baseline --port 6006
```

配置文件位于 `config/` 目录:
- `train_config.json` — PPO 超参数、AMP、日志/保存设置
- `env_config.json` — 环境参数(场景、覆盖率、智能体数量、奖励权重)
- `model_config.json` — 网络结构(CNN 通道、隐藏层维度、正交初始化)

运行时 TensorBoard 日志写入 `runs/{scenario}/{timestamp}[-{tag}]/tensorboard/`。

**legacy CLI 模式**:

```bash
python -m multi_robot_rl.train_entry --total_timesteps 1000000 --scenario multi_1
```

多场景训练:

```bash
# 基准场景
python -m multi_robot_rl.train_entry --tag multi1 --scenario multi_1

# 中等难度
python -m multi_robot_rl.train_entry --tag multi2 --scenario multi_2

# 最难场景
python -m multi_robot_rl.train_entry --tag multi3 --scenario multi_3 --total_timesteps 1500000
```

`--seed` 参数可使用不同随机种子重复运行同一场景(报告数据默认使用 5 个种子)。

### 5.3 评估

```bash
# 评估训练好的模型
python scripts/eval_mappo.py --ckpt runs/multi_1/baseline --n_episodes 20

# 输出 JSON 结果
python scripts/eval_mappo.py --ckpt runs/multi_1/baseline --json eval_results.json
```

### 5.4 ONNX 导出

```bash
# 导出为 ONNX 格式(含数值校验)
python tools/export_onnx.py --ckpt runs/multi_1/baseline -o exported/

# 输出: exported/policy.onnx + exported/meta.json
```

### 5.5 Gazebo + ROS 2 完整仿真栈

```bash
ros2 launch multi_robot_rl simulation.launch.py \
    world:=$(ros2 pkg prefix multi_robot_rl)/share/multi_robot_rl/worlds/multi_room.world \
    n_robots:=3 headless:=true
```

每台机器人(或同一机器人的每个 namespace):

```bash
ROS_NAMESPACE=robot_1 ros2 run multi_robot_rl agent_node \
    --ros-args -p model_path:=/path/to/mappo_actor.pth -p robot_id:=1
```

`agent_node` 订阅:

- `/robot_<i>/map` — 自身 SLAM 发布的 `OccupancyGrid`
- `/robot_<i>/amcl_pose` — 自身位姿估计
- `/merged_map` — `map_merge_node` 输出,经 `shared_map_publisher` 切片为 64×64 窗口

发布目标点到 `/robot_<i>/move_base_simple/goal`,由机器人上的 `move_base` 动作服务器消费。

### 5.6 裸机部署(3 台 Jetson Orin NX)

无 Gazebo 时,`deployment.launch.py` 跳过仿真器,假设 `agent_node`
直接在各 Jetson 上运行,各指向自身的 `/robot_<i>/map` 和 `/amcl_pose`。
`/merged_map` 可由任何发出单一 `nav_msgs/OccupancyGrid` 的来源提供
(通常为运行在主机或机器人 1 上的 `map_merge_node`):

```bash
ros2 launch multi_robot_rl deployment.launch.py \
    model_path:=/path/to/mappo_actor.pth n_agents:=3
```

### 5.7 仓库结构

```
multi-robot-rl/
├── package.xml                     # ROS 2 ament_python
├── setup.py
├── setup.cfg
├── README.md
├── requirements.txt                # Python 依赖
├── .gitignore
├── config/
│   ├── train_config.json           # PPO 超参数 + AMP + 日志设置
│   ├── env_config.json             # 环境参数 + 奖励权重
│   └── model_config.json           # 网络结构 + 正交初始化参数
├── launch/
│   ├── training.launch.py          # 2D 栅格环境 + MAPPO 训练器
│   └── deployment.launch.py        # 3 × agent_node
├── scripts/
│   └── eval_mappo.py               # 离线评估脚本
├── tools/
│   └── export_onnx.py              # ONNX 导出工具(onxxruntime 校验)
├── runs/                           # 训练运行目录(时间戳 + tag)
└── multi_robot_rl/
    ├── __init__.py
    ├── grid_world.py               # 3 场景,感知,运动,指标
    ├── frontier_detector.py        # 8 邻域卷积 + Farthest-point 聚类
    ├── reward_functions.py         # 7 项团队奖励 + Jain 公平性
    ├── observation.py              # 共享观测构建(训练 + 部署共用)
    ├── multi_agent_env.py          # Gymnasium 多智能体环境接口
    ├── networks.py                 # DistributedActor + CentralizedCritic
    ├── replay_buffer.py            # On-policy Rollout 缓冲 + GAE
    ├── mappo_trainer.py            # 训练循环 + AMP + 值归一化
    ├── config_loader.py            # JSON 配置加载 + CLI 覆盖
    ├── metric_writer.py            # TensorBoard + JSONL 双通道日志
    ├── run_manager.py              # 运行目录管理 + 统一 checkpoint
    ├── train_entry.py              # 训练 CLI 入口(--fresh/--resume/--tag)
    └── agent_node.py               # ROS 2 Actor 节点
```

---

## 6. 参考

1. **Ruan, X., et al.** End-to-End Deep Reinforcement Learning for Autonomous Exploration. 2021.
2. **Yu, C., et al.** The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games. *NeurIPS*, 2021.
3. **Yamauchi, B.** Frontier-based exploration using multiple robots. *Agents*, 1998.
4. **Rashid, T., et al.** Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning. *JMLR*, 2020.
5. ROS Wiki. **multirobot_map_merge**. http://wiki.ros.org/multirobot_map_merge
6. anurye/gym-turtlebot — ROS 2 + Gazebo + Gymnasium 参考布局. https://github.com/anurye/gym-turtlebot
