"""Standalone training entry point (convenience wrapper).

Delegates to ``multi_robot_rl.train_entry:main`` so that both
``python scripts/train_mappo.py`` and ``ros2 run multi_robot_rl train_mappo``
share the same implementation.
"""

import sys
from multi_robot_rl.train_entry import main

if __name__ == '__main__':
    sys.exit(main())
