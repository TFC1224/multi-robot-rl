from setuptools import find_packages, setup

package_name = 'multi_robot_rl'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/training.launch.py',
                                                 'launch/deployment.launch.py']),
        ('share/' + package_name + '/config', ['config/train_config.json',
                                                  'config/env_config.json',
                                                  'config/model_config.json']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='DLUT ROS Course',
    maintainer_email='ros@dlut.edu.cn',
    description='Pure-CTDE multi-robot collaborative exploration (MAPPO)',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'train_mappo = multi_robot_rl.train_entry:main',
            'agent_node = multi_robot_rl.agent_node:main',
        ],
    },
)
