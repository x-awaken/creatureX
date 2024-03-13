import gymnasium as gym
from gymnasium import spaces
import numpy as np

# 定义空间
space = gym.spaces.Dict({
    "food": gym.spaces.Box(0, 3**2, shape=(8,), dtype=int),
    "agent": gym.spaces.Box(0, 5, shape=(2,), dtype=int)
})

# 生成全为零的动作
zero_action = space.sample()
print(zero_action)
for k in zero_action:
    zero_action[k] = zero_action[k]*0

# 输出全为零的动作
print("全为零的动作：", zero_action)
