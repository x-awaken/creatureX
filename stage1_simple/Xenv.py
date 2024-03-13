import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces


class CellWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, food_size=5, agent_size=10, fov_size=50):
        self.food_size = food_size  # The size of the square grid
        self.agent_size = agent_size
        self.fov_size = fov_size
        self.window_size = 512  # The size of the PyGame window
        self.food_source_cnt = 2
        self.food_source_size = 100
        self.target_size = 1
        self.starve_rate = 0.01

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "food": spaces.Box(0, self.window_size**2, shape=(8,), dtype=int),
                "agent": spaces.Box(0, self.window_size, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Box(-1,1,shape=(2,2), dtype=float)
        self.step_size = 1
        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
    
    def _get_obs(self):
        return {"food": self._food_obs, "agent": self._agent_location}
    
    def _get_info(self):
        return {
            "eat_food": self._eat_food,
            "cur_food": self._cur_food
        }
    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.window_size, size=2, dtype=int).astype(float)
        self._food_obs = self.observation_space.sample()['food']*0
        self._eat_food = 0
        self._cur_food = 5.0

        # We will sample the target's location randomly until it does not coincide with the agent's location
        fsource_locations = []
        for i in range(self.food_source_cnt):
            source_location = self._agent_location
            while np.array_equal(source_location, self._agent_location):
                source_location = self.np_random.integers(
                    0, self.window_size, size=2, dtype=int
                )
            fsource_locations.append(source_location)

        self._food_locations = np.random.normal(loc=0, 
                                                 scale=self.window_size*0.12, 
                                                 size= (self.food_source_cnt*self.food_source_size, 2))

        for i, fsouce_location in enumerate(fsource_locations): 
            self._food_locations[i*self.food_source_size:(i+1)*self.food_source_size] += fsouce_location                 

        condition = (self._food_locations[:, 0] > 0) & (self._food_locations[:, 0] < self.window_size) & (self._food_locations[:, 1] > 0) & (self._food_locations[:, 1] < self.window_size)

        self._food_locations = self._food_locations[condition]
        self.visible_food = np.zeros(self._food_locations.shape[0],dtype=bool)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        
        return observation, info
    
    def calculate_angles(self, source_location, target_locations):
        """
        计算源点到目标点连线与 x 轴的夹角（弧度）。

        参数：
        source_location：圆心坐标，形状为 (2,) 的 NumPy 数组。
        target_locations：目标点坐标，形状为 (n, 2) 的 NumPy 数组，其中 n 是目标点的数量。

        返回值：
        angles：源点到每个目标点连线与 x 轴的夹角，形状为 (n,) 的 NumPy 数组。
        """
        # 转换为np数组
        source_location = np.array(source_location)
        target_locations = np.array(target_locations)
        # 计算目标点相对于圆心的位置向量
        vectors = target_locations - source_location

        # 计算每个向量与 x 轴的夹角
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])

        return angles

    
    def step(self, action):
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location += action.sum(axis=0)*self.step_size
        self._agent_location = np.clip(self._agent_location,a_min=0,a_max=self.window_size)
        # compute eaten food
        distances = np.linalg.norm(self._food_locations-self._agent_location, axis=1)
        eaten = distances <= self.agent_size/2
        reward = eaten.sum()
        self._food_locations = self._food_locations[~eaten]
        self._cur_food += reward-self.starve_rate
        self._eat_food += reward

        # compute visible food
        distances = np.linalg.norm(self._food_locations-self._agent_location, axis=1)
        self.visible_food = distances <= self.fov_size/2
        # An episode is done iff the agent has reached the target
        terminated = self._cur_food<=0
        
        angles = self.calculate_angles(self._agent_location, self._food_locations[self.visible_food])
        obs_size = self.observation_space['food'].shape[0]
        angle_block = 2*np.pi/obs_size
        cur_obs = self.observation_space.sample()
        for k in cur_obs:
            cur_obs[k] = cur_obs[k]*0
        for i in range(obs_size):
            matched_food =  (angles>=(angle_block*i-np.pi))&(angles < (angle_block*(i+1)-np.pi))
            cur_obs['food'][i] = matched_food.sum()
        self._food_obs = cur_obs['food']
        

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((221, 221, 221))
        # pix_square_size = (
        #     self.window_size / self.size
        # )  # The size of a single grid square in pixels

        # First we draw the target
        # pygame.draw.rect(
        #     canvas,
        #     (255, 0, 0),
        #     pygame.Rect(
        #         pix_square_size * self._food_locations,
        #         (pix_square_size, pix_square_size),
        #     ),
        # )
        for target_location, is_visible in zip(self._food_locations, self.visible_food):
            if is_visible:
                color = (30, 230, 116)
            else:
                color = (82, 117, 244)
            pygame.draw.circle(
                canvas,
                color,
                target_location,
                self.food_size / 2,
            )

        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (45, 175, 211),
            self._agent_location,
            self.agent_size / 2,
        )
        pygame.draw.circle(
            canvas,
            (45, 175, 211),
            self._agent_location,
            self.fov_size / 2,
            width=1
        )

        # # Finally, add some gridlines
        # for x in range(self.size + 1):
        #     pygame.draw.line(
        #         canvas,
        #         0,
        #         (0, pix_square_size * x),
        #         (self.window_size, pix_square_size * x),
        #         width=3,
        #     )
        #     pygame.draw.line(
        #         canvas,
        #         0,
        #         (pix_square_size * x, 0),
        #         (pix_square_size * x, self.window_size),
        #         width=3,
        #     )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

from gymnasium.envs.registration import register
import sys
register(
     id="stage1/CellWorldEnv-v0",
     entry_point="stage1_simple.Xenv:CellWorldEnv",
     max_episode_steps=300000,
)