import os
import time
os.environ['DISPLAY'] = ':20'
import gymnasium as gym
import cv2

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class AtariWorld(object):
    def __init__(self, agent=None) -> None:
        self.agent = agent
        self.cur_action = 0
        # 创建环境
        self.env = gym.make("BattleZone",render_mode="rgb_array",obs_type='grayscale')

    def run(self, max_frame=1000, render_mode='human', fps=20):
        #重置环境
        observation, info = self.env.reset(seed=42)

        # 设置目标帧率
        if render_mode == 'human':
            target_fps = fps
            frame_time = 1.0 / target_fps
            last_frame = 0
            period_time_start = time.time()

        #开始循环迭代
        for frame_cnt in range(1,max_frame+1):
            if render_mode == 'human':
                start_time = time.time()
                rgb_array = self.env.render()
                resized_array = rgb_array 
                #resized_array = cv2.resize(rgb_array, (210, 160))  # 调整图像大小
                cv2.imshow('BattleZone', cv2.cvtColor(resized_array, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
                            # 计算帧时间并延迟以达到目标帧率
                if frame_cnt % 100 == 0:
                    print('cur fps:%s'%((frame_cnt-last_frame)/(time.time()-period_time_start)))
                    last_frame = frame_cnt
                    period_time_start = time.time()
                    print(info)
            observation, reward, terminated, truncated, info = self.env.step(self.cur_action)
            self.cur_action = self.agent_act(observation,reward,info,self.cur_action)
            if terminated or truncated:
                observation, info = self.env.reset()
            if render_mode == 'human':
                frame_time_elapsed = time.time() - start_time
                if frame_time_elapsed < frame_time:
                    time.sleep(frame_time - frame_time_elapsed)
        cv2.destroyAllWindows()
    
    def agent_act(self, observation, reward, info, cur_action):
        return self.env.action_space.sample()

    def stop(self):
        self.env.close()

if __name__ == '__main__':
    atari = AtariWorld()
    atari.run(render_mode='human')



