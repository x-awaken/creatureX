import os
import sys
import time

import gymnasium as gym
import cv2

sys.path.append('./')
import stage1_simple.Xenv as xenv

class BaseWorld(object):
    def __init__(self, agent=None, os_type='win') -> None:
        self.agent = agent
        self.cur_action = None
        self.os_type = os_type
        # 创建环境
        if self.os_type == 'linux-remote':
            os.environ['DISPLAY'] = ':20'
            self.render_mode = 'rgb_array'
        else:
            self.render_mode = 'human'
        self.init_env()

    def init_env(self):
        raise NotImplementedError
    
    def run(self, max_frame=10000, render_mode='human', fps=20):
        #重置环境
        observation, info = self.env.reset(seed=42)
        self.cur_action = self.agent_act(observation,None,info,None)

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
                if self.os_type == 'linux-remote':
                    rgb_array = self.env.render()
                    resized_array = rgb_array 
                    #resized_array = cv2.resize(rgb_array, (210, 160))  # 调整图像大小
                    cv2.imshow('env', cv2.cvtColor(resized_array, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
                if frame_cnt % 100 == 0:
                    print('cur fps:%s'%((frame_cnt-last_frame)/(time.time()-period_time_start)))
                    last_frame = frame_cnt
                    period_time_start = time.time()
                    print(info)
                    print('observation:',observation)
            observation, reward, terminated, truncated, info = self.env.step(self.cur_action)
            self.cur_action = self.agent_act(observation,reward,info,self.cur_action)
            if terminated or truncated:
                observation, info = self.env.reset()
            if render_mode == 'human':
                # 计算帧时间并延迟以达到目标帧率
                frame_time_elapsed = time.time() - start_time
                if frame_time_elapsed < frame_time:
                    time.sleep(frame_time - frame_time_elapsed)
        cv2.destroyAllWindows()
    
    def agent_act(self, observation, reward, info, cur_action):
        return self.env.action_space.sample()+[1,0]

    def stop(self):
        self.env.close()

class AtariWorld(BaseWorld):
    def __init__(self, agent=None, os_type='win') -> None:
        super().__init__(agent,os_type)
    
    def init_env(self):
        self.env = gym.make("BattleZone",render_mode=self.render_mode,obs_type='grayscale')


class CellWorld(BaseWorld):
    def __init__(self, agent=None, os_type='win') -> None:
        super().__init__(agent,os_type)
    
    def init_env(self):
        self.env = gym.make("stage1/CellWorldEnv-v0",render_mode=self.render_mode)


if __name__ == '__main__':
    # world = AtariWorld(os_type='linux-remote')
    world = CellWorld(os_type='linux-remote')
    world.run(render_mode='human')




