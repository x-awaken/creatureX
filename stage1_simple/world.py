import os
import sys
import time

import gymnasium as gym
import cv2

sys.path.append('./')
import stage1_simple.Xenv as xenv

class BaseWorld(object):
    '''
    dispay_mode:
        linux-remote-rgb , redirecting x-client display requests and using open cv to display continual images
        linux-remote, redirecting x-client display requests and using default human display mode
        others, using local display engine
    render_mode:
        human, displaying animations of the env
        rgb_array, do not display animations,but return rgb_array when calling render()  
    '''
    def __init__(self, agent=None, dispay_mode='linux-remote-rgb', render_mode='human', DISPLAY_NO=13) -> None:
        self.agent = agent
        self.cur_action = None
        self.dispay_mode = dispay_mode
        # 创建环境
        self.render_mode = render_mode
        if self.dispay_mode.startswith('linux-remote'):
            os.environ['DISPLAY'] = ':%s'%DISPLAY_NO
        if self.render_mode =='human' and self.dispay_mode == 'linux-remote-rgb':
            self.real_render_mode = 'rgb_array'
        else:
            self.real_render_mode = self.render_mode
        self.init_env()

    def init_env(self):
        raise NotImplementedError
    
    def run(self, max_frame=10000, fps=20):
        #重置环境
        observation, info = self.env.reset(seed=42)
        self.cur_action = self.agent_act(observation,None,info,None)

        # 设置目标帧率
        if self.render_mode == 'human'and self.dispay_mode=='linux-remote-rgb':
            self.target_fps = fps
            frame_time = 1.0 / self.target_fps
        period_time_start = time.time()
        last_frame = 0

        #开始循环迭代
        for frame_cnt in range(1,max_frame+1):
            if self.render_mode == 'human' and self.dispay_mode == 'linux-remote-rgb':
                start_time = time.time()
                rgb_array = self.env.render()
                resized_array = rgb_array 
                resized_array = cv2.resize(rgb_array, (210, 160))  # 调整图像大小
                cv2.imshow('env', cv2.cvtColor(resized_array, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

            observation, reward, terminated, truncated, info = self.env.step(self.cur_action)
            self.cur_action = self.agent_act(observation,reward,info,self.cur_action)
            if terminated or truncated:
                observation, info = self.env.reset()
            if frame_cnt % 10000 == 0:
                print('cur fps:%s'%((frame_cnt-last_frame)/(time.time()-period_time_start)))
                last_frame = frame_cnt
                period_time_start = time.time()
                print(info)
                print('observation:',observation)
            if self.render_mode == 'human' and self.dispay_mode=='linux-remote-rgb':
                # 计算帧时间并延迟以达到目标帧率
                frame_time_elapsed = time.time() - start_time
                if frame_time_elapsed < frame_time:
                    time.sleep(frame_time - frame_time_elapsed)
        cv2.destroyAllWindows()
    
    def agent_act(self, observation, reward, info, cur_action):
        #return self.env.action_space.sample()+[1,0]
        return self.env.action_space.sample()

    def stop(self):
        self.env.close()

class AtariWorld(BaseWorld):
    def __init__(self, agent=None,dispay_mode='linux-remote-rgb',render_mode='human',DISPLAY_NO=13) -> None:
        super().__init__(agent,dispay_mode,render_mode=render_mode, DISPLAY_NO=DISPLAY_NO)
    
    def init_env(self):
        self.env = gym.make("BattleZone",render_mode=self.real_render_mode,obs_type='grayscale')


class CellWorld(BaseWorld):
    def __init__(self, agent=None, dispay_mode='linux-remote-rgb',render_mode='human',DISPLAY_NO=13) -> None:
        super().__init__(agent,dispay_mode,render_mode=render_mode, DISPLAY_NO=DISPLAY_NO)
    
    def init_env(self):
        self.env = gym.make("stage1/CellWorldEnv-v0",render_mode=self.real_render_mode)


if __name__ == '__main__':
    # world = AtariWorld(dispay_mode='linux-remote-rgb',render_mode='human', DISPLAY_NO=10)
    world = CellWorld(dispay_mode='linux-remote',render_mode='human', DISPLAY_NO=10)
    world.run()




