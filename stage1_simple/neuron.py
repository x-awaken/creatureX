import torch
import random
from collections import deque

class TransmitterItem(object):
    def __init__(self,ts_value, ts_clock, upgrade_time=0, decay_time=0) -> None:
        self.ts_value = ts_value
        self.ts_clock = ts_clock
        self.upgrade_time = upgrade_time
        self.decay_time = decay_time

## 突触接收到神经递质后就会产生电位差，产生电位差的大小与突触的类型、强度、长度等有关
# 突触具有恢复期，恢复期内接收相同的信号所产生的电位差较小
# 从接收神经递质到产生输出到神经元细胞体需要一定的传导时间
class Synapse:
    def __init__(self, neurotransmitter_type="excitatory", 
                 strength=1.0, 
                 transmit_time=10, 
                 clock_interval=5,
                 upgrade_time = 5,
                 decay_time = 5,
                 decay_mode = 'linear',
                 activate_interval=20.0,
                 abs_max_dp = 15
                 ):
        self.neurotransmitter_type = neurotransmitter_type  # 神经递质类型
        self.strength = strength  # 突触强度
        self.transmit_time = transmit_time  #  产生电位差传导到细胞体的时间，单位ms,通常是1~10ms
        self.clock_interval = clock_interval # 时钟步长
        self.default_upgrade_time = upgrade_time #控制接收到神经递质后达到电位差峰值的时间
        self.default_decay_time = decay_time #控制接收到神经递质后从电位差峰值衰减到0的时间
        self.decay_mode = decay_mode # 衰减的模式
        self.abs_max_dp = abs_max_dp # 突触可达到的最大电位差
        self.accumulate_weight = 3
        self.clock = 0
        self.activate_clock = 0
        self.activate_interval = activate_interval # 两次激活的时间间隔
        self.cur_transmitter = [] # 当前的接收到的神经递质
        self.activate_history = deque(maxlen=50)

    # 接收神经递质，产生电位差
    def recieve_transmitter(self,transmitter_cnt,  refractory_time, upgrade_time=0, decay_time=0):
        if transmitter_cnt == 0:
            return
        # 根据突触的恢复期时长计算强度衰减
        strength = self.strength - self.strength*self.activate_clock/self.activate_interval
        # 不应期期间，影响兴奋增强信号的产生
        if self.neurotransmitter_type == "excitatory" and refractory_time==0:
            decayed_transmitter_cnt =  transmitter_cnt*strength  # 兴奋性突触根据强度放大信号
        elif self.neurotransmitter_type == "inhibitory":
            decayed_transmitter_cnt =  -transmitter_cnt*strength  # 抑制性突触根据强度减弱信号
        else:
            return
         # potential difference
        ts = TransmitterItem(decayed_transmitter_cnt, self.clock+self.transmit_time, upgrade_time, decay_time)
        self.cur_transmitter.append(ts)
        self.activate_history.appendleft(ts)

    def generate_dp(self,):
        # 超出神经递质的作用时间，移除
        self.cur_transmitter = [e for e in self.cur_transmitter if self.clock <= e.ts_clock+e.upgrade_time+e.decay_time]

        total_ts_value = 0
        for transmitter in self.cur_transmitter:
            ts_value = transmitter.ts_value
            #衰减期
            if self.clock >= transmitter.ts_clock+transmitter.upgrade_time:
                ts_value = ts_value*(transmitter.ts_clock+transmitter.upgrade_time+transmitter.decay_time-self.clock)/transmitter.decay_time
            #上升期
            elif self.clock > transmitter.ts_clock:
                ts_value = ts_value*float(self.clock-transmitter.ts_clock)/transmitter.upgrade_time
            else:
                continue
            total_ts_value += ts_value

        if 0==total_ts_value:
            return 0
        dp_value = self.activation_fun(total_ts_value)
        return dp_value

    def activation_fun(self, ts_value):
        ts_value = torch.Tensor([ts_value])
        dp_value = self.abs_max_dp*(torch.sigmoid(ts_value*self.accumulate_weight+2)-0.88)
        dp_value = (torch.sigmoid(ts_value*self.accumulate_weight+3)-0.95)/0.05
        return dp_value

    def step(self, transmitter_value=0, upgrade_time=10, decay_time=10,refractory_time=0):
        self.clock += self.clock_interval
        if upgrade_time==0:
            upgrade_time = self.default_upgrade_time
        if decay_time == 0:
            decay_time = self.default_decay_time
        self.recieve_transmitter(transmitter_value, refractory_time,upgrade_time, decay_time)
        dp_value = self.generate_dp()
        self.dynamic_update()
        return dp_value
    
    # 根据激活历史，更新突触效率
    def dynamic_update(self):
        pass

class Dendrite:#树突
    def __init__(self, synapses):
        self.synapses = synapses # 突触列表
        self.coordinate = (0,0,0) # 树突的根部中心坐标，用于仿真多个树突产生的电位差
        self.radus = 0 # 树突可以影响的细胞膜半径，用于开展EPSP或IPSP在空间上的叠加
        self.cur_dp = 0 
    
    def size(self):
        return(len(self.synapses))
        
    def add_synapse(self, synapse):
        self.synapses.append(synapse)  # 添加突触
        
    def remove_synapse(self, synapse):
        self.synapses.remove(synapse)  # 移除突触
    
    def step(self, transmitter_values=[], upgrade_times=[], decay_times=[], refractory_time=0):
        assert(len(transmitter_values)==len(self.synapses))
        total_dp = 0
        for transmitter_value,upgrade_time,decay_time,synapse in zip(transmitter_values,upgrade_times,decay_times,self.synapses):
            dp = synapse.step(transmitter_value,upgrade_time,decay_time,refractory_time)
            total_dp += dp
        self.update()
        return total_dp
    
    def update(self):
        pass
    

class Neuron:
    def __init__(self, resting_potential=-70, threshold_potential=-55, refractory_period=5):
        self.resting_potential = resting_potential  # 静息电位
        self.threshold_potential = threshold_potential  # 阈值电位
        self.refractory_period = refractory_period  # 不应期时长
        self.membrane_potential = self.resting_potential  # 膜电位
        self.refractory_time = 0  # 当前不应期时间
        self.dendrites = []  # 树突列表
        self.axon = None  # 轴突
        self.clock = 0  # 神经元时钟
        
    def add_dendrite(self, dendrite):
        self.dendrites.append(dendrite)  # 添加树突
        
    def remove_dendrite(self, dendrite):
        self.dendrites.remove(dendrite)  # 移除树突
        
    def set_axon(self, axon):
        self.axon = axon  # 设置轴突
    
    def step(self, transmitter_values=[], upgrade_times=[], decay_times=[]):
        k = 0
        for i in range(len(self.dendrites)):
            dendrite_size = len(self.dendrites[i])
            self.dendrites[i].step(transmitter_values[k:k+dendrite_size], 
                                   upgrade_times[k:k+dendrite_size], 
                                   decay_times[k:k+dendrite_size])
            k+=dendrite_size
        # todo, 需要实现EPSP以及IPSP在时间与空间上的累加！！
        
    def receive_signals(self):
        if self.refractory_time > 0:
            return  # 如果在不应期内,不接收新的信号
        
        total_signal = sum([dendrite.receive_signals() for dendrite in self.dendrites])  # 接收所有树突的信号并求和
        self.membrane_potential += total_signal  # 更新膜电位
        
        if self.membrane_potential >= self.threshold_potential:
            self.fire_action_potential()  # 如果膜电位达到阈值,产生动作电位
        
    def fire_action_potential(self):
        self.membrane_potential = 30  # 动作电位峰值
        self.refractory_time = self.refractory_period  # 进入不应期
        if self.axon:
            self.axon.transmit_signal(30)  # 如果有轴突,将动作电位传递给下一个神经元
        
    def update(self):
        self.clock += self.clock_interval  # 更新神经元时钟
        if self.refractory_time > 0:
            self.refractory_time -= self.clock_interval   # 更新不应期时间
            if self.refractory_time == 0:
                self.membrane_potential = self.resting_potential # 不应期结束,恢复静息电位
        else:
            self.receive_signals() # 接收信号并更新膜电位

class Axon:
    def init(self, terminal):
        self.terminal = terminal # 连接的神经元终端
    
    def transmit_signal(self, signal):
        self.terminal.receive_signal(signal)  # 将动作电位传递给连接的神经元终端

class NeuronTerminal:
    def init(self, neuron):
        self.neuron = neuron # 所属的神经元
    def receive_signal(self, signal):
        self.neuron.receive_signals()  # 接收信号并传递给所属的神经元进行处理


if __name__ == "__main__":
    s = Synapse(clock_interval=5)
    s.accumulate_weight = 3
    for i in range(10):
        i = i*0.1
        print(i,'clock:',s.clock, ' output:',s.step(i,upgrade_time=15, decay_time=15))
        print(i,'clock:',s.clock, ' output:',s.step(i,upgrade_time=15, decay_time=15))
        print(i,'clock:',s.clock, ' output:',s.step())
        print(i,'clock:',s.clock, ' output:',s.step())
        print(i,'clock:',s.clock, ' output:',s.step())
