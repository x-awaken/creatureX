import torch

class Neuron(object):
    def __init__(self) -> None:

        self.gamma = 0.1
        self.input_residual = torch.zeros([1,2])
        self.internal_signals = torch.zeros([1,8])
        #轴突
        self.axon = 0

    #如何模拟能量守恒（化学物质、电量等），例如被连续激活之后输出就会减弱（久入厕而不觉臭(●'◡'●)）
    def step(self, x):
        
        self.residual_input = self.gamma*self.residual_input + x
        self.axon = torch.sigmoid(x)

import random

class Neuron:
    def __init__(self, resting_potential=-70, threshold_potential=-55, refractory_period=5):
        self.resting_potential = resting_potential  # 静息电位
        self.threshold_potential = threshold_potential  # 阈值电位
        self.refractory_period = refractory_period  # 不应期时长
        self.membrane_potential = self.resting_potential  # 膜电位
        self.refractory_time = 0  # 当前不应期时间
        self.dendrites = []  # 树突列表
        self.axon = None  # 轴突
        
    def add_dendrite(self, dendrite):
        self.dendrites.append(dendrite)  # 添加树突
        
    def remove_dendrite(self, dendrite):
        self.dendrites.remove(dendrite)  # 移除树突
        
    def set_axon(self, axon):
        self.axon = axon  # 设置轴突
        
    def receive_signal(self, signal, neurotransmitter_type):
        if self.refractory_time > 0:
            return  # 如果在不应期内,不接收新的信号
        
        if neurotransmitter_type == "excitatory":
            self.membrane_potential += signal  # 接收兴奋性信号,增加膜电位
        elif neurotransmitter_type == "inhibitory":
            self.membrane_potential -= signal  # 接收抑制性信号,减少膜电位
        
        if self.membrane_potential >= self.threshold_potential:
            self.fire_action_potential()  # 如果膜电位达到阈值,产生动作电位
        
    def fire_action_potential(self):
        self.membrane_potential = 30  # 动作电位峰值
        self.refractory_time = self.refractory_period  # 进入不应期
        if self.axon:
            self.axon.transmit_signal(30)  # 如果有轴突,将动作电位传递给下一个神经元
        
    def update(self):
        if self.refractory_time > 0:
            self.refractory_time -= 1  # 更新不应期时间
            if self.refractory_time == 0:
                self.membrane_potential = self.resting_potential  # 不应期结束,恢复静息电位
        else:
            self.membrane_potential = self.resting_potential + sum([dendrite.get_signal() for dendrite in self.dendrites])  # 更新膜电位,根据所有树突的信号求和
            
class Dendrite:
    def __init__(self, neurotransmitter_type, strength):
        self.neurotransmitter_type = neurotransmitter_type  # 神经递质类型
        self.strength = strength  # 树突强度
        
    def get_signal(self):
        return random.randint(0, self.strength)  # 根据树突强度产生随机信号
    
class Axon:
    def __init__(self, terminal):
        self.terminal = terminal  # 连接的神经元终端
        
    def transmit_signal(self, signal):
        self.terminal.receive_signal(signal, "excitatory")  # 将动作电位传递给连接的神经元终端
        
class NeuronTerminal:
    def __init__(self, neuron):
        self.neuron = neuron  # 所属的神经元
        
    def receive_signal(self, signal, neurotransmitter_type):
        self.neuron.receive_signal(signal, neurotransmitter_type)  # 接收信号并传递给所属的神经元

class CellAgent(object):
    def __init__(self) -> None:
        pass