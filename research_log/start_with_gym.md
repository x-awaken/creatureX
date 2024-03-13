## 一、在Linux下采用gym搭建Agent的仿真环境

### 1.python及其依赖环境安装

分别执行以下命令：
``` shell
pip install gymnasium
pip install gymnasium[atari]
pip install gymnasium[accept-rom-license]

# 或者直接安装所有的依赖
pip install "gymnasium[all]
```

### 2.撰写示例代码

此时可参考官网写一个示例代码测试安装情况
``` python
import gymnasium as gym
env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
```
如果是直接在有界面的服务器上本地运行那么可以看到环境运行的画面。但我这是采用vscode连接的服务器，无法在我的本地客户端显示，若需要进行可视化则需要继续执行下面的步骤

### 3. Windows 10系统上安装X-SERVER软件：[vcxsrv-64.1.20.14.0.installer](https://sourceforge.net/projects/vcxsrv/files/)

安装X-SERVER的原因稍微解释一下，在Linux系统中，通常采用X11作为显示的软件，它分为x-client和x-server两部分，x-client通过tcp通信的方式与x-server进行连接：

__X客户端（X client）：__

- X客户端是运行在计算机系统上的图形应用程序，它们向X服务器发送请求以获取用户输入并在屏幕上显示图形。
- X客户端可以是任何图形应用程序，例如文本编辑器、终端模拟器、游戏等等。
- 当X客户端运行时，它会连接到X服务器并发送请求，告诉X服务器如何绘制图形元素以及如何响应用户输入。

__X服务器（X server）：__

- X服务器是负责管理显示硬件的软件组件，它负责接收来自X客户端的请求，并将它们转换为实际的图形显示。
- X服务器控制显示设备（例如显示器、键盘和鼠标），并负责将图形元素绘制到屏幕上。
- X服务器还负责处理用户的输入事件（如键盘按键、鼠标点击等），并将它们发送给X客户端。

为了能够显示远程服务器输出的图形界面，需要在windows10系统上安装一个x-server程序，并将来自远程主机的x-client请求转发到这个x-server程序。

下载安装包后直接双击安装包运行即可完成安装

在启动时需要注意

- 需要指定显示编号（Display number）
这个显示编号是客户端与服务端通信端口的一个组成部分（默认是6000+显示编号），例如下图中将显示编号设置为了20，那么这个x-server启动之后将会绑定到6020端口上


<div align=center> <img src=imgs/x-server-1.png style="zoom:70%;" /> </div>


- 需要放开访问控制，以允许其他client的访问

<div align=center> <img src=imgs/x-server-2.png style="zoom:70%;" /> </div>

### 4. 设置端口映射（如果服务器可以直接连接本地机器，跳过此步）
我的情况是本地可访问服务器，服务器不可访问本地（可能是NAT的原因），因此需要设置一个反向隧道
在x-shell中的设置方式如下图：


<div align=center> <img src=imgs/x-shell隧道.png style="zoom:60%;" /> </div>

### 5. 重定向x-client的请求地址
在Linux中可以用以下命令查看
``` shell
echo $DISPLAY
# localhost:10.0
```
这一环境变量的格式为HOST:D.S，D表示Display编号，默认情况下表示X Server运行在6000+D的TCP端口上
因此要改变x-client的请求地址核心就是改变环境变量 DISPLAY

在shell中可以通过
``` shell
export DISPLAY=:20 
```
在python程序中可以通过
``` python
import os
os.environ['DISPLAY'] = ':20'
```

### 6. 更改后的示例程序
``` python
import os
import time
os.environ['DISPLAY'] = ':20'
import gymnasium as gym
import matplotlib.pyplot as plt
env = gym.make("BattleZone",render_mode="human")

observation, info = env.reset(seed=42)
for _ in range(1000):
    rgb_array = env.render()
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if _ %10 == 0:
        print(_)
    if terminated or truncated:
        observation, info = env.reset()
env.close()
```
这种方式是gym原生的可视化方法，但部分环境显示时窗口分辨率太大，会导致显示延迟严重，因此采用以下方式进行优化

### 7. 缩小分辨率的示例程序
gym本身并不支持直接对显示分辨率进行控制，因此采用了rgb_array的render_mode缩小分辨率
```python
import os
import cv2
import gymnasium as gym

os.environ['DISPLAY'] = ':20'
env = gym.make("BattleZone", render_mode="rgb_array", render_fps=30, render_resolution=(320, 240))

observation, info = env.reset(seed=42)
for _ in range(1000):
    rgb_array = env.render()
    cv2.imshow('BattleZone', cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if _ % 10 == 0:
        print(_)
    if terminated or truncated:
        observation, info = env.reset()
env.close()
cv2.destroyAllWindows()
```
当然除了这种方法，还可采用matplotlib进行可视化：
``` python
import os
import time
os.environ['DISPLAY'] = ':20'
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('TkAgg') # 或者尝试其他后端，如 'Qt5Agg', 'WXAgg', 'MacOSX' 等

class AtariWorld(object):
    def init(self, agent=None) -> None:
        self.agent = agent
        self.cur_action = 0
        # 创建环境
        self.env = gym.make("BattleZone",render_mode="rgb_array")
        # 重置环境
        self.observation = self.env.reset()
        # 创建一个空图像，并指定figsize
        self.fig, self.ax = plt.subplots(figsize=(8, 6))  # 设置宽度为8英寸，高度为6英寸
        self.img = self.ax.imshow(self.env.render())
        plt.axis('off')

    def run(self):
        self.tick = 0
        # 定义更新函数
        def update(frame):
            observation, reward, terminated, truncated, info = self.env.step(self.cur_action)
            if terminated or truncated:
                observation, info = self.env.reset()
            self.cur_action = self.agent_act(observation, reward, info, self.cur_action)
            self.img.set_array(self.env.render())
            if self.tick % 100 == 0:
                print(self.tick)
            self.tick += 1
            return self.img,
        # 创建动画
        ani = FuncAnimation(self.fig, update, frames=range(50), interval=1, blit=True)
        plt.show()

    def agent_act(self, observation, reward, info, cur_action):
        return self.env.action_space.sample()

    def stop(self):
        self.env.close()

if name == 'main':
    atari = AtariWorld()
    atari.run()
```

参考文章：[如何使用vscode远程debug linux图形界面程序](https://www.cnblogs.com/mingcc/p/17283045.html)

