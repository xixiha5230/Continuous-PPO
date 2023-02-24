# PPO-PyTorch

## 参考

- [conv1d、conv2d网络]([GitHub - BlueFisher/Advanced-Soft-Actor-Critic: Soft Actor-Critic with advanced features](https://github.com/BlueFisher/Advanced-Soft-Actor-Critic))

- [lstm、多线程]([GitHub - MarcoMeter/recurrent-ppo-truncated-bptt: Baseline implementation of recurrent PPO using truncated BPTT](https://github.com/MarcoMeter/recurrent-ppo-truncated-bptt))

- [PPO精简实现]([GitHub - seungeunrho/minimalRL: Implementations of basic RL algorithms with minimal lines of codes! (pytorch based)](https://github.com/seungeunrho/minimalRL))

- [PPO实现](https://github.com/nikhilbarhate99/PPO-PyTorch)

- [ppo优化技巧]([GitHub - Lizhi-sjtu/DRL-code-pytorch: Concise pytorch implements of DRL algorithms, including REINFORCE, A2C, DQN, PPO(discrete and continuous), DDPG, TD3, SAC.](https://github.com/Lizhi-sjtu/DRL-code-pytorch)) 

## 功能

1. 支持离散和连续动作
2. 多进程采集数据
3. 支持lstm
4. 支持复杂输入(图像 与 向量)

## 版本

python 3.8

```
torch                              1.12.1
torchaudio                         0.12.1
torchvision                        0.13.1
gym                                0.26.2
gymnasium                          0.27.1
```