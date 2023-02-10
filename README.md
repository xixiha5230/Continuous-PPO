# PPO-PyTorch

## 参考

- [GitHub - BlueFisher/Advanced-Soft-Actor-Critic: Soft Actor-Critic with advanced features](https://github.com/BlueFisher/Advanced-Soft-Actor-Critic)

- [GitHub - seungeunrho/minimalRL: Implementations of basic RL algorithms with minimal lines of codes! (pytorch based)](https://github.com/seungeunrho/minimalRL)

- [GitHub - nikhilbarhate99/PPO-PyTorch: Minimal implementation of clipped objective Proximal Policy Optimization (PPO) in PyTorch](https://github.com/nikhilbarhate99/PPO-PyTorch)

- [GitHub - MarcoMeter/recurrent-ppo-truncated-bptt: Baseline implementation of recurrent PPO using truncated BPTT](https://github.com/MarcoMeter/recurrent-ppo-truncated-bptt)

## 功能

支持离散和连续动作

支持lstm

支持复杂输入

## 特性？

ref：`https://zhuanlan.zhihu.com/p/512327050`

1. 连续动作下，`actor`网络的输出层只输出`mean`，同时采用`nn.Parameter`的方式来训练一个“状态独立”的`log_std`，这往往比直接让神经网络同时输出`mean`和`std`效果好。
2. 

## 版本

python 3.8

```
torch                              1.12.1
torchaudio                         0.12.1
torchvision                        0.13.1
gym                                0.26.2
```