# PPO-PyTorch

## 参考 感谢！

- [conv1d、conv2d网络](https://github.com/BlueFisher/Advanced-Soft-Actor-Critic)

- [lstm、多线程](https://github.com/MarcoMeter/recurrent-ppo-truncated-bptt)

- [PPO精简实现](https://github.com/seungeunrho/minimalRL)

- [PPO实现](https://github.com/nikhilbarhate99/PPO-PyTorch)

- [ppo优化技巧](https://github.com/Lizhi-sjtu/DRL-code-pytorch)

- [RND 随机网络蒸馏](https://github.com/alirezakazemipour/PPO-RND)
  
## 功能

1. 支持离散和连续动作
2. 多进程采集数据
3. 支持lstm
4. 支持复杂输入(图像 与 向量)
5. 支持rnd随机网络蒸馏(探索)
6. 多任务训练、任务预测、任务自主选择

## 注意事项

1. 所有环境返回的obs，都必须为list，即使只有一个观测信息

## 环境

python 3.8

```
torch                              1.12.1
torchaudio                         0.12.1
torchvision                        0.13.1
gym                                0.26.2
gymnasium                          0.27.1
```
