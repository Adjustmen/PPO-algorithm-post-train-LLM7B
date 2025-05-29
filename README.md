# PPO-algorithm-post-train-LLM-7B
## 本项目的基本思路
Trained with the PPO algorithm, quantized and fine-tuned with Lora, the 7B large model can run on lightweight cloud servers

本项目基于RL对LLM-7B的模型进行post-train以实现LLM-7B可以自主的玩zork游戏，在我们的训练过程我们使用了LLM7-B进行训练

## 项目基本流程

**1** 腾讯云服务器的讲 

（1）注册腾讯云服务器
（2）创建云服务实例
（3）登录服务器

```
ssh ubuntu@<公网IP>  # Ubuntu系统默认用户为ubuntu
```

**2** 
