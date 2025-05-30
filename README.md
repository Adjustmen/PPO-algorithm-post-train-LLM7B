# PPO-algorithm-post-train-LLM-7B
## 本项目的基本思路
Trained with the PPO algorithm, quantized and fine-tuned with Lora, the 7B large model can run on lightweight cloud servers

本项目基于RL对LLM-7B的模型进行post-train以实现LLM-7B可以自主的玩zork游戏，在我们的训练过程我们使用了LLM7-B进行训练

## 项目基本流程

**1、腾讯云服务器的配置**

（1）注册腾讯云服务器

（2）创建云服务实例

（3）登录服务器

```
ssh ubuntu@<公网IP>  # Ubuntu系统默认用户为ubuntu
```
输入密码或使用密钥对认证。

**2、安装需要的依赖** 

```
sudo apt update && sudo apt upgrade -y
```
**2、配置python环境**

安装minconda

```
wget https://repo.anaconda.com/miniconda/MiniConda3-latest-Linux-x86_64.sh
bash MiniConda3-latest-Linux-x86_64.sh
source ~/.bashrc
```
创建虚拟环境

```
conda create -n llm_rl python=3.9
conda activate llm_rl
```

**3、部署zork环境**

安装joerich框架

```
git clone https://github.com/Microsoft/jericho.git
cd jericho
pip install -e .
```
测试zork环境

```
import jericho
env = jericho.FrotzEnv("zork1.z5")  # 需下载游戏文件（如zork1.z5）
```

**4、**
