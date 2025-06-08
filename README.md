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

**4、加载大模型llm-7B**

安装库
```
pip install transformers torch accelerate
```

下载7B的大模型
```
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
```

**5、实现RL框架**

在实验过程中最终选择PPO算法并且在进行量化和lora微调的情况下来实现大模型的训练；
agent、PPO、量化和lora微调的代码均放置在experience文件夹中

**实验中进行的参数的改进**
- PPO算法的改进
- Lora微调参数改进

**（1）PPO算法参数调优**
```
  class AdvancedPPOTrainer:
    def __init__(self, ...):
        # 动态调整的超参数
        self.kl_coef = 0.2       # KL惩罚系数（初始值）
        self.clip_range = 0.2    # 策略裁剪范围
        self.gamma = 0.99        # 折扣因子
        self.lam = 0.95          # GAE系数
        self.entropy_coef = 0.01 # 熵奖励系数
        
    def update(self, samples):
        # 动态调整KL系数
        if self.kl_mean > 0.03:  # 监控KL散度
            self.kl_coef *= 1.5
        elif self.kl_mean < 0.01:
            self.kl_coef *= 0.8
        
        # 自适应clip_range
        self.clip_range = max(0.1, 0.2 * (1 - self.update_step/1e5))
```
参数改进的说明：

1、动态KL控制：根据实际KL散度动态调整惩罚系数，避免策略过早收敛

2、精细微调：随着训练进行逐步减小clip_range，后期微调更精细

3、熵奖励衰减：在训练后期降低熵系数，增强确定性

算法内部改进：

1、算法中增加了优势标准化和clip值函数
```
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
vf_loss = torch.max(
    (values - returns) ** 2,
    (values_clipped - returns) ** 2
)
```
2、增加平滑正则化策略
```
old_probs = torch.exp(old_log_probs)
new_probs = torch.exp(log_probs)
js_div = 0.5 * (kl_div(old_probs, (old_probs+new_probs)/2) + 
                kl_div(new_probs, (old_probs+new_probs)/2))
loss += 0.1 * js_div
```

**模型架构**
在这个模型中，我们一共创建了八个类，分别是：

- env，环境类，提供游戏的基本环境、游戏玩家（智能体）从该类中或许必要的信息

- agent 智能体，相当于游戏玩家

- action_precdure 表示智能体下一步所需要走的动作是前往哪里，内部调用了奖励函数，来评估每一步的token值是多少

- rewards 设计的奖励函数，通过设计不同的行为给予奖励或惩罚

- config 设置游戏配置，训练环境的配置和DQN算法训练的配置

- lora_comple and lora introduction表示为lora微调的参数设置

- PPO_loss，PPO算法的训练函数，通过改变参数增强算法的适用性

- optimize 量化函数， 节约显存提高效率；

## 实现过程
交互数据的收集
```
from jericho import FrotzEnv

def collect_trajectories(env_path="zork1.z5", num_episodes=10):
    env = FrotzEnv(env_path)
    trajectories = []
    
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        episode = []
        
        while not done:
            action = input(f"Obs: {obs}\nYour action: ") 
            new_obs, reward, done, _ = env.step(action)
            episode.append((obs, action, reward))
            obs = new_obs
            
        trajectories.append(episode)
    return trajectories
```
lora微调
```
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,                  
    lora_alpha=32,        
    target_modules=["q_proj", "v_proj"], 
    lora_dropout=0.05,
    bias="none"
)

model = AutoModelForCausalLM.from_pretrained("llm7b")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters() 
```
采用8-bit量化减少显存开销
```
from transformers import BitsAndBytesConfig

quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0  # 过滤大数值异常值
)

quantized_model = AutoModelForCausalLM.from_pretrained(
    "./fine_tuned_model",
    quantization_config=quant_config,
    device_map="auto"
)
```
PPO算法配置和奖励函数的设置
具体参数值需要在运行中能够得到最佳参数
```
    batch_size
    learning_rate
    ppo_epochs
    clip_range
    gamma
    gae_lambda
    target_kl
    seed
```
在训练过程反复调整学习率和熵系数
```
from trl import PPOTrainer

ppo_trainer = PPOTrainer(
    model=model,
    config=ppo_config,
    tokenizer=tokenizer
)

for epoch in range(100):
    # 生成动作
    queries = [env.reset() for _ in range(ppo_config.batch_size)]
    responses = [model.generate(q, max_length=20) for q in queries]
    
    # 环境执行
    rewards = []
    for action in responses:
        _, r, done, _ = env.step(action)
        rewards.append(r + 0.1 * calculate_curiosity(env.state))  # 自定义奖励
        
    # PPO 更新
    stats = ppo_trainer.step(queries, responses, rewards)
    
    # 日志记录
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Avg Reward = {np.mean(rewards):.2f}")
        model.save_pretrained(f"./checkpoints/epoch_{epoch}")
```
模型性能评估和优化
