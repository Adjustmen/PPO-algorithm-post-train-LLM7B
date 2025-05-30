class ZorkWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # 定义标准动作模板
        self.action_templates = [
            "go {direction}", "take {item}", 
            "use {item}", "examine {object}"
        ]
    
    def preprocess_observation(self, text):
        """清理游戏输出的冗余信息"""
        return remove_status_bars(text)
    
    def get_reward(self):
        """复合奖励计算"""
        base = self.env.get_score() 
        exploration = len(self.discovered_rooms)*0.1
        return base + exploration - 0.01  # 效率惩罚
