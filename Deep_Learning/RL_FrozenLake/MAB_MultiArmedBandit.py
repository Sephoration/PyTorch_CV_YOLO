"""
探索与利用 — 对应 PPT #4: 探索和利用_0610

多臂老虎机 (Multi-Armed Bandit, MAB) 问题:
  面前有 K 台老虎机，每台的奖励概率不同。
  你需要决定: 去探索未知的机器(探索)，还是坚持当前最好的机器(利用)？

核心策略:
  ε-greedy: 以 ε 概率随机探索，以 1-ε 概率选择当前最优
"""

import random
import matplotlib.pyplot as plt
import numpy as np


class Bandit:
    """
    单台老虎机
    每次拉杆以概率 p 奖励 1，否则奖励 0
    """

    def __init__(self, p):
        self.p = p

    def pull(self):
        return 1 if random.random() < self.p else 0


class BanditGame:
    """
    多臂老虎机游戏
    管理 K 台老虎机，记录玩家选择与奖励
    """

    def __init__(self, probs):
        self.bandits = [Bandit(p) for p in probs]
        self.K = len(probs)
        self.reset()

    def reset(self):
        """重置游戏记录"""
        self.Q = np.zeros(self.K)            # 每台机器的估计价值
        self.N = np.zeros(self.K, dtype=int) # 每台机器的尝试次数
        self.total_reward = 0
        self.reward_history = []

    def step(self, action):
        """
        执行动作: 拉第 action 号老虎机
        返回: 奖励
        """
        reward = self.bandits[action].pull()
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]
        self.total_reward += reward
        self.reward_history.append(reward)
        return reward

    def get_best_action(self):
        """返回当前估计价值最高的动作"""
        return np.argmax(self.Q)


class EpsilonGreedyAgent:
    """
    ε-greedy 策略 Agent
    以 ε 概率随机探索
    以 1-ε 概率选择当前最优动作
    """

    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.name = f"ε={epsilon}"

    def act(self, game):
        if random.random() < self.epsilon:
            return random.randrange(game.K)
        else:
            return game.get_best_action()


def simulate_one(agent, game, steps=1000):
    """
    在游戏上模拟一个 agent 的完整过程
    steps: 拉杆次数
    返回: 累积奖励列表
    """
    game.reset()
    cumulative = []
    total = 0

    for _ in range(steps):
        action = agent.act(game)
        reward = game.step(action)
        total += reward
        cumulative.append(total)

    return cumulative


def compare_strategies(probs=[0.1, 0.3, 0.8], steps=1000, trials=100):
    """
    对比不同 ε 值的累积奖励表现
    重复 trials 次取平均，使结果更稳定
    """
    agents = [
        EpsilonGreedyAgent(epsilon=0.0),   # 纯利用
        EpsilonGreedyAgent(epsilon=0.1),   # 10% 探索
        EpsilonGreedyAgent(epsilon=0.3),   # 30% 探索
        EpsilonGreedyAgent(epsilon=1.0),   # 纯探索
    ]

    print("=" * 60)
    print("多臂老虎机 (Multi-Armed Bandit)")
    print("=" * 60)
    print(f"\n老虎机配置: {len(probs)} 台")
    for i, p in enumerate(probs):
        print(f"  机器 {i+1}: 奖励概率 {p} (最优: 机器 {np.argmax(probs)+1})")

    plt.figure(figsize=(12, 5))

    for agent in agents:
        all_cumulative = np.zeros(steps)

        for _ in range(trials):
            game = BanditGame(probs)
            cum = simulate_one(agent, game, steps)
            all_cumulative += np.array(cum)

        avg_cumulative = all_cumulative / trials
        final_reward = avg_cumulative[-1]

        print(f"\n{agent.name}:")
        print(f"  最终平均累积奖励: {final_reward:.1f} / {steps}")
        print(f"  平均每次拉杆收益: {final_reward/steps:.3f}")

        plt.plot(avg_cumulative, label=agent.name, linewidth=2)

    plt.xlabel("拉杆次数")
    plt.ylabel("累积奖励")
    plt.title(f"不同 ε 策略对比 (平均 {trials} 次试验)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("MAB_comparison.png", dpi=150)
    plt.show()


def manual_game():
    """
    手动交互模式:
    玩家自己选择拉哪台机器，实时观察奖励
    """
    probs = [0.1, 0.5, 0.8]
    game = BanditGame(probs)

    print("\n" + "=" * 60)
    print("手动模式: 选择老虎机")
    print("=" * 60)
    print(f"\n老虎机 1: 奖励概率 {probs[0]}")
    print(f"老虎机 2: 奖励概率 {probs[1]}")
    print(f"老虎机 3: 奖励概率 {probs[2]}")
    print()

    for t in range(20):
        print(f"\n--- 第 {t+1} 次拉杆 ---")
        print(f"当前估计价值: {[f'{q:.2f}' for q in game.Q]}")
        print(f"各机器已尝试: {game.N}")

        try:
            action = int(input("请选择老虎机 (1/2/3): ")) - 1
            if action not in range(3):
                print("输入无效，默认为 1")
                action = 0
        except (ValueError, EOFError):
            action = 0

        reward = game.step(action)
        print(f"奖励: {reward}")
        print(f"总奖励: {game.total_reward}")


if __name__ == '__main__':
    # 练习: 自动对比不同策略
    compare_strategies(probs=[0.1, 0.5, 0.8], steps=500, trials=50)

    # 练习: 手动体验探索 vs 利用
    # manual_game()  # 取消注释可进入手动模式
