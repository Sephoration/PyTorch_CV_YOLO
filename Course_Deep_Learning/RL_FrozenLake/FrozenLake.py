import random
import numpy as np


class FrozenLake:
    """
    Frozen Lake 环境 (4x4)
    S: 起点(0), F: 冰面, H: 冰洞(5,7,11,12), G: 目标(15)
    动作: 0=左, 1=下, 2=右, 3=上
    冰面滑动: 每个动作有 1/3 概率分别滑向 意图方向 / 左偏90度 / 右偏90度
    """

    def __init__(self):
        self.reset()
        self.set_tran()

    def reset(self):
        """重置环境，Agent回到起点"""
        self.position = 0
        self.set_map()
        return self.position

    def set_map(self):
        """创建游戏地图，当前位置标记为 *"""
        self.map = list(range(16))
        self.map[self.position] = "*"

    def show(self):
        """显示当前游戏状态 (4x4 网格)"""
        self.set_map()
        print(f"state: {self.position}")
        for i, s in enumerate(self.map):
            print("| ", end="")
            if s == "*":
                print(s, end="  ")
            else:
                print(str(s).zfill(2), end=" ")
            if (i + 1) % 4 == 0:
                print("|")

    def set_tran(self):
        """
        设置状态转移表
        self.transition[state][action] = [(p, s', r, done), ...]
        """
        self.transition = {
            # ===================== state 0 (起点, 左上) =====================
            0: {
                0: [(2 / 3, 0, 0.0, False),   (1 / 3, 4, 0.0, False)],
                1: [(1 / 3, 0, 0.0, False),   (1 / 3, 4, 0.0, False),   (1 / 3, 1, 0.0, False)],
                2: [(1 / 3, 4, 0.0, False),   (1 / 3, 1, 0.0, False),   (1 / 3, 0, 0.0, False)],
                3: [(1 / 3, 1, 0.0, False),   (2 / 3, 0, 0.0, False)],
            },
            # ===================== state 1 (冰面) =====================
            1: {
                0: [(1 / 3, 1, 0.0, False),   (1 / 3, 0, 0.0, False),   (1 / 3, 5, 0.0, True)],
                1: [(1 / 3, 0, 0.0, False),   (1 / 3, 5, 0.0, True),    (1 / 3, 2, 0.0, False)],
                2: [(1 / 3, 5, 0.0, True),    (1 / 3, 2, 0.0, False),   (1 / 3, 1, 0.0, False)],
                3: [(1 / 3, 2, 0.0, False),   (1 / 3, 1, 0.0, False),   (1 / 3, 0, 0.0, False)],
            },
            # ===================== state 2 (冰面) =====================
            2: {
                0: [(1 / 3, 2, 0.0, False),   (1 / 3, 1, 0.0, False),   (1 / 3, 6, 0.0, False)],
                1: [(1 / 3, 1, 0.0, False),   (1 / 3, 6, 0.0, False),   (1 / 3, 3, 0.0, False)],
                2: [(1 / 3, 6, 0.0, False),   (1 / 3, 3, 0.0, False),   (1 / 3, 2, 0.0, False)],
                3: [(1 / 3, 3, 0.0, False),   (1 / 3, 2, 0.0, False),   (1 / 3, 1, 0.0, False)],
            },
            # ===================== state 3 (右上, 冰洞7旁边) =====================
            3: {
                0: [(1 / 3, 3, 0.0, False),   (1 / 3, 2, 0.0, False),   (1 / 3, 7, 0.0, True)],
                1: [(1 / 3, 2, 0.0, False),   (1 / 3, 7, 0.0, True),    (1 / 3, 3, 0.0, False)],
                2: [(1 / 3, 7, 0.0, True),    (2 / 3, 3, 0.0, False)],
                3: [(2 / 3, 3, 0.0, False),   (1 / 3, 2, 0.0, False)],
            },
            # ===================== state 4 (冰面) =====================
            4: {
                0: [(1 / 3, 0, 0.0, False),   (1 / 3, 4, 0.0, False),   (1 / 3, 8, 0.0, False)],
                1: [(1 / 3, 4, 0.0, False),   (1 / 3, 8, 0.0, False),   (1 / 3, 5, 0.0, True)],
                2: [(1 / 3, 8, 0.0, False),   (1 / 3, 5, 0.0, True),    (1 / 3, 0, 0.0, False)],
                3: [(1 / 3, 5, 0.0, True),    (1 / 3, 0, 0.0, False),   (1 / 3, 4, 0.0, False)],
            },
            # ===================== state 5 (冰洞 H) =====================
            5: {
                0: [(1.0, 5, 0, True)],
                1: [(1.0, 5, 0, True)],
                2: [(1.0, 5, 0, True)],
                3: [(1.0, 5, 0, True)],
            },
            # ===================== state 6 (冰面) =====================
            6: {
                0: [(1 / 3, 2, 0.0, False),   (1 / 3, 5, 0.0, True),    (1 / 3, 10, 0.0, False)],
                1: [(1 / 3, 5, 0.0, True),    (1 / 3, 10, 0.0, False),  (1 / 3, 7, 0.0, True)],
                2: [(1 / 3, 10, 0.0, False),  (1 / 3, 7, 0.0, True),    (1 / 3, 2, 0.0, False)],
                3: [(1 / 3, 7, 0.0, True),    (1 / 3, 2, 0.0, False),   (1 / 3, 5, 0.0, True)],
            },
            # ===================== state 7 (冰洞 H) =====================
            7: {
                0: [(1.0, 7, 0, True)],
                1: [(1.0, 7, 0, True)],
                2: [(1.0, 7, 0, True)],
                3: [(1.0, 7, 0, True)],
            },
            # ===================== state 8 (冰面) =====================
            8: {
                0: [(1 / 3, 4, 0.0, False),   (1 / 3, 8, 0.0, False),   (1 / 3, 12, 0.0, True)],
                1: [(1 / 3, 8, 0.0, False),   (1 / 3, 12, 0.0, True),   (1 / 3, 9, 0.0, False)],
                2: [(1 / 3, 12, 0.0, True),   (1 / 3, 9, 0.0, False),   (1 / 3, 4, 0.0, False)],
                3: [(1 / 3, 9, 0.0, False),   (1 / 3, 4, 0.0, False),   (1 / 3, 8, 0.0, False)],
            },
            # ===================== state 9 (冰面) =====================
            9: {
                0: [(1 / 3, 5, 0.0, True),    (1 / 3, 8, 0.0, False),   (1 / 3, 13, 0.0, False)],
                1: [(1 / 3, 8, 0.0, False),   (1 / 3, 13, 0.0, False),  (1 / 3, 10, 0.0, False)],
                2: [(1 / 3, 13, 0.0, False),  (1 / 3, 10, 0.0, False),  (1 / 3, 5, 0.0, True)],
                3: [(1 / 3, 10, 0.0, False),  (1 / 3, 5, 0.0, True),    (1 / 3, 8, 0.0, False)],
            },
            # ===================== state 10 (冰面) =====================
            10: {
                0: [(1 / 3, 6, 0.0, False),   (1 / 3, 9, 0.0, False),   (1 / 3, 14, 0.0, False)],
                1: [(1 / 3, 9, 0.0, False),   (1 / 3, 14, 0.0, False),  (1 / 3, 11, 0.0, True)],
                2: [(1 / 3, 14, 0.0, False),  (1 / 3, 11, 0.0, True),   (1 / 3, 6, 0.0, False)],
                3: [(1 / 3, 11, 0.0, True),   (1 / 3, 6, 0.0, False),   (1 / 3, 9, 0.0, False)],
            },
            # ===================== state 11 (冰洞 H) =====================
            11: {
                0: [(1.0, 11, 0, True)],
                1: [(1.0, 11, 0, True)],
                2: [(1.0, 11, 0, True)],
                3: [(1.0, 11, 0, True)],
            },
            # ===================== state 12 (冰洞 H) =====================
            12: {
                0: [(1.0, 12, 0, True)],
                1: [(1.0, 12, 0, True)],
                2: [(1.0, 12, 0, True)],
                3: [(1.0, 12, 0, True)],
            },
            # ===================== state 13 (冰面) =====================
            13: {
                0: [(1 / 3, 12, 0.0, True),   (1 / 3, 13, 0.0, False),  (1 / 3, 9, 0.0, False)],
                1: [(1 / 3, 13, 0.0, False),  (1 / 3, 14, 0.0, False),  (1 / 3, 12, 0.0, True)],
                2: [(1 / 3, 14, 0.0, False),  (1 / 3, 13, 0.0, False),  (1 / 3, 9, 0.0, False)],
                3: [(1 / 3, 9, 0.0, False),   (1 / 3, 14, 0.0, False),  (1 / 3, 12, 0.0, True)],
            },
            # ===================== state 14 (冰面, 目标旁边) =====================
            14: {
                0: [(1 / 3, 10, 0.0, False),  (1 / 3, 13, 0.0, False),  (1 / 3, 14, 0.0, False)],
                1: [(1 / 3, 13, 0.0, False),  (1 / 3, 14, 0.0, False),  (1 / 3, 15, 1.0, True)],
                2: [(1 / 3, 14, 0.0, False),  (1 / 3, 15, 1.0, True),   (1 / 3, 10, 0.0, False)],
                3: [(1 / 3, 15, 1.0, True),   (1 / 3, 10, 0.0, False),  (1 / 3, 13, 0.0, False)],
            },
            # ===================== state 15 (目标 G) =====================
            15: {
                0: [(1.0, 15, 0, True)],
                1: [(1.0, 15, 0, True)],
                2: [(1.0, 15, 0, True)],
                3: [(1.0, 15, 0, True)],
            },
        }

    def step(self, action):
        """
        执行动作，返回 (下一状态, 奖励, 是否结束)
        """
        node = self.transition[self.position][action]
        probs, states, rewards, dones = zip(*node)
        choice = random.choices(population=states, weights=probs, k=1)[0]
        i = states.index(choice)
        self.position = states[i]
        return states[i], rewards[i], dones[i]


class Random_Agent:
    """
    随机策略 Agent
    每个状态随机选择一个动作，固定记录在 policy_dict 中
    """

    def __init__(self):
        self.policy_dict = {
            k: random.choices(range(4), k=1)[0] for k in range(16)
        }

    def action(self, state):
        return self.policy_dict[state]


def print_policy(policy_fn):
    """
    以 4x4 网格显示策略: 每个格子显示动作箭头
    policy_fn: 函数 policy_fn(state) -> action
        动作: 0=左, 1=下, 2=右, 3=上
    """
    arrows = ["←", "↓", "→", "↑"]
    labels = {5: "H", 7: "H", 11: "H", 12: "H", 15: "G"}
    print("策略 (Policy):")
    print("-" * 21)
    for s in range(16):
        if s in labels:
            print(f"| {labels[s]:^3}", end=" ")
        else:
            print(f"| {arrows[policy_fn(s)]:^3}", end=" ")
        if (s + 1) % 4 == 0:
            print("|")
    print("-" * 21)


def test_game(env, pi, n_episodes=100, max_steps=100):
    """
    测试一个策略在环境中的表现
    env: 环境实例
    pi: 策略函数，输入 state 返回 action
    n_episodes: 测试回合数
    max_steps: 每回合最大步数
    返回: 成功率 (到达目标的比例)
    """
    results = []
    for _ in range(n_episodes):
        state = env.reset()
        Done = False
        steps = 0
        while not Done and steps < max_steps:
            action = pi(state)
            state, reward, Done = env.step(action)
            steps += 1
        results.append(reward > 0)
    return np.sum(results) / len(results)


if __name__ == '__main__':
    # ===== 练习: 创建 FrozenLake 实例并显示当前状态 =====
    env = FrozenLake()
    print("初始状态:")
    env.show()
    print()

    # 练习: Agent 执行 action=1 (向下) 后显示状态
    print("执行 action=1 (向下) 后:")
    print(env.step(1))
    env.show()
    print()

    # ===== 测试 Random Agent =====
    print("测试 Random Agent ...")
    agent = Random_Agent()
    success_rate = test_game(env, agent.action, n_episodes=1000, max_steps=100)
    print(f"Random Agent 成功率: {success_rate:.2%}")
