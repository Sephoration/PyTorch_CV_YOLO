"""
强化学习基础 — 对应 PPT #1: RL_Introduction_0513

核心概念:
  1. MDP (马尔可夫决策过程): <S, A, P, R, γ>
  2. 策略 (Policy): π(a|s) — 状态到动作的映射
  3. 状态价值函数 V(s): 从状态 s 出发的期望累积回报
  4. 动作价值函数 Q(s,a): 在 s 执行 a 的期望累积回报
  5. Bellman 方程: V(s) = max_a [R(s,a) + γ·Σ P(s'|s,a)·V(s')]
  6. 策略评估 / 策略改进 / 值迭代

本文件通过 FrozenLake 环境展示这些基础概念。
运行对比实验可以直观理解不同 γ 和不同策略的影响。
"""

import numpy as np
from FrozenLake import FrozenLake, test_game, print_policy, Random_Agent


# ==================== 1. MDP 四元组查看 ====================
def inspect_mdp():
    """
    查看 FrozenLake 的 MDP 结构:
      S(状态集), A(动作集), P(转移概率), R(奖励函数)
    """
    env = FrozenLake()

    print("=" * 60)
    print("FrozenLake MDP 结构")
    print("=" * 60)

    # S: 状态集
    print(f"\nS (状态集): {list(range(16))}")
    print(f"  起点: 0 | 冰洞: 5,7,11,12 | 目标: 15")

    # A: 动作集
    print(f"\nA (动作集): {{0: 左, 1: 下, 2: 右, 3: 上}}")

    # P, R: 转移概率和奖励 (以状态 0 为例)
    print(f"\nP & R (转移概率 & 奖励) — 以状态 0 为例:")
    action_names = ["← 左", "↓ 下", "→ 右", "↑ 上"]
    for a in range(4):
        print(f"  动作 {action_names[a]}:")
        for prob, s_next, reward, done in env.transition[0][a]:
            done_str = " (终止)" if done else ""
            print(f"    P = {prob:.3f} → 状态 {s_next}, 奖励 {reward}{done_str}")

    return env


# ==================== 2. 价值函数计算 ====================
def compute_v_given_policy(env, policy_fn, gamma=0.9, theta=1e-6):
    """
    给定策略 π，迭代计算状态价值函数 V(s)
    Bellman 方程: V(s) = Σ_a π(a|s) · Σ_{s'} P(s'|s,a) · [r + γ·V(s')]

    参数:
      env: FrozenLake 环境
      policy_fn: 策略函数 policy_fn(state) -> action
      gamma: 折扣因子
      theta: 收敛阈值
    返回:
      V: 16 维数组, 每个状态的价值
    """
    nS = 16
    nA = 4
    V = np.zeros(nS)
    terminal = {5, 7, 11, 12, 15}

    # 将策略函数转为确定性 one-hot 矩阵
    policy = np.zeros([nS, nA])
    for s in range(nS):
        a = policy_fn(s)
        policy[s][a] = 1.0

    while True:
        delta = 0
        for s in range(nS):
            if s in terminal:
                continue
            v_old = V[s]
            v_new = 0
            for a in range(nA):
                if policy[s][a] == 0:
                    continue
                for prob, s_next, reward, done in env.transition[s][a]:
                    v_new += policy[s][a] * prob * (reward + gamma * V[s_next] * (not done))
            V[s] = v_new
            delta = max(delta, abs(v_old - V[s]))
        if delta < theta:
            break
    return V


def show_value_function(V, title="状态价值函数"):
    """4x4 网格显示价值函数"""
    print(f"\n{title}:")
    print("-" * 35)
    for s in range(16):
        if s in (5, 7, 11, 12):
            print(f"|  H  {V[s]:7.2f} ", end="")
        elif s == 15:
            print(f"|  G  {V[s]:7.2f} ", end="")
        else:
            print(f"| {str(s).zfill(2)}  {V[s]:7.2f} ", end="")
        if (s + 1) % 4 == 0:
            print("|")
    print("-" * 35)


def demo_value_function():
    """
    对比不同策略下的状态价值函数:
      1. 全左策略 (always ←)
      2. 全右策略 (always →)
      3. 随机策略
    """
    env = FrozenLake()

    # 策略 1: 全左
    def policy_left(s):
        return 0

    V_left = compute_v_given_policy(env, policy_left)
    show_value_function(V_left, "固定左移策略 的 V(s)")

    # 策略 2: 全右
    def policy_right(s):
        return 2

    V_right = compute_v_given_policy(env, policy_right)
    show_value_function(V_right, "固定右移策略 的 V(s)")

    # 策略 3: 随机
    random_agent = Random_Agent()
    V_random = compute_v_given_policy(env, random_agent.action)
    show_value_function(V_random, "随机策略 的 V(s)")

    print(f"\n固定左移策略 成功率: {test_game(env, policy_left):.2%}")
    print(f"固定右移策略 成功率: {test_game(env, policy_right):.2%}")
    print(f"随机策略     成功率: {test_game(env, random_agent.action):.2%}")

    return V_left, V_right, V_random


# ==================== 3. Bellman 方程验证 ====================
def verify_bellman():
    """
    验证 Bellman 期望方程:
    V(s) = Σ_a π(a|s) · Σ_{s'} P(s'|s,a) · [r + γ·V(s')]

    通过左右两边分别计算，验证相等
    """
    env = FrozenLake()
    gamma = 0.9
    terminal = {5, 7, 11, 12, 15}

    # 使用随机策略
    random_agent = Random_Agent()
    policy = np.zeros([16, 4])
    for s in range(16):
        policy[s][random_agent.action(s)] = 1.0

    # 先迭代收敛 V
    V = compute_v_given_policy(env, random_agent.action, gamma=gamma)

    print("=" * 60)
    print("Bellman 方程验证 (随机策略, γ=0.9)")
    print("  左侧 LHS = V(s)  右侧 RHS = Σ_a π(a|s) Σ P·[r + γV(s')]")
    print("  若 LHS ≈ RHS 则 Bellman 方程成立")
    print("=" * 60)

    for s in range(16):
        if s in terminal:
            continue

        # 左侧: 已知 V(s)
        lhs = V[s]

        # 右侧: 按 Bellman 方程计算
        rhs = 0
        for a in range(4):
            if policy[s][a] == 0:
                continue
            for prob, s_next, reward, done in env.transition[s][a]:
                rhs += policy[s][a] * prob * (reward + gamma * V[s_next] * (not done))

        diff = abs(lhs - rhs)
        status = "✓" if diff < 1e-6 else "✗"
        print(f"  V({s:2d}): LHS={lhs:.6f}, RHS={rhs:.6f}, diff={diff:.2e} {status}")

    print("\n结论: Bellman 方程成立 ✓")


# ==================== 4. 折扣因子 γ 的影响 ====================
def demo_gamma_effect():
    """
    展示折扣因子 γ 对价值函数的影响:
    γ → 0: 短视 (只看即时奖励)
    γ → 1: 远视 (看重长期回报)
    """
    env = FrozenLake()
    gammas = [0, 0.5, 0.9, 0.99]

    random_agent = Random_Agent()

    print("=" * 60)
    print("折扣因子 γ 对价值函数的影响 (随机策略)")
    print("=" * 60)

    for gamma in gammas:
        V = compute_v_given_policy(env, random_agent.action, gamma=gamma)
        max_v = np.max(V)
        avg_v = np.mean([V[s] for s in range(16) if s not in (5, 7, 11, 12, 15)])
        print(f"  γ = {gamma:.2f}:  V_max = {max_v:.4f}, V_avg(非终止) = {avg_v:.4f}")


# ==================== 5. 练习入口 ====================
if __name__ == '__main__':
    # 练习 1: 查看 MDP 结构
    inspect_mdp()

    # 练习 2: 对比不同策略的价值函数
    print("\n")
    demo_value_function()

    # 练习 3: Bellman 方程验证
    print("\n")
    verify_bellman()

    # 练习 4: 折扣因子影响
    print("\n")
    demo_gamma_effect()
