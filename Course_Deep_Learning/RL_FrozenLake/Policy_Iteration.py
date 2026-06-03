"""
策略迭代 (Policy Iteration) — 对应 PPT #3: RL_策略迭代_0527

算法步骤:
  1. Policy Evaluation (策略评估): 对当前策略 π 迭代计算状态价值函数 V(s)
  2. Policy Improvement (策略改进): 基于 V(s) 贪心更新策略 π
  3. 重复直到策略收敛 (π 不再变化)

依赖:
  - FrozenLake.py 中的 FrozenLake 环境 (老师提供的环境代码)
  - test_game() / print_policy() 用于测试和可视化
"""

import numpy as np
from FrozenLake import FrozenLake, test_game, print_policy


class PolicyIteration:
    """
    策略迭代求解 FrozenLake 最优策略
    """

    def __init__(self, env, gamma=0.9, theta=1e-6):
        """
        参数:
          env: FrozenLake 环境实例
          gamma: 折扣因子 (默认 0.9)
          theta: 策略评估收敛阈值 (默认 1e-6)
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.nS = 16          # 状态数 (4x4)
        self.nA = 4           # 动作数 (0=左,1=下,2=右,3=上)
        self.V = np.zeros(self.nS)        # 状态价值函数
        # 初始策略: 均匀随机 (每个状态每个动作等概率)
        self.policy = np.ones([self.nS, self.nA]) / self.nA

        # 终止状态 (冰洞 + 目标) — 在这些状态上 V=0, 策略无需改进
        self.terminal_states = {5, 7, 11, 12, 15}

    # -------------------- 模型提取: 从环境 transition 构建 P(s'|s,a) --------------------
    def _build_model(self):
        """
        将 FrozenLake.transition 转为标准格式:
          P[s][a] = [(prob, next_state, reward, done), ...]
        """
        P = {}
        for s in range(self.nS):
            P[s] = {}
            for a in range(self.nA):
                P[s][a] = self.env.transition[s][a]
        return P

    # -------------------- 策略评估 (Policy Evaluation) --------------------
    def policy_evaluation(self, policy=None, V=None):
        """
        迭代策略评估: 计算当前策略下的状态价值函数 V(s)

        V(s) = Σ_a π(a|s) · Σ_{s'} P(s'|s,a) · [r + γ·V(s')]

        参数:
          policy: 策略矩阵 [nS, nA], 默认使用 self.policy
          V: 初始价值函数, 默认使用 self.V
        返回:
          V: 收敛后的状态价值函数
        """
        if policy is None:
            policy = self.policy
        if V is None:
            V = self.V.copy()

        P = self._build_model()

        while True:
            delta = 0
            for s in range(self.nS):
                if s in self.terminal_states:
                    continue

                v_old = V[s]
                v_new = 0
                for a in range(self.nA):
                    action_prob = policy[s][a]
                    if action_prob == 0:
                        continue
                    for prob, s_next, reward, done in P[s][a]:
                        # V(s') = 0 if done (终止状态不再有未来回报)
                        v_new += action_prob * prob * (reward + self.gamma * V[s_next] * (not done))
                V[s] = v_new
                delta = max(delta, abs(v_old - V[s]))

            if delta < self.theta:
                break

        return V

    # -------------------- 策略改进 (Policy Improvement) --------------------
    def policy_improvement(self, V=None):
        """
        贪心策略改进:
          π_new(s) = argmax_a Σ_{s'} P(s'|s,a) · [r + γ·V(s')]

        参数:
          V: 当前价值函数
        返回:
          policy_stable: 策略是否已收敛 (True = 最优)
          new_policy: 改进后的策略 (确定性策略, one-hot 形式)
        """
        if V is None:
            V = self.V

        P = self._build_model()
        policy_stable = True
        new_policy = np.zeros([self.nS, self.nA])

        for s in range(self.nS):
            if s in self.terminal_states:
                # 终止状态: 保持任意策略
                new_policy[s] = self.policy[s]
                continue

            # 计算每个动作的 action value Q(s,a)
            q_values = np.zeros(self.nA)
            for a in range(self.nA):
                for prob, s_next, reward, done in P[s][a]:
                    q_values[a] += prob * (reward + self.gamma * V[s_next] * (not done))

            # 选最优动作 (贪心)
            best_action = np.argmax(q_values)
            new_policy[s][best_action] = 1.0

            # 检查策略是否变化
            if not np.array_equal(new_policy[s], self.policy[s]):
                policy_stable = False

        return policy_stable, new_policy

    # -------------------- 主循环: 策略迭代 --------------------
    def solve(self, max_iterations=1000):
        """
        执行策略迭代主循环, 直到策略收敛

        参数:
          max_iterations: 最大迭代次数
        返回:
          V: 最优状态价值函数
          policy: 最优策略 (确定性)
          n_iter: 实际迭代次数
        """
        print("=" * 60)
        print("开始策略迭代 (Policy Iteration)")
        print(f"  折扣因子 γ = {self.gamma}")
        print(f"  收敛阈值 θ = {self.theta}")
        print("=" * 60)

        for i in range(max_iterations):
            # Step 1: 策略评估
            self.V = self.policy_evaluation(policy=self.policy, V=self.V)

            # Step 2: 策略改进
            policy_stable, self.policy = self.policy_improvement(V=self.V)

            print(f"\n迭代 {i + 1}: 策略评估完成, 最大 V = {np.max(self.V):.4f}")

            if policy_stable:
                print(f"\n✓ 第 {i + 1} 次迭代后策略收敛！")
                return self.V, self.policy, i + 1

        print(f"\n⚠ 达到最大迭代次数 {max_iterations}, 策略可能未完全收敛")
        return self.V, self.policy, max_iterations

    # -------------------- 结果展示 --------------------
    def extract_deterministic_policy(self):
        """从概率策略中提取确定性策略: state -> action"""
        return {s: np.argmax(self.policy[s]) for s in range(self.nS)}

    def show_values(self):
        """4x4 网格显示状态价值函数"""
        print("\n状态价值函数 V(s):")
        print("-" * 35)
        for s in range(self.nS):
            if s in self.terminal_states:
                label = " H" if s != 15 else " G"
                print(f"| {label}  {self.V[s]:7.2f} ", end="")
            else:
                print(f"| {str(s).zfill(2)}  {self.V[s]:7.2f} ", end="")
            if (s + 1) % 4 == 0:
                print("|")
        print("-" * 35)


# ==================== 练习: 从随机策略开始对比 ====================
def demo_random_vs_optimal():
    """对比 Random Agent 和 策略迭代得到的最优策略"""
    env = FrozenLake()

    # ----- 1. 随机策略 -----
    print("\n" + "█" * 60)
    print("  随机策略 (Random Agent)")
    print("█" * 60)
    from FrozenLake import Random_Agent
    random_agent = Random_Agent()
    print_policy(random_agent.action)
    random_rate = test_game(env, random_agent.action, n_episodes=1000)
    print(f"随机策略 成功率: {random_rate:.2%}")

    # ----- 2. 策略迭代 -----
    print("\n" + "█" * 60)
    print("  策略迭代求解最优策略")
    print("█" * 60)
    solver = PolicyIteration(env)

    # 先把策略初始化为随机策略 (与 Random_Agent 保持一致)
    solver.policy = np.ones([solver.nS, solver.nA]) / solver.nA

    V_opt, policy_opt, n_iter = solver.solve()
    solver.show_values()

    # ----- 3. 最优策略可视化 -----
    opt_policy_dict = solver.extract_deterministic_policy()

    def optimal_policy_fn(state):
        return opt_policy_dict[state]

    print("\n最优策略:")
    print_policy(optimal_policy_fn)

    # ----- 4. 测试最优策略 -----
    opt_rate = test_game(env, optimal_policy_fn, n_episodes=1000)
    print(f"\n最优策略 成功率: {opt_rate:.2%}")
    print(f"策略迭代收敛次数: {n_iter}")
    print(f"性能提升: {opt_rate / max(random_rate, 0.001) - 1:.1%}")

    return opt_policy_dict, V_opt


# ==================== 练习: 调整折扣因子对比 ====================
def demo_gamma_comparison():
    """比较不同折扣因子 γ 对最优策略的影响"""
    env = FrozenLake()
    gammas = [0.5, 0.9, 0.99]

    print("\n" + "█" * 60)
    print("  不同折扣因子对比")
    print("█" * 60)

    for gamma in gammas:
        solver = PolicyIteration(env, gamma=gamma)
        solver.policy = np.ones([solver.nS, solver.nA]) / solver.nA
        V_opt, policy_opt, n_iter = solver.solve()

        opt_dict = solver.extract_deterministic_policy()

        def policy_fn(s, d=opt_dict):
            return d[s]

        rate = test_game(env, policy_fn, n_episodes=1000)
        print(f"\n  γ = {gamma:.2f}: 迭代 {n_iter} 次, 最大 V = {np.max(V_opt):.4f}, 成功率 = {rate:.2%}")


if __name__ == '__main__':
    # 基础测试: 展示策略迭代过程
    env = FrozenLake()
    solver = PolicyIteration(env, gamma=0.9)
    solver.policy = np.ones([solver.nS, solver.nA]) / solver.nA
    V_opt, policy_opt, n_iter = solver.solve()
    solver.show_values()

    opt_dict = solver.extract_deterministic_policy()
    print("\n最优策略:")
    print_policy(lambda s: opt_dict[s])

    # 测试最优策略
    rate = test_game(env, lambda s: opt_dict[s], n_episodes=1000)
    print(f"\n最优策略 成功率: {rate:.2%}")
    print(f"收敛迭代次数: {n_iter}")

    # 对比实验
    demo_random_vs_optimal()
    demo_gamma_comparison()
