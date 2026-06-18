"""
策略迭代改造与多状态评估分析

功能:
  1. Value Iteration 实现 (与 Policy Iteration 对比)
  2. 收敛过程可视化 (V(s) 随迭代变化曲线)
  3. 多状态起始评估 (所有 16 个状态的 mean_return)
  4. gamma 超参数敏感度分析
  5. V(s) / Q(s,a) 热力图可视化
  6. 策略对比 (Random vs Human vs Optimal)

用法:
  python Policy_Iteration_Analysis.py

输出:
  figures/ 目录下所有 PNG 图表 (用于 Word 报告)
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from collections import defaultdict

from FrozenLake import FrozenLake, test_game, print_policy
from Policy_Iteration import PolicyIteration


FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ====================================================================
#  1. Value Iteration
# ====================================================================
class ValueIteration:
    """
    值迭代 (Value Iteration) — 与 Policy Iteration 对比分析

    核心区别:
      - Policy Iteration:  完整策略评估 → 策略改进 → 重复
      - Value Iteration:   一步 Bellman 更新 → 提取策略 (无显式策略评估)
    """

    def __init__(self, env, gamma=0.9, theta=1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.nS = 16
        self.nA = 4
        self.V = np.zeros(self.nS)
        self.terminal_states = {5, 7, 11, 12, 15}
        self._convergence_history = []

    def _build_model(self):
        P = {}
        for s in range(self.nS):
            P[s] = {}
            for a in range(self.nA):
                P[s][a] = self.env.transition[s][a]
        return P

    def solve(self, max_iterations=1000):
        """
        值迭代主循环: V(s) = max_a Σ P(s'|s,a) [r + γ V(s')]

        返回:
          V: 最优状态价值函数
          n_iter: 实际迭代次数
        """
        P = self._build_model()
        self._convergence_history = [self.V.copy()]

        for i in range(max_iterations):
            delta = 0
            V_new = np.zeros(self.nS)

            for s in range(self.nS):
                if s in self.terminal_states:
                    continue

                max_q = float('-inf')
                for a in range(self.nA):
                    q_val = 0
                    for prob, s_next, reward, done in P[s][a]:
                        q_val += prob * (reward + self.gamma * self.V[s_next] * (not done))
                    max_q = max(max_q, q_val)

                V_new[s] = max_q
                delta = max(delta, abs(self.V[s] - V_new[s]))

            self.V = V_new
            self._convergence_history.append(self.V.copy())

            if delta < self.theta:
                return self.V, i + 1

        return self.V, max_iterations

    def extract_policy(self):
        """从 V(s) 提取贪心策略"""
        P = self._build_model()
        policy = {}
        for s in range(self.nS):
            if s in self.terminal_states:
                policy[s] = 0
                continue
            q_values = np.zeros(self.nA)
            for a in range(self.nA):
                for prob, s_next, reward, done in P[s][a]:
                    q_values[a] += prob * (reward + self.gamma * self.V[s_next] * (not done))
            policy[s] = int(np.argmax(q_values))
        return policy

    def get_convergence_history(self):
        return np.array(self._convergence_history)


# ====================================================================
#  2. 多状态评估
# ====================================================================
def mean_return(env, pi, state0, n_episodes=5000, max_steps=100):
    """计算从 state0 开始使用策略 pi 的平均回报"""
    results = []
    for _ in range(n_episodes):
        env.position = state0
        Done = False
        steps = 0
        total_reward = 0.0
        while not Done and steps < max_steps:
            action = pi(state0) if callable(pi) else pi[state0]
            state0, reward, Done = env.step(action)
            total_reward += reward
            steps += 1
        results.append(total_reward)
    return float(np.mean(results))


# ====================================================================
#  3. 收敛过程可视化
# ====================================================================
def plot_convergence_comparison():
    """Policy Iteration vs Value Iteration 收敛速度对比"""
    env = FrozenLake()

    # --- Policy Iteration ---
    pi_solver = PolicyIteration(env, gamma=0.9, theta=1e-6)
    pi_solver.policy = np.ones([pi_solver.nS, pi_solver.nA]) / pi_solver.nA
    pi_V = [np.zeros(16)]
    for i in range(100):
        pi_solver.V = pi_solver.policy_evaluation(policy=pi_solver.policy, V=pi_solver.V)
        policy_stable, pi_solver.policy = pi_solver.policy_improvement(V=pi_solver.V)
        pi_V.append(pi_solver.V.copy())
        if policy_stable:
            break
    pi_V = np.array(pi_V)
    pi_iterations = pi_V.shape[0] - 1

    # --- Value Iteration ---
    vi_solver = ValueIteration(env, gamma=0.9, theta=1e-6)
    vi_V, vi_iters = vi_solver.solve(max_iterations=200)
    vi_history = vi_solver.get_convergence_history()

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 选择有代表性的状态
    repr_states = [0, 1, 6, 10, 14, 15]
    state_labels = ['S(起点)', '1', '6', '10', '14', 'G(目标)']

    ax = axes[0]
    for i, (s, label) in enumerate(zip(repr_states, state_labels)):
        values = pi_V[:, s]
        ax.plot(range(len(values)), values, label=f'State {s} {label}',
                linewidth=1.5, marker='o', markersize=3, markevery=max(1, len(values)//10))
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('V(s)', fontsize=12)
    ax.set_title(f'Policy Iteration Convergence (γ=0.9, {pi_iterations} iters)', fontsize=13)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.axvline(x=pi_iterations, color='red', linestyle='--', alpha=0.7, label=f'Converge at iter {pi_iterations}')

    ax = axes[1]
    for i, (s, label) in enumerate(zip(repr_states, state_labels)):
        values = vi_history[:, s]
        ax.plot(range(len(values)), values, label=f'State {s} {label}',
                linewidth=1.5, marker='s', markersize=3, markevery=max(1, len(values)//10))
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('V(s)', fontsize=12)
    ax.set_title(f'Value Iteration Convergence (γ=0.9, {vi_iters} iters)', fontsize=13)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.axvline(x=vi_iters, color='red', linestyle='--', alpha=0.7, label=f'Converge at iter {vi_iters}')

    plt.tight_layout()
    path = os.path.join(FIG_DIR, '01_convergence_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {path}")

    return pi_iterations, vi_iters


# ====================================================================
#  4. 多状态评估分析
# ====================================================================
def plot_multi_state_evaluation():
    """对所有 16 个状态进行多起始状态评估"""
    env = FrozenLake()

    # 训练最优策略
    solver = PolicyIteration(env, gamma=0.9)
    solver.policy = np.ones([solver.nS, solver.nA]) / solver.nA
    V_opt, policy_opt, _ = solver.solve()
    opt_dict = solver.extract_deterministic_policy()

    # 随机策略
    np.random.seed(42)
    random_policy = {s: np.random.randint(4) for s in range(16)}

    # Human Agent 策略 (heuristic)
    human_policy = {
        0: 2, 1: 2, 2: 1, 3: 0,
        4: 1, 5: 0, 6: 1, 7: 0,
        8: 2, 9: 2, 10: 1, 11: 0,
        12: 0, 13: 2, 14: 2, 15: 0
    }

    terminal_states = {5, 7, 11, 12, 15}
    state_names = []
    for s in range(16):
        if s == 0:
            state_names.append(f'{s}\nS')
        elif s in terminal_states:
            label = 'G' if s == 15 else 'H'
            state_names.append(f'{s}\n{label}')
        else:
            state_names.append(str(s))

    # 计算所有状态的 mean_return
    random_returns = []
    human_returns = []
    optimal_returns = []

    for s in range(16):
        r1 = mean_return(env, random_policy, s, n_episodes=5000)
        r2 = mean_return(env, human_policy, s, n_episodes=5000)
        r3 = mean_return(env, opt_dict, s, n_episodes=5000)
        random_returns.append(r1)
        human_returns.append(r2)
        optimal_returns.append(r3)
        print(f"    State {s:2d}: Random={r1:.3f}  Human={r2:.3f}  Optimal={r3:.3f}")

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(16)
    width = 0.25

    bars1 = ax.bar(x - width, random_returns, width, label='Random Agent',
                   color='#E74C3C', alpha=0.8, edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x, human_returns, width, label='Human Agent (Heuristic)',
                   color='#F39C12', alpha=0.8, edgecolor='white', linewidth=0.5)
    bars3 = ax.bar(x + width, optimal_returns, width, label='Optimal (Policy Iteration)',
                   color='#2ECC71', alpha=0.8, edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Starting State', fontsize=13)
    ax.set_ylabel('Mean Return', fontsize=13)
    ax.set_title('Multi-State Evaluation: Mean Return by Starting State', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(state_names, fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2, axis='y')
    ax.axhline(y=0, color='gray', linewidth=0.5)

    for i in range(16):
        if i in terminal_states:
            ax.axvspan(i - 0.4, i + 0.4, alpha=0.08, color='red')

    plt.tight_layout()
    path = os.path.join(FIG_DIR, '02_multi_state_evaluation.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {path}")

    return random_returns, human_returns, optimal_returns


# ====================================================================
#  5. Gamma 敏感度分析
# ====================================================================
def plot_gamma_sensitivity():
    """不同 gamma 对策略迭代收敛速度和性能的影响"""
    env = FrozenLake()
    gammas = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]

    iterations = []
    success_rates = []
    max_Vs = []

    for gamma in gammas:
        solver = PolicyIteration(env, gamma=gamma, theta=1e-6)
        solver.policy = np.ones([solver.nS, solver.nA]) / solver.nA
        V_opt, policy_opt, n_iter = solver.solve()
        opt_dict = solver.extract_deterministic_policy()

        rate = test_game(env, lambda s, d=opt_dict: d[s], n_episodes=2000)
        iterations.append(n_iter)
        success_rates.append(rate)
        max_Vs.append(np.max(V_opt))
        print(f"    γ={gamma:.2f}:  {n_iter} iters,  success={rate:.2%},  max V={np.max(V_opt):.4f}")

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = '#2980B9'
    color2 = '#E74C3C'
    color3 = '#27AE60'

    bars1 = ax1.bar([g - 0.02 for g in range(len(gammas))], iterations, width=0.25,
                    color=color1, alpha=0.8, label='Iterations to Converge', edgecolor='white')
    ax1.set_xlabel('Discount Factor γ', fontsize=13)
    ax1.set_ylabel('Number of Iterations', fontsize=13, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(range(len(gammas)), [r * 100 for r in success_rates], width=0.25,
                    color=color2, alpha=0.7, label='Success Rate (%)', edgecolor='white')
    ax2.set_ylabel('Success Rate (%)', fontsize=13, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    line3, = ax3.plot(range(len(gammas)), max_Vs, 'D-', color=color3, linewidth=2,
                       markersize=8, label='Max V(s)', zorder=5)
    ax3.set_ylabel('Max V(s)', fontsize=13, color=color3)
    ax3.tick_params(axis='y', labelcolor=color3)

    ax1.set_xticks(range(len(gammas)))
    ax1.set_xticklabels([f'{g:.2f}' for g in gammas], fontsize=11)

    bars = [bars1, bars2, line3]
    labels = ['Iterations to Converge', 'Success Rate (%)', 'Max V(s)']
    ax1.legend(bars, labels, loc='upper left', fontsize=10)

    ax1.set_title('Gamma Sensitivity Analysis (γ effect on convergence & performance)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    path = os.path.join(FIG_DIR, '03_gamma_sensitivity.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {path}")

    return gammas, iterations, success_rates


# ====================================================================
#  6. V(s) 与 Q(s,a) 热力图
# ====================================================================
def plot_v_q_heatmaps():
    """V(s) 和 Q(s,a) 热力图可视化"""
    env = FrozenLake()

    solver = PolicyIteration(env, gamma=0.9)
    solver.policy = np.ones([solver.nS, solver.nA]) / solver.nA
    V_opt, policy_opt, _ = solver.solve()
    opt_dict = solver.extract_deterministic_policy()

    # 计算 Q(s,a)
    P = {}
    for s in range(16):
        P[s] = {}
        for a in range(4):
            P[s][a] = env.transition[s][a]

    Q = np.zeros((16, 4))
    for s in range(16):
        if s in {5, 7, 11, 12, 15}:
            continue
        for a in range(4):
            for prob, s_next, reward, done in P[s][a]:
                Q[s][a] += prob * (reward + 0.9 * V_opt[s_next] * (not done))

    # 4x4 grid 标签
    grid_labels = []
    for r in range(4):
        row = []
        for c in range(4):
            s = r * 4 + c
            if s == 0:
                row.append(f'{s}\nS')
            elif s == 15:
                row.append(f'{s}\nG')
            elif s in {5, 7, 11, 12}:
                row.append(f'{s}\nH')
            else:
                row.append(str(s))
        grid_labels.append(row)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    v_grid = V_opt.reshape(4, 4)
    cmap = plt.cm.RdYlGn
    cmap.set_bad('gray')
    v_masked = np.ma.masked_where(np.isnan(v_grid), v_grid)
    im = ax.imshow(v_grid, cmap=cmap, vmin=0, vmax=np.max(V_opt), aspect='equal')

    for i in range(4):
        for j in range(4):
            s = i * 4 + j
            val = V_opt[s]
            color = 'white' if val > np.max(V_opt) * 0.6 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=15,
                    fontweight='bold', color=color)
            sub_label = grid_labels[i][j]
            ax.text(j, i - 0.32, sub_label, ha='center', va='center', fontsize=8, color='gray')

    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(['0', '1', '2', '3'])
    ax.set_yticklabels(['0', '1', '2', '3'])
    ax.set_title('State Value Function V(s) - Optimal Policy', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 标记孔洞
    for s in {5, 7, 11, 12}:
        r, c = s // 4, s % 4
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor='red', linewidth=3, linestyle='--'))

    # Q(s,a) 热力图
    ax = axes[1]
    action_labels = ['← LEFT', '↓ DOWN', '→ RIGHT', '↑ UP']
    im = ax.imshow(Q.T, cmap='viridis', aspect='auto')

    for a in range(4):
        for s in range(16):
            val = Q[s, a]
            if val != 0:
                color = 'white' if val > Q.max() * 0.6 else 'black'
                ax.text(s, a, f'{val:.2f}', ha='center', va='center', fontsize=7, color=color)

    ax.set_yticks(range(4))
    ax.set_yticklabels(action_labels, fontsize=11)
    ax.set_xticks(range(16))
    ax.set_xticklabels(range(16), fontsize=9)
    ax.set_xlabel('State', fontsize=12)
    ax.set_title('Action-Value Function Q(s,a)', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for s in {5, 7, 11, 12, 15}:
        ax.add_patch(plt.Rectangle((s - 0.5, -0.5), 1, 4, fill=False, edgecolor='red', linewidth=2, linestyle='--'))

    plt.tight_layout()
    path = os.path.join(FIG_DIR, '04_V_Q_heatmaps.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {path}")


# ====================================================================
#  7. 策略对比可视化
# ====================================================================
def plot_policy_comparison():
    """对比 Random / Human / Optimal 三种策略"""
    env = FrozenLake()

    solver = PolicyIteration(env, gamma=0.9)
    solver.policy = np.ones([solver.nS, solver.nA]) / solver.nA
    V_opt, policy_opt, _ = solver.solve()
    optimal_dict = solver.extract_deterministic_policy()

    random_dict = {s: np.random.randint(4) for s in range(16)}
    human_dict = {
        0: 2, 1: 2, 2: 1, 3: 0,
        4: 1, 5: 0, 6: 1, 7: 0,
        8: 2, 9: 2, 10: 1, 11: 0,
        12: 0, 13: 2, 14: 2, 15: 0
    }

    arrows = ['←', '↓', '→', '↑']
    labels = {5: 'H', 7: 'H', 11: 'H', 12: 'H', 15: 'G'}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    policies = [
        ('Random Agent', random_dict, '#E74C3C'),
        ('Human Agent (Heuristic)', human_dict, '#F39C12'),
        ('Optimal (Policy Iteration)', optimal_dict, '#2ECC71')
    ]

    for ax, (name, policy, color) in zip(axes, policies):
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(-0.5, 3.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.set_title(name, fontsize=13, fontweight='bold', color=color)

        for s in range(16):
            r, c = s // 4, s % 4
            if s in labels:
                ax.text(c, r, labels[s], ha='center', va='center', fontsize=24,
                        fontweight='bold', color='#333')
            else:
                ax.text(c, r, arrows[policy[s]], ha='center', va='center',
                        fontsize=28, color=color)
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                       edgecolor='#CCCCCC', linewidth=1))

        for s in {5, 7, 11, 12}:
            r, c = s // 4, s % 4
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=True,
                                       color='#FFE0E0', zorder=-1))

        ax.add_patch(plt.Rectangle((3.5, 3.5), 1, 1, fill=True,
                                   color='#E0FFE0', zorder=-1))

    plt.tight_layout()
    path = os.path.join(FIG_DIR, '05_policy_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {path}")


# ====================================================================
#  8. Value Iteration vs Policy Iteration 性能对比表
# ====================================================================
def plot_performance_comparison_table():
    """生成 PI vs VI 性能对比图"""
    env = FrozenLake()

    gammas_list = [0.5, 0.7, 0.9, 0.99]
    results = []

    for gamma in gammas_list:
        # Policy Iteration
        pi = PolicyIteration(env, gamma=gamma, theta=1e-6)
        pi.policy = np.ones([pi.nS, pi.nA]) / pi.nA
        t0 = time.time()
        V_pi, pol_pi, n_pi = pi.solve()
        t_pi = time.time() - t0
        pi_dict = pi.extract_deterministic_policy()
        rate_pi = test_game(env, lambda s, d=pi_dict: d[s], n_episodes=2000)

        # Value Iteration
        vi = ValueIteration(env, gamma=gamma, theta=1e-6)
        t0 = time.time()
        V_vi, n_vi = vi.solve(max_iterations=500)
        t_vi = time.time() - t0
        vi_dict = vi.extract_policy()
        rate_vi = test_game(env, lambda s, d=vi_dict: d[s], n_episodes=2000)

        results.append((gamma, n_pi, t_pi, rate_pi, n_vi, t_vi, rate_vi))
        print(f"    γ={gamma:.2f}:  PI={n_pi}iters/{t_pi:.3f}s/{rate_pi:.2%}  "
              f"VI={n_vi}iters/{t_vi:.3f}s/{rate_vi:.2%}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    g_labels = [f'γ={g:.2f}' for g in gammas_list]

    # Iterations
    ax = axes[0]
    x = np.arange(len(gammas_list))
    w = 0.35
    pi_iters = [r[1] for r in results]
    vi_iters = [r[4] for r in results]
    ax.bar(x - w/2, pi_iters, w, label='Policy Iteration', color='#2980B9', alpha=0.85)
    ax.bar(x + w/2, vi_iters, w, label='Value Iteration', color='#E74C3C', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(g_labels, fontsize=11)
    ax.set_ylabel('Iterations to Converge', fontsize=12)
    ax.set_title('Convergence Speed', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, axis='y')

    # Time
    ax = axes[1]
    pi_times = [r[2] for r in results]
    vi_times = [r[5] for r in results]
    ax.bar(x - w/2, [t * 1000 for t in pi_times], w, label='Policy Iteration', color='#2980B9', alpha=0.85)
    ax.bar(x + w/2, [t * 1000 for t in vi_times], w, label='Value Iteration', color='#E74C3C', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(g_labels, fontsize=11)
    ax.set_ylabel('Computation Time (ms)', fontsize=12)
    ax.set_title('Computation Time', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, axis='y')

    # Success Rate
    ax = axes[2]
    pi_rates = [r[3] * 100 for r in results]
    vi_rates = [r[6] * 100 for r in results]
    ax.bar(x - w/2, pi_rates, w, label='Policy Iteration', color='#2980B9', alpha=0.85)
    ax.bar(x + w/2, vi_rates, w, label='Value Iteration', color='#E74C3C', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(g_labels, fontsize=11)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_title('Policy Quality (Success Rate)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, axis='y')

    for r in pi_rates:
        if r > 0:
            print(f"    PI success rate: {r:.1f}%")
    for r in vi_rates:
        if r > 0:
            print(f"    VI success rate: {r:.1f}%")

    plt.tight_layout()
    path = os.path.join(FIG_DIR, '06_PI_vs_VI_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {path}")


# ====================================================================
#  9. 执行全部实验
# ====================================================================
def run_all():
    """运行所有分析实验并生成图表"""
    print("=" * 65)
    print("  策略迭代改造与多状态评估分析")
    print("  Policy Iteration: Improvement & Multi-State Evaluation")
    print("=" * 65)

    print("\n[1/6] 收敛过程对比 (PI vs VI)...")
    pi_iters, vi_iters = plot_convergence_comparison()
    print(f"  Policy Iteration: {pi_iters} iterations to converge")
    print(f"  Value Iteration:  {vi_iters} iterations to converge")

    print("\n[2/6] 多状态起始评估...")
    random_ret, human_ret, opt_ret = plot_multi_state_evaluation()

    print("\n[3/6] Gamma 敏感度分析...")
    gammas, iters, rates = plot_gamma_sensitivity()

    print("\n[4/6] V(s) / Q(s,a) 热力图...")
    plot_v_q_heatmaps()

    print("\n[5/6] 策略对比可视化...")
    plot_policy_comparison()

    print("\n[6/6] PI vs VI 性能对比...")
    plot_performance_comparison_table()

    print("\n" + "=" * 65)
    print(f"  所有图表已保存至: {FIG_DIR}/")
    print("=" * 65)
    print("\n生成的文件列表:")
    for f in sorted(os.listdir(FIG_DIR)):
        fpath = os.path.join(FIG_DIR, f)
        size = os.path.getsize(fpath)
        print(f"  {f:45s}  {size/1024:.1f} KB")


if __name__ == '__main__':
    run_all()
