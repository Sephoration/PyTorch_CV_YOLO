"""
策略迭代改造与多状态评估分析 — 实验 1

功能（对应 PPT 4 个任务）:
  [Task 1] 策略迭代改造与实现
      - 独立函数 policy_iteration(env) 替代 class 封装
      - 运行得到最优策略 π*, print_policy() 展示
      - 收敛过程可视化
  [Task 2] 不同起始状态评估
      - 对状态 0、10、14 分别计算 mean_return
      - 比较分析哪个状态回报最高及原因
  [Task 3] 环境 transition 改造
      - 让冰洞 7 变安全 → 重新跑策略迭代
      - 对比改造前后策略与成功率
  [Task 4] Q 函数可视化与对比
      - 计算 Q 表，打印 Q 值
      - 对比 Human_Agent vs π* 的 Q 值差异分析
  [Bonus] 学生自己实现 policy_iteration() 占位

用法:
  python Policy_Iteration_Analysis.py

输出:
  figures/ 目录下所有 PNG 图表（用于实验报告）
"""

import os
import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from FrozenLake import FrozenLake, test_game, print_policy


FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

nS = 16
nA = 4
terminal_states = {5, 7, 11, 12, 15}
action_labels = ['\u2190 LEFT', '\u2193 DOWN', '\u2192 RIGHT', '\u2191 UP']
arrows = ['\u2190', '\u2193', '\u2192', '\u2191']
terminal_labels = {5: 'H', 7: 'H', 11: 'H', 12: 'H', 15: 'G'}


# ====================================================================
#  Task 1: 策略迭代改造与实现（函数式，无 class）
# ====================================================================

def _build_model(transition):
    P = {}
    for s in range(nS):
        P[s] = {}
        for a in range(nA):
            P[s][a] = transition[s][a]
    return P


def policy_evaluation(transition, policy, V, gamma=0.9, theta=1e-6):
    P = _build_model(transition)
    while True:
        delta = 0
        for s in range(nS):
            if s in terminal_states:
                continue
            v_old = V[s]
            v_new = 0
            for a in range(nA):
                action_prob = policy[s][a]
                if action_prob == 0:
                    continue
                for prob, s_next, reward, done in P[s][a]:
                    v_new += action_prob * prob * (reward + gamma * V[s_next] * (not done))
            V[s] = v_new
            delta = max(delta, abs(v_old - V[s]))
        if delta < theta:
            break
    return V


def policy_improvement(transition, V, policy, gamma=0.9):
    P = _build_model(transition)
    policy_stable = True
    new_policy = np.zeros([nS, nA])
    for s in range(nS):
        if s in terminal_states:
            new_policy[s] = policy[s]
            continue
        q_values = np.zeros(nA)
        for a in range(nA):
            for prob, s_next, reward, done in P[s][a]:
                q_values[a] += prob * (reward + gamma * V[s_next] * (not done))
        best_action = int(np.argmax(q_values))
        new_policy[s][best_action] = 1.0
        if not np.array_equal(new_policy[s], policy[s]):
            policy_stable = False
    return policy_stable, new_policy


def policy_iteration(env, gamma=0.9, theta=1e-6, max_iterations=1000):
    V = np.zeros(nS)
    policy = np.ones([nS, nA]) / nA
    for i in range(max_iterations):
        V = policy_evaluation(env.transition, policy, V, gamma, theta)
        policy_stable, policy = policy_improvement(env.transition, V, policy, gamma)
        if policy_stable:
            return V, policy, i + 1
    return V, policy, max_iterations


def extract_deterministic_policy(policy):
    return {s: int(np.argmax(policy[s])) for s in range(nS)}


def show_values(V):
    print("\n\u72b6\u6001\u4ef7\u503c\u51fd\u6570 V(s):")
    print("-" * 35)
    for s in range(nS):
        if s in terminal_states:
            label = " H" if s != 15 else " G"
            print(f"| {label}  {V[s]:7.2f} ", end="")
        else:
            print(f"| {str(s).zfill(2)}  {V[s]:7.2f} ", end="")
        if (s + 1) % 4 == 0:
            print("|")
    print("-" * 35)


def plot_convergence():
    """Task 1: 策略迭代收敛过程 + 展示最优策略"""
    env = FrozenLake()

    V_track = [np.zeros(nS)]
    V_curr = np.zeros(nS)
    pi = np.ones([nS, nA]) / nA
    for _ in range(100):
        V_curr = policy_evaluation(env.transition, pi, V_curr)
        stable, pi = policy_improvement(env.transition, V_curr, pi)
        V_track.append(V_curr.copy())
        if stable:
            break
    V_track = np.array(V_track)
    n_iter = V_track.shape[0] - 1
    V_final = V_track[-1]
    policy_dict = extract_deterministic_policy(pi)

    print("=" * 60)
    print("Task 1: \u7b56\u7565\u8fed\u4ee3\u6539\u9020\u4e0e\u5b9e\u73b0")
    print("=" * 60)
    print(f"\u6536\u655b\u8fed\u4ee3\u6b21\u6570: {n_iter}")

    show_values(V_final)

    print("\n\u6700\u4f18\u7b56\u7565 \u03c0*:")
    print_policy(lambda s: policy_dict[s])

    rate = test_game(env, lambda s: policy_dict[s], n_episodes=1000)
    print(f"\u6700\u4f18\u7b56\u7565\u6210\u529f\u7387: {rate:.2%}")

    fig, ax = plt.subplots(figsize=(10, 5))
    repr_states = [0, 1, 6, 10, 14, 15]
    state_labels = ['S(\u8d77\u70b9)', '1', '6', '10', '14', 'G(\u76ee\u6807)']
    for s, label in zip(repr_states, state_labels):
        values = V_track[:, s]
        ax.plot(range(len(values)), values, label=f'State {s} {label}',
                linewidth=1.5, marker='o', markersize=3, markevery=max(1, len(values)//10))
    ax.set_xlabel('Iteration')
    ax.set_ylabel('V(s)')
    ax.set_title(f'Policy Iteration Convergence (\u03b3=0.9, {n_iter} iters)')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, '01_convergence.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {path}")

    return policy_dict, V_final


# ====================================================================
#  Task 2: 不同起始状态评估
# ====================================================================

def mean_return(env, pi_dict, state0, n_episodes=5000, max_steps=100):
    results = []
    for _ in range(n_episodes):
        env.position = state0
        s = state0
        total_reward = 0.0
        for _ in range(max_steps):
            action = pi_dict[s]
            s, reward, done = env.step(action)
            total_reward += reward
            if done:
                break
        results.append(total_reward)
    return float(np.mean(results))


def plot_state_comparison(policy_dict):
    """Task 2: \u5bf9\u72b6\u6001 0\u300110\u300114 \u8ba1\u7b97 mean_return"""
    print("\n" + "=" * 60)
    print("Task 2: \u4e0d\u540c\u8d77\u59cb\u72b6\u6001\u8bc4\u4f30")
    print("=" * 60)

    env = FrozenLake()
    target_states = [0, 10, 14]
    returns = []

    for s in target_states:
        r = mean_return(env, policy_dict, s, n_episodes=5000)
        returns.append(r)
        print(f"  State {s:2d}: mean_return = {r:.4f}")

    best_idx = int(np.argmax(returns))
    worst_idx = int(np.argmin(returns))
    print(f"\n\u5206\u6790: \u72b6\u6001 {target_states[best_idx]} \u7684 mean_return \u6700\u9ad8 ({returns[best_idx]:.4f})\uff0c"
          f"\u72b6\u6001 {target_states[worst_idx]} \u6700\u4f4e ({returns[worst_idx]:.4f})")
    print(f"\u539f\u56e0: \u72b6\u6001 0\uff08\u8d77\u70b9\uff09\u8ddd\u79bb\u76ee\u6807\u6700\u8fdc\u4e14\u51b0\u6d1e\u591a\uff1b"
          f"\u72b6\u6001 14 \u7d27\u90bb\u7ec8\u70b9\uff08state 15\uff09\u4e14\u4e00\u6b65\u5230\u8fbe\u6982\u7387\u9ad8\uff0c\u56e0\u6b64\u56de\u62a5\u6700\u9ad8\u3002")

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#E74C3C', '#F39C12', '#2ECC71']
    bars = ax.bar([str(s) for s in target_states], returns, color=colors, alpha=0.8,
                  edgecolor='white', linewidth=1.5, width=0.5)
    for bar, val in zip(bars, returns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_xlabel('Starting State')
    ax.set_ylabel('Mean Return')
    ax.set_title('Task 2: Mean Return Comparison (States 0, 10, 14)')
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    path = os.path.join(FIG_DIR, '02_state_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {path}")

    return returns


# ====================================================================
#  Task 3: 环境 transition 改造
# ====================================================================

def modify_transition_hole7_safe(original_transition):
    modified = copy.deepcopy(original_transition)
    modified[7] = {
        0: [(1/3, 3, 0.0, False), (1/3, 6, 0.0, False), (1/3, 11, 0.0, False)],
        1: [(1/3, 3, 0.0, False), (1/3, 11, 0.0, False), (1/3, 6, 0.0, False)],
        2: [(1/3, 6, 0.0, False), (1/3, 11, 0.0, False), (1/3, 3, 0.0, False)],
        3: [(1/3, 11, 0.0, False), (1/3, 3, 0.0, False), (1/3, 6, 0.0, False)],
    }
    return modified


class ModifiedFrozenLake(FrozenLake):
    def set_tran(self):
        super().set_tran()
        self.transition = modify_transition_hole7_safe(self.transition)


def plot_policy_before_after():
    """Task 3: \u6539\u9020\u524d\u540e\u7b56\u7565\u5bf9\u6bd4"""
    env_orig = FrozenLake()
    env_mod = ModifiedFrozenLake()

    V_orig, _, n_orig = policy_iteration(env_orig)
    dict_orig = extract_deterministic_policy(
        policy_improvement(env_orig.transition, V_orig, np.ones([nS, nA]) / nA)[1])
    rate_orig = test_game(env_orig, lambda s: dict_orig[s], n_episodes=2000)

    V_mod, _, n_mod = policy_iteration(env_mod)
    dict_mod = extract_deterministic_policy(
        policy_improvement(env_mod.transition, V_mod, np.ones([nS, nA]) / nA)[1])
    rate_mod = test_game(env_mod, lambda s: dict_mod[s], n_episodes=2000)

    print("\n" + "=" * 60)
    print("Task 3: \u73af\u5883 transition \u6539\u9020 \u2014 \u51b0\u6d1e 7 \u53d8\u5b89\u5168")
    print("=" * 60)
    print(f"  \u6539\u9020\u524d: \u8fed\u4ee3 {n_orig} \u6b21, \u6210\u529f\u7387 {rate_orig:.2%}")
    print(f"  \u6539\u9020\u540e: \u8fed\u4ee3 {n_mod} \u6b21, \u6210\u529f\u7387 {rate_mod:.2%}")
    changed = dict_orig != dict_mod
    yes_str = '\u662f'
    no_str = '\u5426'
    print(f"  \u7b56\u7565\u662f\u5426\u6539\u53d8: {yes_str if changed else no_str}")
    print(f"  \u6210\u529f\u7387\u53d8\u5316: {rate_mod - rate_orig:+.2%}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    policies_data = [
        ('Before: Original FrozenLake', dict_orig, '#2980B9'),
        ('After: Hole 7 Safe', dict_mod, '#E74C3C'),
    ]
    for ax, (name, p_dict, color) in zip(axes, policies_data):
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(-0.5, 3.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.set_title(name, fontsize=11, fontweight='bold', color=color)
        for s in range(nS):
            r, c = s // 4, s % 4
            if s in terminal_labels:
                label = 'G' if s == 15 else 'H'
                ax.text(c, r, label, ha='center', va='center', fontsize=22, fontweight='bold', color='#333')
            else:
                ax.text(c, r, arrows[p_dict[s]], ha='center', va='center', fontsize=26, color=color)
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor='#CCCCCC', linewidth=1))
        for s in {5, 11, 12}:
            r, c = s // 4, s % 4
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=True, color='#FFE0E0', zorder=-1))
        r, c = 7 // 4, 7 % 4
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=True,
                                    color='#E0FFE0' if 'After' in name else '#FFE0E0', zorder=-1))
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                    edgecolor='red', linewidth=3, linestyle='--', zorder=2))
        ax.add_patch(plt.Rectangle((3.5, 3.5), 1, 1, fill=True, color='#E0FFE0', zorder=-1))

    plt.tight_layout()
    path = os.path.join(FIG_DIR, '03_policy_before_after.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {path}")

    return dict_orig, dict_mod, rate_orig, rate_mod


# ====================================================================
#  Task 4: Q 函数可视化与对比
# ====================================================================

def compute_q_table(env, V, gamma=0.9):
    P = _build_model(env.transition)
    Q = np.zeros((nS, nA))
    for s in range(nS):
        if s in terminal_states:
            continue
        for a in range(nA):
            for prob, s_next, reward, done in P[s][a]:
                Q[s][a] += prob * (reward + gamma * V[s_next] * (not done))
    return Q


def print_q_table(Q, title=""):
    print(f"\nQ(s,a) \u8868 {title}:")
    print(f"{'State':<8}", end="")
    for a in range(nA):
        print(f"{action_labels[a]:<12}", end="")
    print()
    print("-" * 60)
    for s in range(nS):
        label = f"{s}" if s not in terminal_states else (f"{s}(G)" if s == 15 else f"{s}(H)")
        print(f"{label:<8}", end="")
        for a in range(nA):
            print(f"{Q[s][a]:<12.4f}", end="")
        print()


def plot_q_comparison():
    """Task 4: \u5bf9\u6bd4 Human_Agent vs \u03c0* \u7684 Q \u503c\u5dee\u5f02"""
    env = FrozenLake()

    policy_human = np.zeros([nS, nA])
    human_dict = {
        0: 2, 1: 2, 2: 1, 3: 0,
        4: 1, 5: 0, 6: 1, 7: 0,
        8: 2, 9: 2, 10: 1, 11: 0,
        12: 0, 13: 2, 14: 2, 15: 0,
    }
    for s in range(nS):
        if s not in terminal_states:
            policy_human[s][human_dict[s]] = 1.0

    V_opt, pi_opt, _ = policy_iteration(env)
    opt_dict = extract_deterministic_policy(pi_opt)
    Q_opt = compute_q_table(env, V_opt)

    V_human = np.zeros(nS)
    V_human = policy_evaluation(env.transition, policy_human, V_human)
    Q_human = compute_q_table(env, V_human)

    print("\n" + "=" * 60)
    print("Task 4: Q \u51fd\u6570\u53ef\u89c6\u5316\u4e0e\u5bf9\u6bd4")
    print("=" * 60)

    print_q_table(Q_opt, "\u2014 Optimal \u03c0*")
    print_q_table(Q_human, "\u2014 Human Agent")

    diff = np.abs(Q_opt - Q_human)
    diff_per_state = np.mean(diff, axis=1)
    top_diff_states = np.argsort(diff_per_state)[::-1][:5]
    print(f"\n\u5dee\u5f02\u6700\u5927\u7684\u72b6\u6001: {top_diff_states.tolist()}")
    for s in top_diff_states:
        if s not in terminal_states:
            print(f"  State {s}: Human={human_dict[s]}({arrows[human_dict[s]]}) vs Optimal={opt_dict[s]}({arrows[opt_dict[s]]})")
            print(f"    Q_human = {Q_human[s]}")
            print(f"    Q_opt   = {Q_opt[s]}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    vmin = min(Q_opt.min(), Q_human.min())
    vmax = max(Q_opt.max(), Q_human.max())

    im = axes[0].imshow(Q_opt.T, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    axes[0].set_yticks(range(nA))
    axes[0].set_yticklabels(action_labels, fontsize=9)
    axes[0].set_xticks(range(nS))
    axes[0].set_xticklabels(range(nS), fontsize=8)
    axes[0].set_title('Q(s,a) \u2014 Optimal \u03c0*')
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    im = axes[1].imshow(Q_human.T, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    axes[1].set_yticks(range(nA))
    axes[1].set_yticklabels(action_labels, fontsize=9)
    axes[1].set_xticks(range(nS))
    axes[1].set_xticklabels(range(nS), fontsize=8)
    axes[1].set_title('Q(s,a) \u2014 Human Agent')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    im = axes[2].imshow(diff.T, cmap='hot', aspect='auto')
    axes[2].set_yticks(range(nA))
    axes[2].set_yticklabels(action_labels, fontsize=9)
    axes[2].set_xticks(range(nS))
    axes[2].set_xticklabels(range(nS), fontsize=8)
    axes[2].set_title('|Q_opt - Q_human| (Difference)')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    for s in terminal_states:
        for ax in axes:
            ax.add_patch(plt.Rectangle((s - 0.5, -0.5), 1, nA, fill=False, edgecolor='red', linewidth=1.5, linestyle='--'))

    plt.tight_layout()
    path = os.path.join(FIG_DIR, '04_q_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  [OK] {path}")

    return Q_opt, Q_human


# ====================================================================
#  Bonus: 学生自己实现
# ====================================================================

def my_policy_iteration(env, gamma=0.9):
    """
    Bonus: \u5b66\u751f\u81ea\u5df1\u5b9e\u73b0\u7684 policy_iteration
    \u53ef\u53c2\u8003\u4e0a\u9762\u7684 policy_iteration()\uff0c\u6539\u7528 for-loop / lambda \u7b49\u4e0d\u540c\u5199\u6cd5
    """
    raise NotImplementedError("\u8bf7\u5b9e\u73b0\u81ea\u5df1\u7684 policy_iteration()")


# ====================================================================
#  \u4e3b\u7a0b\u5e8f
# ====================================================================

def run_all():
    print("=" * 65)
    print("  \u5b9e\u9a8c 1\uff1a\u7b56\u7565\u8fed\u4ee3\u6539\u9020\u4e0e\u591a\u72b6\u6001\u8bc4\u4f30\u5206\u6790")
    print("  Policy Iteration: Improvement & Multi-State Evaluation")
    print("=" * 65)

    print("\n" + "\u2588" * 60)
    print("  Task 1: \u7b56\u7565\u8fed\u4ee3\u6539\u9020\u4e0e\u5b9e\u73b0")
    print("\u2588" * 60)
    policy_dict, V_opt = plot_convergence()

    print("\n" + "\u2588" * 60)
    print("  Task 2: \u4e0d\u540c\u8d77\u59cb\u72b6\u6001\u8bc4\u4f30")
    print("\u2588" * 60)
    returns = plot_state_comparison(policy_dict)

    print("\n" + "\u2588" * 60)
    print("  Task 3: \u73af\u5883 transition \u6539\u9020")
    print("\u2588" * 60)
    dict_before, dict_after, rate_before, rate_after = plot_policy_before_after()

    print("\n" + "\u2588" * 60)
    print("  Task 4: Q \u51fd\u6570\u53ef\u89c6\u5316\u4e0e\u5bf9\u6bd4")
    print("\u2588" * 60)
    Q_opt, Q_human = plot_q_comparison()

    print("\n" + "=" * 65)
    print(f"  \u6240\u6709\u56fe\u8868\u5df2\u4fdd\u5b58\u81f3: {FIG_DIR}/")
    print("=" * 65)
    print("\n\u751f\u6210\u7684\u6587\u4ef6:")
    for f in sorted(os.listdir(FIG_DIR)):
        fpath = os.path.join(FIG_DIR, f)
        print(f"  {f:45s}  {os.path.getsize(fpath)/1024:.1f} KB")

    return {'policy': policy_dict, 'V': V_opt, 'returns': returns}


if __name__ == '__main__':
    run_all()
