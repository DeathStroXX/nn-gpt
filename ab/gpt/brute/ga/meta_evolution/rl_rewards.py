import math

def calculate_meta_reward(current_score, best_ever_score, baseline_score, top3_mean, archive_novelty, valid_syntax):
    if not valid_syntax:
        print("   [RL] Penalty: Invalid Syntax/Crash.")
        return -1.0

    reward = 0.0
    delta_baseline = current_score - baseline_score

    # 1. Baseline Regression Penalty (Crucial to prevent Model Collapse)
    if delta_baseline < -5.0:
        # Penalize severely bad code so the LLM doesn't learn mediocrity
        penalty = delta_baseline * 0.2
        reward += penalty
        print(f"   [RL] PENALTY: Severe Regression below baseline ({delta_baseline:.2f}%). Penalty: {penalty:.4f}")
        return reward # Immediately return, do not reward density or novelty if it's a severe regression!
    elif delta_baseline < 0:
        # Mild regression (within 5% tolerance). Apply penalty but allow novelty/density to potentially rescue it!
        penalty = delta_baseline * 0.2
        reward += penalty
        print(f"   [RL] PENALTY: Mild Regression ({delta_baseline:.2f}%). Penalty: {penalty:.4f}")

    # 2. Primary Reward: Delta SOTA (Frontier Expansion)
    delta_sota = current_score - best_ever_score
    if delta_sota > 0:
        sota_bonus = math.log1p(delta_sota) * 5.0
        reward += sota_bonus
        print(f"   [RL] PRIMARY: +{delta_sota:.2f}% SOTA Improvement! Bonus: {sota_bonus:.4f}")
    
    # 3. Secondary Reward: Quality Density (Top-3 Mean)
    if top3_mean > 0:
        density_reward = (top3_mean / 100.0) * 1.5 
        reward += density_reward
        print(f"   [RL] SECONDARY: Top-3 Quality ({top3_mean:.2f}%). Bonus: {density_reward:.4f}")

    # 4. Tertiary Reward: Behavioral Novelty (MAP-Elites)
    if archive_novelty > 0:
        novelty_bonus = archive_novelty * 0.5
        reward += novelty_bonus
        print(f"   [RL] TERTIARY: Archive Novelty ({archive_novelty} cells updated). Bonus: {novelty_bonus:.4f}")
    elif archive_novelty == 0 and delta_sota <= 0:
        stagnation_penalty = -2.0
        reward += stagnation_penalty
        print(f"   [RL] PENALTY: Zero Archive Novelty (Stagnation/Premature Convergence). Penalty: {stagnation_penalty:.4f}")

    reward = min(reward, 15.0) 
    print(f"   [RL] Total Reward: {reward:.4f}")
    return reward