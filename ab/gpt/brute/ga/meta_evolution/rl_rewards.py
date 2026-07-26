import math

def calculate_meta_reward(current_score, best_ever_score, baseline_score, top3_mean, archive_novelty, valid_syntax):
    if not valid_syntax:
        print("   [RL] Penalty: Invalid Syntax/Crash.")
        return -1.0

    reward = 0.0

    # [MODIFIED_FOR_INNOVATION] Rewrote reward logic to remove baseline regression penalty 
    # and SOTA failure penalty. Priority reordered: Novelty (Primary) > SOTA (Secondary) > Density.

    # 1. PRIORITY 1: Behavioral Novelty (MAP-Elites Innovation)
    if archive_novelty > 0:
        # Heavily reward finding new, unique architectures
        novelty_bonus = archive_novelty * 1.5
        reward += novelty_bonus
        print(f"   [RL] PRIMARY (NOVELTY): Archive Novelty ({archive_novelty} cells updated). Bonus: {novelty_bonus:.4f}")

    # 2. PRIORITY 2: Delta SOTA (Frontier Expansion)
    delta_sota = current_score - best_ever_score
    if delta_sota > 0:
        sota_bonus = math.log1p(delta_sota) * 5.0
        reward += sota_bonus
        print(f"   [RL] SECONDARY (SOTA): +{delta_sota:.2f}% SOTA Improvement! Bonus: {sota_bonus:.4f}")
    
    # 3. PRIORITY 3: Quality Density (Top-3 Mean)
    if top3_mean > 0:
        density_reward = (top3_mean / 100.0) * 1.5 
        reward += density_reward
        print(f"   [RL] TERTIARY (DENSITY): Top-3 Quality ({top3_mean:.2f}%). Bonus: {density_reward:.4f}")

    # (Removed baseline regression penalty and failed SOTA penalty to encourage risk-taking)

    reward = min(reward, 15.0) 
    print(f"   [RL] Total Reward: {reward:.4f}")
    return reward