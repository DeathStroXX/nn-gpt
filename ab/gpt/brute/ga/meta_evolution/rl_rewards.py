import math

def calculate_meta_reward(current_score, best_ever_score, top3_mean, archive_novelty, valid_syntax):
    """
    Calculates the meta-reward based on MAP-Elites and SOTA-frontier expansion principles.
    Instead of rewarding the LLM for producing a "good model in the abstract", we reward it for:
    1. Expanding the global SOTA frontier (Primary)
    2. Maintaining high overall quality in the generation (Secondary)
    3. Discovering novel architectures in the MAP-Elites archive (Tertiary)
    """
    if not valid_syntax:
        print("   [RL] Penalty: Invalid Syntax/Crash.")
        return -1.0

    reward = 0.0

    # 1. Primary Reward: Delta SOTA (Frontier Expansion)
    # Did this generation produce an all-time SOTA?
    delta_sota = current_score - best_ever_score
    if delta_sota > 0:
        # Shaped reward using logarithmic scaling to avoid extreme sparsity but prevent explosion
        sota_bonus = math.log1p(delta_sota) * 5.0
        reward += sota_bonus
        print(f"   [RL] PRIMARY: +{delta_sota:.2f}% SOTA Improvement! Bonus: {sota_bonus:.4f}")
    
    # 2. Secondary Reward: Quality Density (Top-3 Mean)
    # This provides a dense gradient signal even if SOTA isn't broken
    # It encourages the LLM to propose operators that consistently produce good results
    if top3_mean > 0:
        # Scale the top3 mean so it contributes a small dense signal (e.g., 80% accuracy -> 0.8 reward)
        density_reward = (top3_mean / 100.0) * 1.5 
        reward += density_reward
        print(f"   [RL] SECONDARY: Top-3 Quality ({top3_mean:.2f}%). Bonus: {density_reward:.4f}")

    # 3. Tertiary Reward: Behavioral Novelty (MAP-Elites)
    # Reward the LLM for proposing moves that fill empty archive cells or improve existing ones
    if archive_novelty > 0:
        # 0.5 per novel/improved cell
        novelty_bonus = archive_novelty * 0.5
        reward += novelty_bonus
        print(f"   [RL] TERTIARY: Archive Novelty ({archive_novelty} cells updated). Bonus: {novelty_bonus:.4f}")

    # Absolute safety cap for extreme edge cases to protect LLM gradients
    reward = min(reward, 15.0) 
    print(f"   [RL] Total Reward: {reward:.4f}")
    return reward