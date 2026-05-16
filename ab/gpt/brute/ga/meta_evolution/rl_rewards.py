import math

def calculate_meta_reward(current_score, baseline_score, valid_syntax):
    # 1. Syntax/Crash Penalty
    if not valid_syntax:
        print("   [RL] Penalty: Invalid Syntax/Crash.")
        return -1.0

    # 2. Define Significance Threshold
    delta = current_score - baseline_score
    NOISE_THRESHOLD = 0.5  # Must improve by at least 0.5% to prove it wasn't just luck

    # 3. Improvement Reward (Only if it beats the noise threshold)
    if delta > NOISE_THRESHOLD:
        # Base reward ensures gradients don't vanish
        base_reward = 1.0 
        
        # Difficulty Multiplier (Harder to improve at 90% than at 50%)
        room_for_improvement = max(1.0, 100.0 - baseline_score)
        difficulty_multiplier = 10.0 / room_for_improvement
        
        # Logarithmic soft-cap: Replaces `min(5.0)`. 
        # A delta of +5 yields ~2.6 bonus. A delta of +20 yields ~4.0 bonus.
        bonus = math.log1p(delta) * difficulty_multiplier
        
        reward = base_reward + bonus
        
        # Absolute safety cap for extreme edge cases to protect LLM gradients
        reward = min(reward, 10.0) 
        print(f"   [RL] TRUE SUCCESS: +{delta:.2f}% improvement. Reward: {reward:.4f}")
        return reward

    # 4. Stagnation / Noise Band
    elif abs(delta) <= NOISE_THRESHOLD:
        print(f"   [RL] Stagnation/Noise (Delta: {delta:.2f}%). Reward: 0.0")
        return 0.0

    # 5. Regression Penalty
    else:
        # Soft penalty for regression to discourage bad code, but not as harsh as a crash
        reward = max(delta * 0.1, -0.5)
        print(f"   [RL] Regression (Delta: {delta:.2f}%). Reward: {reward:.4f}")
        return reward