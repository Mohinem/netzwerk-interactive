"""Shared utilities for testing environments"""

import numpy as np
from typing import Dict, Any

def validate_observation_space(obs: Dict[str, Any], expected_keys: list) -> bool:
    """Validate that observation contains expected keys and proper types"""
    for key in expected_keys:
        if key not in obs:
            return False
    return True

def run_random_episode(env, max_steps: int = 10) -> Dict[str, Any]:
    """Run a random episode and return statistics"""
    obs, info = env.reset()
    total_reward = 0
    steps = 0
    
    while steps < max_steps:
        # Get valid actions
        valid_actions = [i for i, mask in enumerate(obs.get('action_mask', [])) if mask == 1]
        if not valid_actions:
            break
            
        action = np.random.choice(valid_actions)
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        steps += 1
        
        if terminated or truncated:
            break
    
    return {
        'total_reward': total_reward,
        'steps': steps,
        'won': info.get('won', False),
        'target_word': info.get('target_word', 'UNKNOWN')
    }
