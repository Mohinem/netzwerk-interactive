import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import utils
import os
import sys
import time
import numpy as np

# Add parent directory to path to import envs
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.wordle_env import WordleEnv, WordleActionWrapper

def train_dqn_with_rendering():
    """Train DQN agent with visible human rendering"""
    
    # Create single environment with human rendering
    base_env = WordleEnv(render_mode="human")
    env = WordleActionWrapper(base_env)
    env = Monitor(env, "./logs/wordle_dqn")
    
    # Create DQN model
    model = DQN(
        "MultiInputPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=50000,
        learning_starts=500,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=500,
        exploration_fraction=0.4,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.1,
        verbose=1
    )

    # Add this before your training loop starts (after creating the model):
    if not hasattr(model, '_logger') or model._logger is None:
        
        model._logger = utils.configure_logger(model.verbose, model.tensorboard_log, "manual_train", False)    
    
    print("Starting DQN training with human rendering...")
    print("You'll see every game the agent plays!")
    
    # Training loop
    total_timesteps = 30000
    timestep = 0
    episode = 0
    
    obs, info = env.reset()
    print(f"\n=== Episode {episode + 1} ===")
    print(f"Target: {info['target_word']}")
    
    try:
        while timestep < total_timesteps:
            # Get valid actions
            valid_actions = env.env.get_valid_actions()
            
            if len(valid_actions) == 0:
                obs, info = env.reset()
                episode += 1
                print(f"\n=== Episode {episode + 1} ===")
                print(f"Target: {info['target_word']}")
                continue
            
            # Get action from model
            if timestep < model.learning_starts or np.random.random() < model.exploration_rate:
                valid_word = np.random.choice(valid_actions)
                action = env.reverse_action(valid_word)
            else:
                action, _ = model.predict(obs, deterministic=False)
                chosen_word = env.action(action)
                if chosen_word not in valid_actions:
                    valid_word = np.random.choice(valid_actions)
                    action = env.reverse_action(valid_word)
            
            guess_word = env.action(action)
            print(f"Step {timestep}: Agent guesses '{guess_word}'")
            
            # Take step
            new_obs, reward, terminated, truncated, info = env.step(action)

            # Add transition to replay buffer (with proper variable order)
            # model.replay_buffer.add(
            #     obs,                      # Previous observation (dict)
            #     action,                   # Action taken (integer)
            #     reward,                   # Reward received (float)
            #     new_obs,                  # New observation (dict)
            #     terminated or truncated,  # Done flag (boolean)
            #     [{}]                     # Info (list of dicts)
            # )
            
            # Render and show result
            env.render()
            print(f"Reward: {reward}")
            
            # time.sleep(0.5)  # Pause to see result
            timestep += 1
            
            # **KEY FIX**: Only train when environment is NOT done
            if (timestep > model.learning_starts and 
                timestep % model.train_freq.frequency == 0 and 
                not terminated and not truncated and
                model.replay_buffer.size() >= model.batch_size):  # ← Add this check
                
                    model._update_current_progress_remaining(timestep, total_timesteps)
                    model.train(gradient_steps=1, batch_size=model.batch_size)
                        
            if terminated or truncated:
                episode += 1
                won = info.get('won', False)
                guesses = info.get('guess_count', 0)
                
                print(f"\nEpisode {episode}: {'🎉 WON' if won else '😞 LOST'} in {guesses} guesses")
                print("-" * 50)
                
                # time.sleep(1.5)  # Pause between episodes
                
                # Reset for next episode
                obs, info = env.reset()
                print(f"\n=== Episode {episode + 1} ===")
                print(f"Target: {info['target_word']}")
            else:
                obs = new_obs
        
    except KeyboardInterrupt:
        print(f"\nTraining interrupted at timestep {timestep}")
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model.save("models/wordle_dqn_visual")
    print(f"\nTraining complete! Model saved after {episode} episodes")
    
    return model


if __name__ == "__main__":
    # Create log directory
    os.makedirs("logs", exist_ok=True)
    
    # Train the model
    model = train_dqn_with_rendering()
    
    # Test the trained model
    print("\n" + "="*50)
    print("TESTING TRAINED MODEL")
    print("="*50)
    
    test_env = WordleActionWrapper(WordleEnv(render_mode='human'))
    
    for episode in range(3):
        obs, info = test_env.reset()
        print(f"\n=== Test Episode {episode + 1} ===")
        print(f"Target: {info['target_word']}")
        
        for step in range(6):
            action, _ = model.predict(obs, deterministic=True)
            
            # Convert numpy array to int if needed
            if isinstance(action, np.ndarray):
                action = int(action.item())
            
            # ✅ FIX: Ensure we don't repeat guesses
            guess_word = test_env.action(action)
            
            # Check if this word was already guessed
            if guess_word in obs.get('guessed_words', []):
                # Find a valid alternative
                valid_actions = test_env.env.get_valid_actions()
                if valid_actions:
                    guess_word = valid_actions[0]  # Pick first available
                    action = test_env.reverse_action(guess_word)
                else:
                    print("No more valid actions available!")
                    break
            
            print(f"Trained agent guesses: '{guess_word}'")
            
            obs, reward, terminated, truncated, info = test_env.step(action)
            test_env.render()
            
            if terminated:
                result = "WON" if info.get('won', False) else "LOST"
                print(f"Result: {result}")
                break
            
            time.sleep(1.0)
        
        time.sleep(2.0)
    
    test_env.close()
    print("\nDone!")
