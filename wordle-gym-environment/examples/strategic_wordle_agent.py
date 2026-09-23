import os
import sys
import time
import numpy as np

# Add parent directory to path to import envs
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from envs.wordle_env import WordleEnv, WordleActionWrapper
from utils.training_stats import WordleTrainingStats

class StrategyAgent:
    """Pure strategy-based Wordle agent - no neural networks needed!"""
    
    def __init__(self, env):
        self.env = env
        
        # Strategic opening words (high information value)
        self.opening_words = [
            'ARISE', 'SOARE', 'ADIEU', 'AUDIO', 'TARES', 
            'ROATE', 'RAISE', 'SLATE', 'CRATE', 'TRACE'
        ]
        
        # Common letters for tie-breaking
        self.letter_frequency = {
            'E': 11.16, 'A': 8.34, 'R': 7.58, 'I': 7.54, 'O': 7.16,
            'T': 6.95, 'N': 6.65, 'S': 5.74, 'L': 5.49, 'C': 4.54,
            'U': 3.63, 'D': 3.38, 'P': 3.17, 'M': 3.01, 'H': 2.85,
            'G': 2.47, 'B': 2.07, 'F': 1.81, 'Y': 1.78, 'W': 1.29,
            'K': 1.11, 'V': 1.01, 'X': 0.27, 'Z': 0.44, 'J': 0.24, 'Q': 0.20
        }
    
    def predict(self, obs, deterministic=True):
        """Main prediction method - pure strategy, no neural network"""
        
        # First guess: Use proven strategic opener
        if obs['guess_count'][0] == 0:
            available_openers = [word for word in self.opening_words 
                               if word in self.env.word_list]
            if available_openers:
                chosen_word = available_openers[0]  # ARISE or first available
                return self.env.word_list.index(chosen_word)
        
        # Get strategically valid actions based on constraints
        valid_words = self.env.get_valid_strategic_actions()
        
        if not valid_words:
            # Fallback to any valid action
            valid_words = self.env.get_valid_actions()
        
        if len(valid_words) == 1:
            # Only one option - take it!
            return self.env.word_list.index(valid_words[0])
        
        # Multiple options: Use heuristics to pick best
        return self.env.word_list.index(self._choose_best_word(valid_words, obs))
    
    def _choose_best_word(self, valid_words, obs):
        """Choose best word from valid options using heuristics"""
        
        # Strategy 1: If we have strong constraints, pick word with common letters
        constraints = self.env._get_letter_constraints()
        known_letters_count = np.sum(constraints['known_letters'])
        
        if known_letters_count >= 3:
            # We know a lot - prioritize fitting the pattern
            return self._pick_by_pattern_fit(valid_words, constraints)
        
        # Strategy 2: Early game - maximize information gain
        return self._pick_by_information_gain(valid_words, constraints)
    
    def _pick_by_pattern_fit(self, valid_words, constraints):
        """Pick word that best fits known pattern"""
        scored_words = []
        
        for word in valid_words:
            score = 0
            
            # Bonus for using known letters in right positions
            for pos in range(5):
                if constraints['known_positions'][pos] != -1:
                    if self.env._word_to_array(word)[pos] == constraints['known_positions'][pos]:
                        score += 10
            
            # Bonus for common letters
            for letter in word:
                score += self.letter_frequency.get(letter, 0)
            
            scored_words.append((word, score))
        
        # Return highest scoring word
        return max(scored_words, key=lambda x: x[1])[0]
    
    def _pick_by_information_gain(self, valid_words, constraints):
        """Pick word that maximizes information gain"""
        scored_words = []
        
        for word in valid_words:
            score = 0
            letters_in_word = set(word)
            
            # Bonus for using diverse, common letters
            unique_letters = len(letters_in_word)
            score += unique_letters * 5  # Diversity bonus
            
            # Bonus for common letters
            for letter in letters_in_word:
                score += self.letter_frequency.get(letter, 0)
            
            # Penalty for reusing known letters in exploration phase
            if np.sum(constraints['known_letters']) < 2:
                for letter_num in range(26):
                    if constraints['known_letters'][letter_num] == 1:
                        if self.env._num_to_letter(letter_num) in word:
                            score -= 2  # Small penalty for reusing known letters early
            
            scored_words.append((word, score))
        
        return max(scored_words, key=lambda x: x[1])[0]

def test_strategy_agent(num_episodes=100, show_games=True, stats_interval=10):
    """Test the strategy agent performance with enhanced stats display"""
    
    # Create environment
    base_env = WordleEnv(render_mode="human" if show_games else None)
    env = WordleActionWrapper(base_env)
    
    # Create strategy agent (no neural network!)
    agent = StrategyAgent(env.env)
    
    # ✅ Initialize statistics using utils
    stats = WordleTrainingStats(window_size=50)
    
    print("🎯 Testing Pure Strategy Agent (No DQN Training!)")
    print("=" * 60)
    print(f"Running {num_episodes} episodes...")
    print(f"Stats will be shown every {stats_interval} episodes")
    print("=" * 60)
    
    # Test for multiple episodes
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        
        if show_games:
            print(f"\n=== Episode {episode + 1} ===")
            print(f"Target: {info['target_word']}")
        
        for step in range(6):
            # Get action from strategy agent
            action = agent.predict(obs, deterministic=True)
            
            guess_word = env.action(action)
            if show_games:
                print(f"Agent guesses: '{guess_word}'")
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            # Show result
            if show_games:
                env.render()
                print(f"Reward: {reward}")
            
            if terminated:
                won = info.get('won', False)
                guesses = info.get('guess_count', 0)
                
                # ✅ Record statistics using utils
                stats.record_episode(
                    reward=episode_reward,
                    won=won,
                    win_attempt=guesses if won else None,
                    length=step + 1  # Number of guesses made
                )


                if show_games:
                    result = "🎉 WON" if won else "😞 LOST"
                    print(f"\n{result} in {guesses} guesses! Episode Reward: {episode_reward}")
                break
            
            if show_games:
                time.sleep(0.3)  # Pause to see result
        
        # ✅ Show progress using utils stats every N episodes
        if (episode + 1) % stats_interval == 0:
            stats.print_progress(episode_interval=stats_interval)
    
    # ✅ Final results using utils stats
    final_stats = stats.get_current_stats()
    print("\n" + "=" * 60)
    print("🏆 FINAL RESULTS - PURE STRATEGY AGENT")
    print("=" * 60)
    print(f"Episodes Completed: {final_stats['total_episodes']}")
    print(f"Total Wins: {final_stats['total_wins']}")
    print(f"Overall Win Rate: {final_stats['overall_win_rate']:.1%}")
    print(f"Recent Win Rate: {final_stats['recent_win_rate']:.1%}")
    print(f"Average Win Attempt: {final_stats['avg_win_attempt']:.2f}")
    print(f"Average Episode Reward: {final_stats['recent_avg_reward']:.2f}")
    
    # Show win distribution
    win_dist = [f"{i}:{final_stats[f'wins_attempt_{i}']}" for i in range(1,7)]
    print(f"Win Distribution by Attempt: {win_dist}")
    
    # Show time taken
    print(f"Total Time: {final_stats['training_time']:.1f} seconds")
    print(f"Time per Episode: {final_stats['training_time']/final_stats['total_episodes']:.2f} seconds")
    
    env.close()
    return stats

def run_quick_test():
    """Quick test with fewer episodes and full display"""
    print("🚀 Quick Test (10 episodes with full display)")
    return test_strategy_agent(num_episodes=10, show_games=True, stats_interval=5)

def run_full_test():
    """Full test with many episodes, minimal display"""
    print("📊 Full Performance Test (500 episodes, stats only)")
    return test_strategy_agent(num_episodes=500, show_games=False, stats_interval=50)

def run_benchmark():
    """Benchmark against different episode counts"""
    print("🏁 Strategy Agent Benchmark Suite")
    print("=" * 60)
    
    test_sizes = [50, 100, 200, 500]
    results = {}
    
    for size in test_sizes:
        print(f"\n📈 Testing with {size} episodes...")
        stats = test_strategy_agent(num_episodes=size, show_games=False, stats_interval=size//5)
        final_stats = stats.get_current_stats()
        results[size] = {
            'win_rate': final_stats['overall_win_rate'],
            'avg_attempts': final_stats['avg_win_attempt'],
            'avg_reward': final_stats['recent_avg_reward']
        }
        
        print(f"✅ {size} episodes: {final_stats['overall_win_rate']:.1%} win rate")
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("📊 BENCHMARK SUMMARY")
    print("=" * 60)
    for size, result in results.items():
        print(f"{size:3d} episodes: {result['win_rate']:.1%} win rate, "
              f"{result['avg_attempts']:.2f} avg attempts, "
              f"{result['avg_reward']:.1f} avg reward")
    
    return results

if __name__ == "__main__":
    """Run full test with human rendering"""
    
    # Create environment with human rendering
    base_env = WordleEnv(render_mode="human")
    env = WordleActionWrapper(base_env)
    
    # Create strategy agent
    agent = StrategyAgent(env.env)
    
    # Initialize statistics
    stats = WordleTrainingStats(window_size=50)
    
    print("🎯 Testing Pure Strategy Agent (No DQN Training!)")
    print("📊 Full Performance Test (500 episodes) with human rendering")
    print("=" * 60)
    
    num_episodes = 500
    stats_interval = 50
    
    try:
        # Test for 500 episodes with full display
        for episode in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0
            
            print(f"\n=== Episode {episode + 1} ===")
            print(f"Target: {info['target_word']}")
            
            for step in range(6):
                # Get action from strategy agent
                action = agent.predict(obs, deterministic=True)
                
                guess_word = env.action(action)
                print(f"Agent guesses: '{guess_word}'")
                
                # Take step
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                
                # Show result
                env.render()
                print(f"Reward: {reward}")
                
                if terminated:
                    won = info.get('won', False)
                    guesses = info.get('guess_count', 0)
                    
                    # Record statistics
                    stats.record_episode(
                        reward=episode_reward,
                        won=won,
                        win_attempt=guesses if won else None,
                        length=step + 1
                    )
                    
                    result = "🎉 WON" if won else "😞 LOST"
                    print(f"\n{result} in {guesses} guesses! Episode Reward: {episode_reward}")
                    break
                
                time.sleep(0.3)  # Pause to see result
            
            # Show progress every 50 episodes
            if (episode + 1) % stats_interval == 0:
                stats.print_progress(episode_interval=stats_interval)
        
        # Final results
        final_stats = stats.get_current_stats()
        print("\n" + "=" * 60)
        print("🏆 FINAL RESULTS - PURE STRATEGY AGENT")
        print("=" * 60)
        print(f"Episodes Completed: {final_stats['total_episodes']}")
        print(f"Total Wins: {final_stats['total_wins']}")
        print(f"Overall Win Rate: {final_stats['overall_win_rate']:.1%}")
        print(f"Recent Win Rate: {final_stats['recent_win_rate']:.1%}")
        print(f"Average Win Attempt: {final_stats['avg_win_attempt']:.2f}")
        print(f"Average Episode Reward: {final_stats['recent_avg_reward']:.2f}")
        
        # Show win distribution
        win_dist = [f"{i}:{final_stats[f'wins_attempt_{i}']}" for i in range(1,7)]
        print(f"Win Distribution by Attempt: {win_dist}")
        
        # Show time taken
        print(f"Total Time: {final_stats['training_time']:.1f} seconds")
        print(f"Time per Episode: {final_stats['training_time']/final_stats['total_episodes']:.2f} seconds")
        
        # Save results
        stats.save_stats("strategy_agent_results.json")
        print(f"\n💾 Detailed statistics saved to strategy_agent_results.json")
        
        # Generate plots
        try:
            stats.plot_training_curves("strategy_agent_performance.png")
            print("📈 Performance plots saved to strategy_agent_performance.png")
        except Exception as e:
            print(f"📈 Plotting skipped (no display): {e}")
        
        print("\n✅ Strategy agent test complete!")
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        env.close()

