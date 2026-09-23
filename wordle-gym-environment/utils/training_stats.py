import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import json
import time

class WordleTrainingStats:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.reset_stats()
    
    def reset_stats(self):
        # Episode-level metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_wins = []
        self.episode_win_attempts = []  # On which attempt did wins occur
        
        # Rolling windows for smooth tracking
        self.reward_window = deque(maxlen=self.window_size)
        self.win_rate_window = deque(maxlen=self.window_size)
        
        # Detailed statistics
        self.total_episodes = 0
        self.total_wins = 0
        self.wins_by_attempt = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        
        # Training progress
        self.training_start_time = time.time()
        
    def record_episode(self, reward, length, won, win_attempt=None):
        """Record statistics for a completed episode"""
        self.total_episodes += 1
        
        # Store raw data
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_wins.append(won)
        
        # Update rolling windows
        self.reward_window.append(reward)
        self.win_rate_window.append(1 if won else 0)
        
        # Track wins
        if won:
            self.total_wins += 1
            if win_attempt and 1 <= win_attempt <= 6:
                self.wins_by_attempt[win_attempt] += 1
                self.episode_win_attempts.append(win_attempt)
    
    def get_current_stats(self):
        """Get current training statistics"""
        if self.total_episodes == 0:
            return {}
        
        # Calculate metrics
        overall_win_rate = self.total_wins / self.total_episodes
        recent_win_rate = np.mean(self.win_rate_window) if self.win_rate_window else 0
        recent_avg_reward = np.mean(self.reward_window) if self.reward_window else 0
        
        # Win distribution
        win_distribution = {}
        for attempt in range(1, 7):
            win_distribution[f'wins_attempt_{attempt}'] = self.wins_by_attempt[attempt]
        
        # Average win attempt (for successful games only)
        avg_win_attempt = np.mean(self.episode_win_attempts) if self.episode_win_attempts else 0
        
        return {
            'total_episodes': self.total_episodes,
            'total_wins': self.total_wins,
            'overall_win_rate': overall_win_rate,
            'recent_win_rate': recent_win_rate,
            'recent_avg_reward': recent_avg_reward,
            'avg_win_attempt': avg_win_attempt,
            'training_time': time.time() - self.training_start_time,
            **win_distribution
        }
    
    def print_progress(self, episode_interval=500):
        """Print training progress"""
        if self.total_episodes % episode_interval == 0:
            stats = self.get_current_stats()
            print(f"\n📊 Training Progress - Episode {self.total_episodes}")
            print(f"Overall Win Rate: {stats['overall_win_rate']:.3f} ({stats['total_wins']}/{stats['total_episodes']})")
            print(f"Recent Win Rate: {stats['recent_win_rate']:.3f} (last {min(len(self.win_rate_window), self.window_size)} episodes)")
            print(f"Recent Avg Reward: {stats['recent_avg_reward']:.2f}")
            print(f"Avg Win Attempt: {stats['avg_win_attempt']:.2f}")
            # ✅ FIXED: No more red squiggly
            win_dist = [f"{i}:{stats[f'wins_attempt_{i}']}" for i in range(1,7)]
            print(f"Win Distribution: {win_dist}")
            print(f"Training Time: {stats['training_time']/60:.1f} minutes")
            print("-" * 60)
    
    def save_stats(self, filename):
        """Save statistics to file"""
        stats = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_wins': self.episode_wins,
            'episode_win_attempts': self.episode_win_attempts,
            'current_stats': self.get_current_stats()
        }
        
        with open(filename, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def plot_training_curves(self, save_path=None):
        """Plot training progress curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Win rate over time
        window = 100
        if len(self.episode_wins) >= window:
            win_rates = [np.mean(self.episode_wins[max(0, i-window):i+1]) 
                        for i in range(len(self.episode_wins))]
            axes[0,0].plot(win_rates)
            axes[0,0].set_title('Win Rate (Rolling Average)')
            axes[0,0].set_xlabel('Episode')
            axes[0,0].set_ylabel('Win Rate')
            axes[0,0].grid(True)
        
        # Reward over time
        if len(self.episode_rewards) >= window:
            reward_smooth = [np.mean(self.episode_rewards[max(0, i-window):i+1]) 
                           for i in range(len(self.episode_rewards))]
            axes[0,1].plot(reward_smooth)
            axes[0,1].set_title('Average Reward (Rolling)')
            axes[0,1].set_xlabel('Episode')
            axes[0,1].set_ylabel('Reward')
            axes[0,1].grid(True)
        
        # Win attempt distribution
        attempts = list(self.wins_by_attempt.keys())
        counts = list(self.wins_by_attempt.values())
        axes[1,0].bar(attempts, counts)
        axes[1,0].set_title('Wins by Attempt Number')
        axes[1,0].set_xlabel('Attempt Number')
        axes[1,0].set_ylabel('Number of Wins')
        axes[1,0].grid(True)
        
        # Cumulative wins over time
        cumulative_wins = np.cumsum(self.episode_wins)
        axes[1,1].plot(cumulative_wins)
        axes[1,1].set_title('Cumulative Wins')
        axes[1,1].set_xlabel('Episode')
        axes[1,1].set_ylabel('Total Wins')
        axes[1,1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
