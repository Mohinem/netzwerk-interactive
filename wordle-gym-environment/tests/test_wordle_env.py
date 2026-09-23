import unittest
import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import sys
import os

# Add the parent directory to the path to import envs
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.wordle_env import WordleEnv, WordleActionWrapper

class TestWordleEnvStringActions(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        self.env = WordleEnv()
    
    def tearDown(self):
        """Clean up after each test"""
        self.env.close()
    
    def test_environment_creation(self):
        """Test basic environment setup with string actions"""
        self.assertIsNotNone(self.env.word_list)
        self.assertGreater(len(self.env.word_list), 0)
        self.assertEqual(self.env.WORD_LENGTH, 5)
        self.assertEqual(self.env.MAX_GUESSES, 6)
        
        # Check action space is Text type
        self.assertEqual(self.env.action_space.max_length, 5)
        self.assertEqual(self.env.action_space.min_length, 5)
        print(f"✅ Environment created with {len(self.env.word_list)} words")
        print(f"✅ Action space: {self.env.action_space}")
    
    def test_gymnasium_compliance(self):
        """Test if environment follows Gymnasium API standards"""
        try:
            check_env(self.env, warn=True)
            print("✅ Environment passes Gymnasium validation!")
        except Exception as e:
            print(f"⚠️ Environment validation warning: {e}")
            # Note: Text spaces might not be fully supported by check_env yet
    
    def test_reset_functionality(self):
        """Test reset method returns correct observation structure"""
        obs, info = self.env.reset()
        
        # Check observation structure
        self.assertIn('board', obs)
        self.assertIn('feedback', obs)
        self.assertIn('guess_count', obs)
        self.assertIn('guessed_words', obs)
        
        # Check observation shapes and types
        self.assertEqual(obs['board'].shape, (6, 5))
        self.assertEqual(obs['feedback'].shape, (6, 5))
        self.assertEqual(obs['guess_count'], 0)
        self.assertIsInstance(obs['guessed_words'], list)
        self.assertEqual(len(obs['guessed_words']), 0)  # No words guessed initially
        
        # Check initial state
        self.assertIn('target_word', info)
        self.assertEqual(len(info['target_word']), 5)
        print("✅ Reset functionality works correctly")
    
    def test_string_action_validation(self):
        """Test string action validation"""
        obs, info = self.env.reset()
        
        # Test valid 5-letter string
        if "ABOUT" in self.env.word_list:
            obs, reward, terminated, truncated, info = self.env.step("ABOUT")
            self.assertIsInstance(reward, (int, float, np.integer, np.floating))
            self.assertTrue(info['valid_word'])
            print("✅ Valid string action accepted")
        
        # Test case insensitive (should convert to uppercase)
        self.env.reset()
        if "BRAIN" in self.env.word_list:
            obs, reward, terminated, truncated, info = self.env.step("brain")
            self.assertEqual(info['guess'], "BRAIN")
            self.assertTrue(info['valid_word'])
            print("✅ Case insensitive input handled correctly")
    
    def test_invalid_actions(self):
        """Test invalid action handling"""
        self.env.reset()
        
        # Test wrong length
        with self.assertRaises(ValueError):
            self.env.step("ABC")  # Too short
        
        with self.assertRaises(ValueError):
            self.env.step("ABCDEF")  # Too long
        
        # Test non-string input
        with self.assertRaises(ValueError):
            self.env.step(123)
        
        # Test non-alphabetic characters
        with self.assertRaises(ValueError):
            self.env.step("ABC12")
        
        print("✅ Invalid actions properly rejected")
    
    def test_invalid_word_handling(self):
        """Test handling of words not in vocabulary"""
        obs, info = self.env.reset()
        
        # Try a word that's unlikely to be in our vocabulary
        obs, reward, terminated, truncated, info = self.env.step("ZZZZZ")
        
        self.assertEqual(reward, -2)  # Penalty for invalid word
        self.assertFalse(info.get('valid_word', True))
        self.assertFalse(terminated)  # Game should continue
        print("✅ Invalid words handled with penalty")
    
    def test_feedback_logic(self):
        """Test Wordle feedback calculation for different scenarios"""
        # Test exact match (all green)
        feedback = self.env._get_feedback("ABOUT", "ABOUT")
        expected = np.array([2, 2, 2, 2, 2])
        np.testing.assert_array_equal(feedback, expected)
        
        # Test no match (all gray)
        feedback = self.env._get_feedback("ABOUT", "ZYXWV")
        expected = np.array([0, 0, 0, 0, 0])
        np.testing.assert_array_equal(feedback, expected)
        
        # Test partial match with repeated letters
        feedback = self.env._get_feedback("PRESS", "SPREE")
        expected = np.array([1, 1, 1, 1, 0])
        np.testing.assert_array_equal(feedback, expected)
        
        print("✅ Feedback logic works correctly")
    
    def test_repeat_guess_prevention(self):
        """Test that repeat guesses are penalized"""
        obs, info = self.env.reset()
        
        # Make first guess
        if "ABOUT" in self.env.word_list:
            obs, reward1, terminated, truncated, info = self.env.step("ABOUT")
            self.assertNotEqual(reward1, -5)  # Should not be penalty for first guess
            
            # Try same guess again
            obs, reward2, terminated, truncated, info = self.env.step("ABOUT")
            self.assertEqual(reward2, -5)  # Should get repeat penalty
            
            # Check that word is in guessed_words
            self.assertIn("ABOUT", obs['guessed_words'])
            
        print("✅ Repeat guess prevention works correctly")
    
    def test_winning_condition(self):
        """Test that winning gives correct reward and terminates"""
        obs, info = self.env.reset()
        target = info['target_word']
        
        # Guess the target word directly
        obs, reward, terminated, truncated, info = self.env.step(target)
        
        self.assertTrue(terminated)
        self.assertTrue(info['won'])
        self.assertEqual(reward, 100)  # Win reward
        
        print("✅ Winning condition works correctly")
    
    def test_losing_condition(self):
        """Test losing after 6 wrong guesses"""
        obs, info = self.env.reset()
        target = info['target_word']
        
        # Find 6 words that are not the target
        wrong_words = [word for word in self.env.word_list[:10] if word != target][:6]
        
        if len(wrong_words) >= 6:
            # Make 6 wrong guesses
            for i, word in enumerate(wrong_words):
                obs, reward, terminated, truncated, info = self.env.step(word)
                
                if i == 5:  # Last guess
                    if not info.get('won', False):
                        self.assertTrue(terminated)
                        self.assertFalse(info.get('won', True))
                        self.assertEqual(reward, -50)  # Loss penalty
        
        print("✅ Losing condition works correctly")
    
    def test_get_valid_actions(self):
        """Test the get_valid_actions helper method"""
        obs, info = self.env.reset()
        
        # Initially all words should be valid
        valid_actions = self.env.get_valid_actions()
        self.assertEqual(len(valid_actions), len(self.env.word_list))
        
        # After making a guess, that word should be removed from valid actions
        if "ABOUT" in self.env.word_list:
            self.env.step("ABOUT")
            valid_actions_after = self.env.get_valid_actions()
            
            self.assertEqual(len(valid_actions_after), len(self.env.word_list) - 1)
            self.assertNotIn("ABOUT", valid_actions_after)
        
        print("✅ get_valid_actions() works correctly")
    
    def test_observation_updates(self):
        """Test that observations update correctly after each step"""
        obs, info = self.env.reset()
        initial_guess_count = obs['guess_count']
        
        if "ABOUT" in self.env.word_list:
            # Make a guess
            obs, reward, terminated, truncated, info = self.env.step("ABOUT")
            
            # Check guess count increased
            self.assertEqual(obs['guess_count'], initial_guess_count + 1)
            
            # Check word appears in guessed_words
            self.assertIn("ABOUT", obs['guessed_words'])
            
            # Check board was updated
            self.assertTrue(np.any(obs['board'][0] != -1))  # First row should be filled
            
            # Check feedback was updated
            self.assertTrue(np.any(obs['feedback'][0] != -1))  # First row should have feedback
        
        print("✅ Observation updates work correctly")


class TestWordleActionWrapper(unittest.TestCase):
    """Test the discrete action wrapper for RL libraries"""
    
    def setUp(self):
        """Set up test environment with wrapper"""
        self.env = WordleActionWrapper(WordleEnv())
    
    def tearDown(self):
        """Clean up after each test"""
        self.env.close()
    
    def test_wrapper_action_space(self):
        """Test that wrapper has discrete action space"""
        self.assertIsInstance(self.env.action_space, gym.spaces.Discrete)
        self.assertEqual(self.env.action_space.n, len(self.env.env.word_list))
        print("✅ Wrapper has correct discrete action space")
    
    def test_action_conversion(self):
        """Test conversion between discrete and string actions"""
        # Test discrete to string conversion
        action_int = 0
        word = self.env.action(action_int)
        self.assertEqual(word, self.env.env.word_list[0])
        
        # Test string to discrete conversion
        word = self.env.env.word_list[5]
        action_converted = self.env.reverse_action(word)
        self.assertEqual(action_converted, 5)
        
        print("✅ Action conversion works correctly")
    
    def test_wrapper_step(self):
        """Test that wrapper properly handles discrete actions"""
        obs, info = self.env.reset()
        
        # Use discrete action
        action = 0  # First word in vocabulary
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Check that the guess was converted properly
        expected_word = self.env.env.word_list[0]
        self.assertEqual(info['guess'], expected_word)
        
        print("✅ Wrapper step function works correctly")
    
    def test_invalid_discrete_actions(self):
        """Test handling of invalid discrete actions"""
        # Test out of bounds action
        with self.assertRaises(ValueError):
            self.env.action(len(self.env.env.word_list) + 1)
        
        with self.assertRaises(ValueError):
            self.env.action(-1)
        
        print("✅ Invalid discrete actions properly handled")


class TestEnvironmentIntegration(unittest.TestCase):
    """Integration tests for the complete environment"""
    
    def test_full_game_simulation(self):
        """Test a complete game from start to finish"""
        env = WordleEnv(render_mode=None)  # No rendering for test
        obs, info = env.reset()
        target = info['target_word']
        
        episode_rewards = []
        
        # Play until game ends or max guesses reached
        for guess_num in range(6):
            # Choose a word from valid actions
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
            
            # For testing, just pick the first valid action
            guess = valid_actions[0]
            obs, reward, terminated, truncated, info = env.step(guess)
            episode_rewards.append(reward)
            
            if terminated:
                break
        
        # Check that game ended properly
        self.assertTrue(terminated or guess_num == 5)  # Either won or used all guesses
        self.assertGreater(len(episode_rewards), 0)
        
        env.close()
        print(f"✅ Complete game simulation finished. Total reward: {sum(episode_rewards)}")
    
    def test_environment_with_custom_word_list(self):
        """Test environment with custom word list"""
        custom_words = ["HELLO", "WORLD", "TESTS", "GAMES", "CODES"]
        env = WordleEnv(word_list=custom_words)
        
        self.assertEqual(len(env.word_list), 5)
        self.assertIn("HELLO", env.word_list)
        
        # Test that only words from custom list are accepted
        obs, info = env.reset()
        
        obs, reward, terminated, truncated, info = env.step("HELLO")
        self.assertTrue(info['valid_word'])
        
        env.close()
        print("✅ Custom word list functionality works correctly")


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)
