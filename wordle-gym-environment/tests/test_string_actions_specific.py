import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.wordle_env import WordleEnv

class TestStringActionSpecifics(unittest.TestCase):
    """Specific tests for string action functionality"""
    
    def setUp(self):
        self.env = WordleEnv()
    
    def tearDown(self):
        self.env.close()
    
    def test_case_handling(self):
        """Test various case combinations"""
        obs, info = self.env.reset()
        
        test_cases = ["about", "ABOUT", "About", "aBout", "  ABOUT  "]
        
        for case in test_cases:
            self.env.reset()
            if "ABOUT" in self.env.word_list:
                obs, reward, terminated, truncated, info = self.env.step(case)
                self.assertEqual(info['guess'], "ABOUT")
                print(f"✅ Input '{case}' correctly converted to 'ABOUT'")
    
    def test_whitespace_handling(self):
        """Test whitespace trimming"""
        obs, info = self.env.reset()
        
        if "BRAIN" in self.env.word_list:
            # Test with various whitespace
            obs, reward, terminated, truncated, info = self.env.step("  BRAIN  ")
            self.assertEqual(info['guess'], "BRAIN")
            
            self.env.reset()
            obs, reward, terminated, truncated, info = self.env.step("\tBRAIN\n")
            self.assertEqual(info['guess'], "BRAIN")
        
        print("✅ Whitespace handling works correctly")
    
    def test_special_character_rejection(self):
        """Test rejection of special characters"""
        self.env.reset()
        
        invalid_inputs = [
            "AB!DE",  # Special character
            "AB DE",  # Space in middle
            "ABCD5",  # Number
            "AB-CD",  # Hyphen
            "AB_CD",  # Underscore
            "CAFÉ1",  # Accent + number (if this somehow gets through length check)
        ]
        
        for invalid_input in invalid_inputs:
            with self.assertRaises(ValueError):
                self.env.step(invalid_input)
        
        print("✅ Special characters properly rejected")
    
    def test_word_validation_scenarios(self):
        """Test different word validation scenarios"""
        obs, info = self.env.reset()
        
        # Test valid word from vocabulary
        if "QUICK" in self.env.word_list:
            obs, reward, terminated, truncated, info = self.env.step("QUICK")
            self.assertTrue(info.get('valid_word', False))
            self.assertNotEqual(reward, -2)
        
        # Test invalid word (not in vocabulary)
        obs, reward, terminated, truncated, info = self.env.step("XYZZZ")
        self.assertFalse(info.get('valid_word', True))
        self.assertEqual(reward, -2)
        
        print("✅ Word validation scenarios work correctly")
    
    def test_reward_structure_with_strings(self):
        """Test reward structure with string actions"""
        obs, info = self.env.reset()
        target = info['target_word']
        
        # Test winning reward
        obs, reward, terminated, truncated, info = self.env.step(target)
        self.assertEqual(reward, 100)
        self.assertTrue(info['won'])
        
        # Reset and test partial match rewards
        obs, info = self.env.reset()
        
        # Find a word that might have partial matches
        test_word = None
        for word in self.env.word_list[:10]:
            if word != info['target_word']:
                test_word = word
                break
        
        if test_word:
            obs, reward, terminated, truncated, info = self.env.step(test_word)
            # Reward should be based on green/yellow letters minus penalty
            # Should be > -50 (not a loss) and < 100 (not a win)
            self.assertGreater(reward, -50)
            self.assertLess(reward, 100)
        
        print("✅ Reward structure works correctly with string actions")
    
    def test_action_mask_replacement(self):
        """Test that guessed_words list works as action mask replacement"""
        obs, info = self.env.reset()
        
        # Initially no words should be guessed
        self.assertEqual(len(obs['guessed_words']), 0)
        
        # Make a guess
        if "DANCE" in self.env.word_list:
            obs, reward, terminated, truncated, info = self.env.step("DANCE")
            
            # Word should appear in guessed_words
            self.assertIn("DANCE", obs['guessed_words'])
            self.assertEqual(len(obs['guessed_words']), 1)
            
            # Valid actions should not include the guessed word
            valid_actions = self.env.get_valid_actions()
            self.assertNotIn("DANCE", valid_actions)
            self.assertEqual(len(valid_actions), len(self.env.word_list) - 1)
        
        print("✅ Guessed words tracking works correctly")


if __name__ == '__main__':
    unittest.main(verbosity=2)
