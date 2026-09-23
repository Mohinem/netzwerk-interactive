import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import string
from typing import Dict, List, Tuple, Any, Optional


class WordleEnv(gym.Env):
    """
    Custom Gymnasium Environment for Wordle Game
    
    The agent tries to guess a 5-letter word within 6 attempts.
    Feedback is provided as:
    - 0: Letter not in word (gray)
    - 1: Letter in word, wrong position (yellow) 
    - 2: Letter in word, correct position (green)
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, word_list: Optional[List[str]] = None, render_mode: Optional[str] = None):
        super().__init__()
        
        # Game constants
        self.WORD_LENGTH = 5
        self.MAX_GUESSES = 6
        self.VOCAB_SIZE = 26  # A-Z
        
        # Load word list (default to a basic set if none provided)
        if word_list is None:
            self.word_list = self._get_default_word_list()
        else:
            self.word_list = [word.upper() for word in word_list if len(word) == self.WORD_LENGTH]
        
        if not self.word_list:
            raise ValueError("Word list is empty or contains no 5-letter words")
            
        self.render_mode = render_mode
        
        # Modified: Action space now accepts strings directly
        # We'll use a custom action space that validates 5-letter strings
        self.action_space = spaces.Text(max_length=5, min_length=5, charset=string.ascii_uppercase)
        
        # Observation space: Dictionary containing:
        # - 'board': 6x5 grid of letters (0-25 for A-Z, -1 for empty)
        # - 'feedback': 6x5 grid of feedback (0=gray, 1=yellow, 2=green, -1=empty)
        # - 'guess_count': number of guesses made
        # - 'valid_words': list of valid words that can be guessed
        self.observation_space = spaces.Dict({
            'board': spaces.Box(low=-1, high=25, shape=(self.MAX_GUESSES, self.WORD_LENGTH), dtype=np.int8),
            'feedback': spaces.Box(low=-1, high=2, shape=(self.MAX_GUESSES, self.WORD_LENGTH), dtype=np.int8),
            'guess_count': spaces.Box(low=0, high=self.MAX_GUESSES, shape=(1,), dtype=np.int8),
            'guessed_words': spaces.Box(low=0, high=1, shape=(len(self.word_list),), dtype=np.int8)
        })
        
        # Initialize game state
        self.reset()
    
    def _get_default_word_list(self) -> List[str]:
        """Returns a basic list of common 5-letter words for testing"""
        return [
            # Your existing words (A-J)
            "ABOUT", "ABOVE", "ABUSE", "ACTOR", "ACUTE", "ADMIT", "ADOPT", "ADULT", "AFTER", "AGAIN",
            "AGENT", "AGREE", "AHEAD", "ALARM", "ALBUM", "ALERT", "ALIEN", "ALIGN", "ALIKE", "ALIVE",
            "ALLOW", "ALONE", "ALONG", "ALTER", "ANGEL", "ANGER", "ANGLE", "ANGRY", "APART", "APPLE",
            "APPLY", "ARENA", "ARGUE", "ARISE", "ARMOR", "ARRAY", "ASIDE", "ASSET", "AUDIO", "AUDIT",
            "AVOID", "AWAKE", "AWARD", "AWARE", "BADLY", "BAKER", "BALLS", "BANDA", "BASIC", "BATCH",
            "BEACH", "BEGAN", "BEGIN", "BEING", "BELOW", "BENCH", "BILLY", "BIRTH", "BLACK", "BLAME",
            "BLANK", "BLIND", "BLOCK", "BLOOD", "BOARD", "BOOST", "BOOTH", "BOUND", "BRAIN", "BRAND",
            "BRAVE", "BREAD", "BREAK", "BREED", "BRIEF", "BRING", "BROAD", "BROKE", "BROWN", "BUILD",
            "BUYER", "CABLE", "CALIF", "CARRY", "CATCH", "CAUSE", "CHAIN", "CHAIR", "CHAOS", "CHARM",
            "CHART", "CHASE", "CHEAP", "CHECK", "CHEST", "CHIEF", "CHILD", "CHINA", "CHOSE", "CIVIL",
            "CLAIM", "CLASS", "CLEAN", "CLEAR", "CLICK", "CLOCK", "CLOSE", "CLOUD", "COACH", "COAST",
            "COULD", "COUNT", "COURT", "COVER", "CRAFT", "CRASH", "CRAZY", "CREAM", "CRIME", "CROSS",
            "CROWD", "CROWN", "CRUDE", "CURVE", "CYCLE", "DAILY", "DANCE", "DATED", "DEALT", "DEATH",
            "DEBUG", "DELAY", "DEPTH", "DOING", "DOUBT", "DOZEN", "DRAFT", "DRAMA", "DRANK", "DREAM",
            "DRESS", "DRILL", "DRINK", "DRIVE", "DROVE", "DYING", "EAGER", "EARLY", "EARTH", "EIGHT",
            "ELITE", "EMPTY", "ENEMY", "ENJOY", "ENTER", "ENTRY", "EQUAL", "ERROR", "EVENT", "EVERY",
            "EXACT", "EXIST", "EXTRA", "FAITH", "FALSE", "FAULT", "FIBER", "FIELD", "FIGHT", "FINAL",
            "FIRST", "FIXED", "FLASH", "FLEET", "FLOOR", "FLUID", "FOCUS", "FORCE", "FORTH", "FORUM",
            "FOUND", "FRAME", "FRANK", "FRAUD", "FRESH", "FRONT", "FRUIT", "FULLY", "FUNNY", "GIANT",
            "GIVEN", "GLASS", "GLOBE", "GOING", "GRACE", "GRADE", "GRAND", "GRANT", "GRASS", "GRAVE",
            "GREAT", "GREEN", "GROSS", "GROUP", "GROWN", "GUARD", "GUESS", "GUEST", "GUIDE", "HAPPY",
            "HARRY", "HEART", "HEAVY", "HENCE", "HENRY", "HORSE", "HOTEL", "HOUSE", "HUMAN", "IDEAL",
            "IMAGE", "INDEX", "INNER", "INPUT", "ISSUE", "JAPAN", "JIMMY", "JOINT", "JONES", "JUDGE",
            
            # Additional words (K-Z)
            "KAYAK", "KICKY", "KNEEL", "KNIFE", "KNOWS", "LABEL", "LARGE", "LAUGH", "LAYER", "LEARN",
            "LEAST", "LEAVE", "LEVEL", "LIGHT", "LIMIT", "LIVER", "LOCAL", "LOGIC", "LUCKY", "MAGIC",
            "MAJOR", "MARCH", "MATCH", "MAYBE", "MEDIA", "MEDIC", "MERGE", "METAL", "MIGHT", "MINOR",
            "MIXED", "MODEL", "MONEY", "MONTH", "MORAL", "MOTOR", "MOUTH", "MUSIC", "NERVE", "NEVER",
            "NIGHT", "NOISE", "NORTH", "NOVEL", "NURSE", "OCEAN", "OFFER", "OFTEN", "ORDER", "OTHER",
            "OWNER", "PAINT", "PANEL", "PARTY", "PEACE", "PILOT", "PINKY", "PLANE", "PLANT", "PLATE",
            "POINT", "POWER", "PRESS", "PRICE", "PRIDE", "PRIZE", "PROOF", "PUSHY", "QUAIL", "QUICK",
            "QUIET", "QUILL", "QUILT", "RADIO", "RANGE", "RAPID", "REACH", "READY", "REALM", "RELAY",
            "REPLY", "RIGHT", "RIVAL", "RIVER", "ROAST", "ROBIN", "ROCKY", "ROUGH", "ROUND", "ROYAL",
            "RURAL", "SAFER", "SAINT", "SALAD", "SALTY", "SANDY", "SCALE", "SCENE", "SCOPE", "SCREW",
            "SENSE", "SERVE", "SEVEN", "SHADE", "SHAKE", "SHAME", "SHAPE", "SHARP", "SHEET", "SHELF",
            "SHIFT", "SHINE", "SHIRT", "SHOCK", "SHOOT", "SHORT", "SHOUT", "SIGHT", "SILLY", "SINCE",
            "SIXTH", "SKILL", "SLEEP", "SLIDE", "SMALL", "SMILE", "SMOKE", "SNACK", "SNAKE", "SNOWY",
            "SOLID", "SOLVE", "SORRY", "SOUND", "SOUTH", "SPACE", "SPARE", "SPEAK", "SPEED", "SPELL",
            "SPEND", "SPICE", "SPINE", "SPITE", "SPLIT", "SPOKE", "SPORT", "STAFF", "STAGE", "STAIR",
            "STAKE", "STAMP", "STAND", "START", "STATE", "STAYS", "STEAM", "STEEL", "STICK", "STILL",
            "STOCK", "STONE", "STOOD", "STORE", "STORM", "STORY", "STRIP", "STUCK", "STUDY", "STUFF",
            "STYLE", "SUGAR", "SUITE", "SUPER", "SURGE", "SWEAT", "SWEET", "SWIFT", "SWING", "SWORD",
            "TABLE", "TAKEN", "TALKS", "TASTE", "TAXES", "TEACH", "TEAMS", "TEARS", "TEENS", "TEETH",
            "TERMS", "TESTS", "THANK", "THEFT", "THEIR", "THEME", "THERE", "THESE", "THICK", "THING",
            "THINK", "THIRD", "THOSE", "THREE", "THREW", "THROW", "THUMB", "TIGHT", "TIMES", "TIRED",
            "TITLE", "TODAY", "TOKEN", "TOOLS", "TOPIC", "TOTAL", "TOUCH", "TOUGH", "TOWER", "TOWNS",
            "TRACK", "TRADE", "TRAIN", "TREAT", "TREES", "TREND", "TRIAL", "TRIBE", "TRICK", "TRIED",
            "TRIES", "TRIPS", "TRUCK", "TRULY", "TRUST", "TRUTH", "TUBES", "TUNED", "TURNS", "TWICE",
            "TWINS", "TWIST", "TYPES", "ULTRA", "UNCLE", "UNDER", "UNION", "UNITY", "UNTIL", "UPPER",
            "UPSET", "URBAN", "URGED", "USAGE", "USERS", "USING", "USUAL", "VALID", "VALUE", "VIDEO",
            "VIEWS", "VIRUS", "VISIT", "VITAL", "VOCAL", "VOICE", "VOTES", "WASTE", "WATCH", "WATER",
            "WAVES", "WAYNE", "WEARY", "WEIRD", "WELLS", "WHEAT", "WHEEL", "WHERE", "WHICH", "WHILE",
            "WHITE", "WHOLE", "WHOSE", "WIDOW", "WIDTH", "WINDS", "WINES", "WINGS", "WIPED", "WIRES",
            "WITCH", "WOMEN", "WORDS", "WORKS", "WORLD", "WORRY", "WORSE", "WORST", "WORTH", "WOULD",
            "WRITE", "WRONG", "WROTE", "YARDS", "YEARS", "YEAST", "YOUNG", "YOURS", "YOUTH", "ZEBRA",
            "ZEROS", "ZONES"
        ]

    def _letter_to_num(self, letter: str) -> int:
        """Convert letter to number (A=0, B=1, ..., Z=25)"""
        return ord(letter) - ord('A')
    
    def _num_to_letter(self, num: int) -> str:
        """Convert number to letter (0=A, 1=B, ..., 25=Z)"""
        return chr(num + ord('A'))
    
    def _word_to_array(self, word: str) -> np.ndarray:
        """Convert word to array of numbers"""
        return np.array([self._letter_to_num(letter) for letter in word], dtype=np.int8)
    
    def _get_feedback(self, guess: str, target: str) -> np.ndarray:
        """
        Calculate feedback for a guess against the target word
        Returns array of feedback codes: 0=gray, 1=yellow, 2=green
        """
        feedback = np.zeros(self.WORD_LENGTH, dtype=np.int8)
        target_letters = list(target)
        guess_letters = list(guess)
        
        # First pass: mark exact matches (green)
        for i in range(self.WORD_LENGTH):
            if guess_letters[i] == target_letters[i]:
                feedback[i] = 2  # Green
                target_letters[i] = None  # Mark as used
                guess_letters[i] = None   # Mark as processed
        
        # Second pass: mark wrong position matches (yellow)
        for i in range(self.WORD_LENGTH):
            if guess_letters[i] is not None:  # Not already processed
                if guess_letters[i] in target_letters:
                    feedback[i] = 1  # Yellow
                    # Remove one instance of this letter from target
                    target_letters[target_letters.index(guess_letters[i])] = None
                # else: feedback[i] remains 0 (gray)
        
        return feedback
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset the environment to start a new game"""
        super().reset(seed=seed)
        
        # Choose random target word
        self.target_word = random.choice(self.word_list)
        
        # Initialize game state
        self.guess_count = 0
        self.guessed_words = set()
        self.game_over = False
        self.won = False
        
        # Initialize observation arrays
        self.board = np.full((self.MAX_GUESSES, self.WORD_LENGTH), -1, dtype=np.int8)
        self.feedback_board = np.full((self.MAX_GUESSES, self.WORD_LENGTH), -1, dtype=np.int8)
        
        observation = self._get_observation()
        info = {'target_word': self.target_word}
        
        return observation, info
    
    def step(self, action: str) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one step of the environment"""
        if self.game_over:
            raise RuntimeError("Game is over. Call reset() to start a new game.")
        
        # Validate input
        if not isinstance(action, str):
            raise ValueError(f"Action must be a string, got {type(action)}")
        
        # Convert to uppercase and validate length
        guess = action.upper().strip()
        
        if len(guess) != self.WORD_LENGTH:
            raise ValueError(f"Invalid guess string '{action}'. Must be exactly {self.WORD_LENGTH} characters")
        
        # Validate that all characters are letters
        if not guess.isalpha():
            raise ValueError(f"Invalid guess string '{action}'. Must contain only letters")
        
        # Optional: Validate that it's a real word from our word list
        if guess not in self.word_list:
            # Give a penalty for invalid words but don't end the game
            reward = -2
            observation = self._get_observation()
            info = {
                'guess': guess,
                'target_word': self.target_word,
                'won': self.won,
                'guess_count': self.guess_count,
                'valid_word': False
            }
            return observation, reward, False, False, info
        
        # Check if word was already guessed (repeat penalty)
        if guess in self.guessed_words:
            reward = -5  # Strong penalty for repeat guess
            observation = self._get_observation()
            info = {
                'guess': guess,
                'target_word': self.target_word,
                'won': self.won,
                'guess_count': self.guess_count,
                'valid_word': True,
                'repeated_guess': True
            }
            return observation, reward, False, False, info
        
        # Add to guessed words
        self.guessed_words.add(guess)
        
        # Update board with the guess
        self.board[self.guess_count] = self._word_to_array(guess)
        
        # Calculate and store feedback
        feedback = self._get_feedback(guess, self.target_word)
        self.feedback_board[self.guess_count] = feedback
        
        # Increment guess count for reward calculation
        self.guess_count += 1
        
        # 🎯 NEW ENHANCED REWARD SYSTEM
        reward = self._calculate_enhanced_reward(guess, feedback, self.guess_count)
        
        # Check win condition
        if np.all(feedback == 2):  # All green - correct guess
            self.won = True
            self.game_over = True
        
        # Check if out of guesses
        if self.guess_count >= self.MAX_GUESSES and not self.won:
            self.game_over = True
            # Failure penalty is already included in _calculate_enhanced_reward
        
        observation = self._get_observation()
        terminated = self.game_over
        truncated = False
        info = {
            'guess': guess,
            'target_word': self.target_word,
            'won': self.won,
            'guess_count': self.guess_count,
            'valid_word': True,
            'green_count': int(np.sum(feedback == 2)),
            'yellow_count': int(np.sum(feedback == 1)),
            'gray_count': int(np.sum(feedback == 0))
        }
        
        return observation, reward, terminated, truncated, info

    def _calculate_enhanced_reward(self, guess: str, feedback: np.ndarray, guess_number: int) -> float:
        """
        Calculate reward using the enhanced Wordle-specific reward structure
        
        Args:
            guess: The word that was guessed
            feedback: Feedback array (0=gray, 1=yellow, 2=green)
            guess_number: Which guess attempt this is (1-6)
        
        Returns:
            Float reward value
        """
        green_count = int(np.sum(feedback == 2))
        yellow_count = int(np.sum(feedback == 1))
        
        # Check if this is a winning guess (all greens)
        if green_count == self.WORD_LENGTH:
            # Win rewards decrease with attempt number (refined scale)
            win_rewards = {
                1: 50,  # Exceptional
                2: 35,  # Excellent  
                3: 25,  # Very good
                4: 15,  # Good
                5: 8,   # Acceptable
                6: 3    # Barely acceptable
            }
            base_win_reward = win_rewards.get(guess_number, 3)
            
            # Step cost for this guess
            step_cost = -1
            
            return base_win_reward + step_cost
        
        # Check if game is over without winning (failure)
        if guess_number >= self.MAX_GUESSES:
            return -30  # Significant failure penalty
        
        # Calculate intermediate rewards for partial progress
        base_reward = 0
        
        # Green letters (correct position) - enhanced with early bonus
        if green_count > 0:
            green_bonus = max(0, 4 - guess_number)  # Earlier greens worth more
            base_reward += green_count * (2 + green_bonus)
        
        # Yellow letters (correct letter, wrong position)
        base_reward += yellow_count * 1
        
        # Step cost (encourages efficiency)
        step_cost = -1
        
        # Optional: Bonus for strategic first guesses with common letters
        if guess_number == 1 and self._has_common_letters(guess):
            base_reward += 1
        
        return base_reward + step_cost

    def _has_common_letters(self, word: str) -> bool:
        """
        Check if word contains common English letters for strategic opening
        
        Args:
            word: The guessed word
            
        Returns:
            True if word contains mostly common letters
        """
        # Common letters in English (and good for Wordle strategy)
        common_letters = set('AEIOURTNLSDH')
        word_letters = set(word)
        
        # Return True if at least 3 letters are common
        return len(word_letters.intersection(common_letters)) >= 3

    
    def _get_observation(self) -> Dict:
        """Get current observation"""
        # Create binary mask for guessed words
        guessed_mask = np.zeros(len(self.word_list), dtype=np.int8)
        for i, word in enumerate(self.word_list):
            if word in self.guessed_words:
                guessed_mask[i] = 1
        
        return {
            'board': self.board.copy(),
            'feedback': self.feedback_board.copy(),
            'guess_count': np.array([self.guess_count], dtype=np.int8),  # ← Must be array!
            'guessed_words': guessed_mask  # ← Must be array!
        }
    
    def _get_enhanced_observation(self) -> Dict:
        """Enhanced observation with explicit constraints"""
        base_obs = self._get_observation()
        
        # Add explicit constraint tracking
        constraints = self._get_letter_constraints()
        
        enhanced_obs = {
            **base_obs,
            'known_positions': constraints['known_positions'],
            'known_letters': constraints['known_letters'],
            'eliminated_letters': constraints['eliminated_letters'],
            'position_constraints': constraints['position_constraints']
        }
        
        return enhanced_obs

    def get_valid_strategic_actions(self):
        """Get valid words that respect discovered constraints"""
        if self.guess_count == 0:
            return self.word_list  # First guess - all words valid
        
        constraints = self._get_letter_constraints()
        valid_words = []
        
        for word in self.word_list:
            if word in self.guessed_words:
                continue  # Skip already guessed words
                
            # Check if word violates constraints
            word_valid = True
            word_array = self._word_to_array(word)
            
            for pos in range(5):
                letter_num = word_array[pos]
                
                # Must use known green letters in correct positions
                if constraints['known_positions'][pos] != -1:
                    if constraints['known_positions'][pos] != letter_num:
                        word_valid = False
                        break
                
                # Can't use eliminated letters
                if constraints['eliminated_letters'][letter_num] == 1:
                    word_valid = False
                    break
                
                # Can't put yellow letters in known wrong positions
                if constraints['position_constraints'][pos][letter_num] == 0:
                    word_valid = False
                    break
            
            # Must include all known letters (yellow/green)
            if word_valid:
                for letter_num in range(26):
                    if constraints['known_letters'][letter_num] == 1:
                        if letter_num not in word_array:
                            word_valid = False
                            break
            
            if word_valid:
                valid_words.append(word)
        
        return valid_words if valid_words else self.get_valid_actions() 

    def _get_letter_constraints(self):
        """Extract strategic constraints from game state"""
        known_positions = np.full(5, -1, dtype=np.int8)
        known_letters = np.zeros(26, dtype=np.int8)
        eliminated_letters = np.zeros(26, dtype=np.int8)
        position_constraints = np.ones((5, 26), dtype=np.int8)
        
        for guess_idx in range(self.guess_count):
            guess_letters = self.board[guess_idx]
            feedback = self.feedback_board[guess_idx]
            
            for pos in range(5):
                letter = guess_letters[pos]
                if feedback[pos] == 2:  # Green - correct position
                    known_positions[pos] = letter
                    known_letters[letter] = 1
                elif feedback[pos] == 1:  # Yellow - wrong position
                    known_letters[letter] = 1
                    position_constraints[pos][letter] = 0
                else:  # Gray - not in word
                    eliminated_letters[letter] = 1
        
        return {
            'known_positions': known_positions,
            'known_letters': known_letters,
            'eliminated_letters': eliminated_letters,
            'position_constraints': position_constraints
        }    
    
    def get_valid_actions(self) -> List[str]:
        """Return list of valid actions (words that haven't been guessed yet)"""
        return [word for word in self.word_list if word not in self.guessed_words]
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render the current state of the game"""
        if mode == 'human':
            print(f"\nWordle Game - Guess {self.guess_count}/{self.MAX_GUESSES}")
            print(f"Target: {'*' * self.WORD_LENGTH}")
            print("-" * 25)
            
            for i in range(self.guess_count):
                word = ''.join([self._num_to_letter(self.board[i][j]) for j in range(self.WORD_LENGTH)])
                feedback_str = ''
                for j in range(self.WORD_LENGTH):
                    if self.feedback_board[i][j] == 2:
                        feedback_str += '🟩'  # Green
                    elif self.feedback_board[i][j] == 1:
                        feedback_str += '🟨'  # Yellow
                    else:
                        feedback_str += '⬜'  # Gray
                print(f"{word} {feedback_str}")
            
            # Show remaining empty rows
            for i in range(self.guess_count, self.MAX_GUESSES):
                print("_____ ⬜⬜⬜⬜⬜")
            
            if self.game_over:
                if self.won:
                    print(f"\n🎉 Congratulations! You guessed '{self.target_word}' in {self.guess_count} tries!")
                else:
                    print(f"\n😞 Game Over! The word was '{self.target_word}'")
        
        elif mode == 'rgb_array':
            # For simplicity, return a basic array representation
            # In a full implementation, you'd create an actual image
            return np.array([[[255, 255, 255]]], dtype=np.uint8)
    
    def close(self):
        """Clean up resources"""
        pass


# Custom wrapper to work with RL libraries that expect discrete actions
class WordleActionWrapper(gym.ActionWrapper):
    """Wrapper to convert between discrete integer actions and string actions"""
    
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        # Keep discrete action space for RL libraries
        self.action_space = spaces.Discrete(len(env.word_list))
    
    def action(self, action):
        """Convert discrete action to string"""
        # Handle numpy arrays with single elements
        if isinstance(action, np.ndarray):
            if action.shape == (1,) or action.shape == ():
                action = int(action.item())  # Extract scalar from array
            else:
                raise ValueError(f"Expected scalar action, got array with shape {action.shape}")
        
        # Handle regular integers (including numpy integers)
        if isinstance(action, (int, np.integer)):
            if 0 <= action < len(self.env.word_list):
                return self.env.word_list[action]
            else:
                raise ValueError(f"Invalid action {action}")
        else:
            return action  # Already a string
    
    def reverse_action(self, action):
        """Convert string action to discrete"""
        if isinstance(action, str):
            try:
                return self.env.word_list.index(action.upper())
            except ValueError:
                raise ValueError(f"Word '{action}' not in vocabulary")
        else:
            return action  # Already an integer



# Register the environment with Gymnasium
gym.register(
    id='Wordle-v0',
    entry_point='__main__:WordleEnv',
    max_episode_steps=6,
)

gym.register(
    id='WordleDiscrete-v0',
    entry_point=lambda: WordleActionWrapper(WordleEnv()),
    max_episode_steps=6,
)


# Example usage and testing
if __name__ == "__main__":
    # Test with string actions
    print("=== Testing with String Actions ===")
    env = WordleEnv(render_mode='human')
    
    print("Wordle Environment Created!")
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    print(f"Vocabulary Size: {len(env.word_list)}")
    
    # Test the environment
    observation, info = env.reset()
    print(f"\nTarget word for testing: {info['target_word']}")
    
    # Manual test with string actions
    test_words = ["ABOUT", "APPLE", "BRAIN"]
    
    for word in test_words:
        print(f"\nTrying guess: {word}")
        observation, reward, terminated, truncated, info = env.step(word)
        env.render()
        print(f"Reward: {reward}")
        print(f"Valid word: {info.get('valid_word', True)}")
        
        if terminated:
            break
    
    env.close()
    
    # Test with discrete action wrapper for RL libraries
    print("\n=== Testing with Discrete Action Wrapper ===")
    wrapped_env = WordleActionWrapper(WordleEnv(render_mode='human'))
    
    observation, info = wrapped_env.reset()
    print(f"Target word: {info['target_word']}")
    print(f"Action space (wrapped): {wrapped_env.action_space}")
    
    # Test with integer actions (for RL libraries)
    for i in range(3):
        action = wrapped_env.action_space.sample()  # Random action
        observation, reward, terminated, truncated, info = wrapped_env.step(action)
        wrapped_env.render()
        print(f"Action: {action} -> Word: {info['guess']}, Reward: {reward}")
        
        if terminated:
            break
    
    wrapped_env.close()
