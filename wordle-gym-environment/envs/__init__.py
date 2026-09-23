from gymnasium.envs.registration import register

# Register Wordle environment
register(
    id='Wordle-v0',
    entry_point='envs.wordle_env:WordleEnv',
    max_episode_steps=6,
)