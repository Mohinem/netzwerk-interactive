# Simple placeholder gamification logic
def calculate_points(is_correct: bool, streak: int) -> int:
    base_points = 10 if is_correct else 0
    streak_bonus = 5 * streak if is_correct else 0
    return base_points + streak_bonus
