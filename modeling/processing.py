import numpy as np
import math


def initialize_state(game):
    """Initialize the state grid."""
    state = np.zeros((game.height, game.width), dtype=np.float32)
    return state


def mark_snake_on_state(game, state):
    """Mark the snake's body, head, and tail on the state grid."""
    for x, y in game.snake[1:-1]:
        state[y, x] = 2
    tail_x, tail_y = game.snake[-1]
    state[tail_y, tail_x] = 3
    head_x, head_y = game.snake[0]
    state[head_y, head_x] = 1
    return state


def mark_food_on_state(game, state):
    """Mark the food's position on the state grid."""
    food_x, food_y = game.food
    state[food_y, food_x] = 4
    return state


def make_border(game, state):
    new_state = np.full((game.height + 2, game.width + 2), 10, dtype=np.float32)
    new_state[1:-1, 1:-1] = state
    return new_state


def calculate_food_direction(game, numeric_value=True):
    """Calculate the direction and distance to the food from the snake's head."""
    head_x, head_y = game.snake[0]
    food_x, food_y = game.food
    if numeric_value:
        left = food_x - head_x
        up = food_y - head_y
        radius = math.sqrt((food_x - head_x) ** 2 + (food_y - head_y) ** 2)
        theta = math.atan2(food_y - head_y, food_x - head_x)
        return np.array([left, up, radius, theta], dtype=np.float32)
    left = food_x < head_x
    right = food_x > head_x
    up = food_y > head_y
    down = food_y < head_y
    return np.array([left, right, up, down], dtype=np.float32)


def calculate_clear_path(game, state):
    """Calculates number of valid moves left till collition in certien direction"""
    head_x, head_y = game.snake[0]
    no_border_game = state[1: -1, 1: -1]

    left = no_border_game[head_y, : head_x] if head_x != 0 else [0]
    right = no_border_game[head_y, head_x + 1:] if head_x != (game.width - 1) else [game.width - 1]
    free_left = np.isin(left, [0, 4], invert=True)
    free_right = np.isin(right, [0, 4], invert=True)
    free_left_moves = np.min(np.where(free_left[::-1])) if np.sum(free_left) > 0 else head_x
    free_right_moves = np.min(np.where(free_right)) if np.sum(free_right) > 0 else game.width - 1 - head_x

    up = no_border_game[:head_y, head_x] if head_y != 0 else [0]
    down = no_border_game[head_y + 1:, head_x] if head_y != (game.height - 1) else [game.height - 1]
    free_up = np.isin(up, [0, 4], invert=True)
    free_down = np.isin(down, [0, 4], invert=True)
    free_up_moves = np.min(np.where(free_up[::-1])) if np.sum(free_up) > 0 else head_y
    free_down_moves = np.min(np.where(free_down)) if np.sum(free_down) > 0 else game.height - 1 - head_y
    return np.array([free_left_moves, free_right_moves, free_up_moves, free_down_moves], dtype=np.float32)



def calculate_death_indicators(game):
    """Calculate indicators for whether moving left, right, up, or down would result in death."""
    head_x, head_y = game.snake[0]
    death_indicators = np.zeros(4, dtype=np.float32)  # [left, right, up, down]
    for i, (dx, dy) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
        nx, ny = head_x + dx, head_y + dy
        if nx < 0 or nx >= game.width or ny < 0 or ny >= game.height or (nx, ny) in game.snake:
            death_indicators[i] = 1
    return death_indicators


def calculate_direction_indicators(game):
    direction = game.snake_direction
    return np.array([np.argmax(direction == ['UP', 'DOWN', 'LEFT', 'RIGHT'])], dtype=np.float32)


def preprocess_state(game, for_cnn=True, food_direction=True, add_death_indicators=True,
                     direction=True, clear_path_pixels=False, length_aware=True):
    """Convert the game state into a 2D grid suitable for CNN input, with optional features."""
    state = initialize_state(game)
    state = mark_snake_on_state(game, state)
    state = mark_food_on_state(game, state)
    state = make_border(game, state)

    additional_features = []

    if food_direction:
        additional_features.extend(calculate_food_direction(game))

    if length_aware:
        additional_features.extend(np.array([len(game.snake)], dtype=np.float32))

    if add_death_indicators:
        additional_features.extend(calculate_death_indicators(game))

    if direction:
        additional_features.extend(calculate_direction_indicators(game))

    if clear_path_pixels:
        additional_features.extend(calculate_clear_path(game, state))

    if for_cnn:
        state = state.reshape(game.height + 2, game.width + 2)
        if additional_features:
            additional_features_chanels = np.stack(
                [np.full((game.height + 2, game.width + 2), feat) for feat in additional_features] + [state], axis=0)
            return additional_features_chanels
    else:
        state = state.flatten()
        if additional_features:
            state = np.concatenate([state, additional_features])

    return state.astype(np.float32)


def postprocess_action(action):
    """Convert the neural network's output action into game action."""
    actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    return actions[action]