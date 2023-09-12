import pygame
from ball import Ball
from paddle import Paddle


# Define the AI's decision-making logic
def ai_move(ball_y, paddle_y):
    if ball_y > paddle_y:
        return 5  # Move paddle down speed can be adjusted
    elif ball_y < paddle_y:
        return -5  # Move paddle up
    else:
        return 0  # Don't move paddle

# Inside the game loop, update the AI's paddle position

