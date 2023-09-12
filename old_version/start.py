from old_version.paddle import Paddle, AIPaddle
from old_version.game import PongGame
import pygame
#print("Which type of game to you want?")
print("-----------------------------")
print("            PONG             ")
print("-----------------------------")
print("[*] For Human against AI press 1?")
print("[*] For AI againstx AI press 2?")
#gameoption = input("[*] Select [1/2]: ")
gameoption = "1"

if gameoption == "1":
    # Create one human player paddles
    paddle1 = Paddle()
    paddle2 = AIPaddle()
    # Create a PongGame instance with the human player paddles
    game = PongGame(paddle1, paddle2)
    # Start the game
    game.playHumanAI()

elif gameoption == "2":
    # Create two human player paddles
    paddle1 = AIPaddle()
    paddle2 = AIPaddle()
    # Create a PongGame instance with the human player paddles
    game = PongGame(paddle1, paddle2)
    # Start the game
    game.playAIAI()
