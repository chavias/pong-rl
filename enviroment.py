
import pygame
import random
import numpy as np

screen_width = 700
screen_height = 500
paddle_height = 50
random.seed = 42


class game:
    def __init__(self):
        self.paddle_left = Paddle(x=0,y=200)
        self.paddle_right = Paddle(x=699,y=200)
        self.ball = Ball(300,100,5,5)
    
    @staticmethod
    def sample():
        return random.choice([1,2,3])

    def reset(self):
        """ resets game to new state"""
        self.paddel_left.y = 200
        self.paddle_right.y = 200
        self.ball.x = 300 + random.uniform(-15,15)
        self.ball.y = 100 + random.uniform(-15,15)
        self.ball.vx = 5*random.choice([-1,1])
        self.ball.vy = 5*random.choice([-1,1])
        return np.array([self.ball.x, self.ball.y, self.ball.vx, self.ball.vy,
                self.paddle_left.y, self.paddle_right.y]) # not sure if this is right

    def step(self, action_left,action_right):
        """ step through the game """
        # update the game 
        self.ball.update()
        self.paddle_left.update(action_left)
        self.paddle_right.update(action_right)
        # perform collision detection
        terminated, reward1, reward2 = self.collision_detection()
        # collect observations
        observation = np.array([self.ball.x, self.ball.y, self.ball.vx, self.ball.vy,
                        self.paddle_left.y, self.paddle_right.y])
        return observation, reward1, reward2, terminated

    def collision_detection(self) -> tuple:
        """ updates velocity of the ball """
        reward1 = 0
        reward2 = 0
        terminated = False
        if self.ball.x >= screen_width or self.ball.x <= 0:
            self.ball.bounce_wall()
        if self.ball.x <= 0:
            if (self.ball.y >= self.paddle_left.y - paddle_height/2 and
                self.ball.y <= self.paddle_left.y + paddle_height/2):
                self.ball.bounce_paddel()
                reward1 = 1
            else: 
                terminated = True
        if self.ball.x <= screen_width:
            if (self.ball.y >= self.paddle_right - paddle_height/2 and
                self.ball.y <= self.paddle_right + paddle_height/2):
                self.ball.bounce_paddel()
                reward2 = 1  
            else: 
                terminated = True      
        return terminated, reward1, reward2

class Ball:  
    def __init__(self,x,y,vx,vy):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy

    def update(self):
        # updates position
        self.x += self.vx
        self.y += self.vy

    def bounce_wall(self):
        # updates velocity
        self.vy = -self.vy

    def bounce_paddel(self):
        # updates velocity
        self.vx = -self.vy

class Paddle:
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def update(self,action,pixles=5):
        # updates paddle positon depending on the action
        if action == 1:
            self.y += pixles
        elif action == 2:
            self.y -= pixles
        if self.y - paddle_height/2 < 0:
          self.y = 0
        elif self.y + paddle_height/2 > screen_height:
          self.y = screen_height