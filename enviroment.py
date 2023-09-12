import random
import numpy as np
import pygame
import torch
from Agent import Agent

SCREEN_WIDTH = 700
SCREEN_HEIGHT = 500
PADDLE_HEIGHT = 50
PADDLE_WIDTH = 10
BALL_SIZE = 5
PADDLE_COLOR = (255,255,255)
BALL_COLOR = (255,255,255)
BACKGROUND_COLOR = (0,0,0)

random.seed = 42

class GameEngine():
    def __init__(self,initialize_pygame=False):
        self.paddle_left = Paddle(x=0,y=200)
        self.paddle_right = Paddle(x=699,y=200)
        self.ball = Ball(300,100,5,5)
        if initialize_pygame:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    @staticmethod
    def sample():
        return random.choice([0,1,2])

    def reset(self):
        """ resets game to new state"""
        self.paddle_left.y = 200
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
        if self.ball.y >= SCREEN_HEIGHT or self.ball.y <= 0:
            self.ball.bounce_wall()
        if self.ball.x <= 5:
            if (self.ball.y >= self.paddle_left.y - PADDLE_HEIGHT/2 and
                self.ball.y <= self.paddle_left.y + PADDLE_HEIGHT/2):
                self.ball.bounce_paddle()
                reward1 = 1
            else: 
                terminated = True
                self.reset()
        if self.ball.x >= SCREEN_WIDTH-5:
            if (self.ball.y >= self.paddle_right.y - PADDLE_HEIGHT/2 and
                self.ball.y <= self.paddle_right.y + PADDLE_HEIGHT/2):
                self.ball.bounce_paddle()
                reward2 = 1  
            else: 
                terminated = True
                self.reset()
        return terminated, reward1, reward2
    
    def draw(self):
        # Clear the screen
        self.screen.fill(BACKGROUND_COLOR)
        # Draw the paddles
        self.paddle_left.draw(self.screen)
        self.paddle_right.draw(self.screen)
        self.ball.draw(self.screen)
        # Update the display
        pygame.display.flip()
    
    def event_handeling(self) -> bool:
        """ checks if programm should be terminated condition """
        for event in pygame.event.get():  # User did something
            if event.type == pygame.QUIT:  # If user clicked close
                return False  # Flag that we are done so we exit this loop
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:  # Pressing the x Key will quit the game
                    return False
        return True

    def run(self,path_left,path_right):
        agent_left = Agent(6,3)
        agent_left.policy_net.load_state_dict(torch.load(path_left,map_location='cpu'))
        agent_right = Agent(6,3)
        agent_right.policy_net.load_state_dict(torch.load(path_right,map_location='cpu'))
        clock = pygame.time.Clock()
        carryOn = True
        while carryOn:
            carryOn = self.event_handeling()
            self.ball.update()
            action_left= self.get_action_ai(agent_left)
            action_right= self.get_action_ai(agent_right)
            self.paddle_left.update(action_left)
            self.paddle_right.update(action_right)
            terminated, _, _ = self.collision_detection()
            if terminated:
                self.reset()
            self.draw()      
            clock.tick(60)   
        pygame.quit()


    def run_human(self,path_left,path_right):
        agent_left = Agent(6,3)
        agent_left.policy_net.load_state_dict(torch.load(path_left,map_location='cpu'))
        clock = pygame.time.Clock()
        carryOn = True
        while carryOn:
            carryOn = self.event_handeling()
            self.ball.update()
            action_left= self.get_action_ai(agent_left)
            self.paddle_left.update(action_left)
            action_human = self.get_action_human()
            self.paddle_right.update(action_human)
            terminated, _, _ = self.collision_detection()
            if terminated:
                self.reset()
            self.draw()      
            clock.tick(60)   
        pygame.quit()
    
    def get_action_ai(self,agent):
        state = np.array([self.ball.x, self.ball.y, self.ball.vx, self.ball.vy,
                        self.paddle_left.y, self.paddle_right.y])
        state = torch.tensor(state, dtype=torch.float32, device='cpu').unsqueeze(0)
        #action_left =  agent_left.policy_net(state).max(1)[1].view(1, 1)
        action =  agent.policy_net(state).max(1)[1].view(1, 1)
        return action
    
    def get_action_human(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            return 2
        if keys[pygame.K_DOWN]:
            return 1

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

    def bounce_paddle(self):
        # updates velocity
        self.vx = -self.vx

    def draw(self,screen):
        pygame.draw.rect(screen,BALL_COLOR,(self.x - BALL_SIZE/2,
                                            self.y - BALL_SIZE/2,
                                            BALL_SIZE,
                                            BALL_SIZE))    

class Paddle:
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def update(self,action,pixles=8):
        # updates paddle positon depending on the action
        if action == 1:
            self.y += pixles
        elif action == 2:
            self.y -= pixles
        if self.y - PADDLE_HEIGHT/2 < 0:
          self.y = PADDLE_HEIGHT/2
        elif self.y + PADDLE_HEIGHT/2 > SCREEN_HEIGHT:
          self.y = SCREEN_HEIGHT - PADDLE_HEIGHT/2

    def draw(self, screen):
        # Draw the paddle on the screen
        pygame.draw.rect(screen, PADDLE_COLOR, (self.x - PADDLE_WIDTH/2,
                                                self.y - PADDLE_HEIGHT/2,
                                                PADDLE_WIDTH,
                                                PADDLE_HEIGHT))

    