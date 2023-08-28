import pygame
from paddle import Paddle
from ball import Ball
import AI
import time

class PongGame:
    def __init__(self, paddle1, paddle2):
        # Initialize Pygame
        pygame.init()
        # Set up the screen
        self.screen_width = 700
        self.screen_height = 500
        self.screen = pygame.display.set_mode(
            (self.screen_width, self.screen_height))
        pygame.display.set_caption("Pong")
        # Set up the game objects
        self.paddle1 = paddle1
        self.paddle2 = paddle2
        self.paddle1.rect.x = 0
        self.paddle1.rect.y = 200
        self.paddle2.rect.x = 699
        self.paddle2.rect.y = 200
        self.ball = Ball()
        # This will be a list that will contain all the sprites we intend to use in our game.
        self.all_sprites_list = pygame.sprite.Group()
        # Add the 2 paddles and the ball to the list of objects
        self.all_sprites_list.add(self.paddle1)
        self.all_sprites_list.add(self.paddle2)
        self.all_sprites_list.add(self.ball)
        # Set up the scores
        self.scoreA = 0
        self.scoreB = 0
        self.done = False
        self.carryOn = True

    def draw(self):
        """ draws court and scores. Does not update sprites"""
        self.screen.fill((0, 0, 0))
        pygame.draw.line(self.screen, (255, 255, 255), (self.screen_width /
                         2, 0), (self.screen_width/2, self.screen_height), 2)
        self.all_sprites_list.draw(self.screen)
        WHITE = (255, 255, 255)
        # Display scores:
        font = pygame.font.Font(None, 74)
        text = font.render(str(self.scoreA), 1, WHITE)
        self.screen.blit(text, (250, 10))
        text = font.render(str(self.scoreB), 1, WHITE)
        self.screen.blit(text, (420, 10))
        pygame.display.flip()

    def reset_game(self):
        """ reset the game """
        self.done = True
        self.paddle1.rect.x = 0
        self.paddle1.rect.y = 200
        self.paddle2.rect.x = 695
        self.paddle2.rect.y = 200
        self.ball.rect.x = 20
        self.ball.rect.y = 20
        # Update game objects
        self.all_sprites_list.update()
        self.draw()
        self.ball.velocity = [-8, -8]
        self.scoreA = 0
        self.scoreB = 0

    def swept_collision_detection_wall(self, ball):
        # Predict the ball's position after a small time step
        projected_x = ball.rect.x + ball.velocity[0]
        projected_y = ball.rect.y + ball.velocity[1]   
        # Check for collision with the right wall
        if projected_x >= 699:
            ball.velocity[0] = -ball.velocity[0]
            ball.rect.x = 699  # Adjust position to avoid going beyond the wall
        # Check for collision with the left wall
            self.scoreB+=1
        if projected_x <= 0 :
            ball.velocity[0] = -ball.velocity[0]
            ball.rect.x = 0 # Adjust position to avoid going beyond the wall
        return 0  # No reward for wall collisions

    def swept_collision_detection(self, ball, paddle1,paddle2):
        # Predict the ball's position after a small time step
        projected_x = ball.rect.x + ball.velocity[0]
        projected_y = ball.rect.y + ball.velocity[1]
        ball.rect.x = projected_x
        ball.rect.y = projected_y
        reward = 0
        if pygame.sprite.collide_rect(ball, paddle1):
            # Reflect the velocity as if bouncing off the paddle
            ball.velocity[0] = -ball.velocity[0]
        if pygame.sprite.collide_rect(ball, paddle2):
            ball.velocity[0] = -ball.velocity[0]
            reward = 1
        return reward
        
    def collision_detection(self):
        reward = 0
        """ Simple collision detection with wall and Paddles """
        if self.ball.rect.y > 490:
            self.ball.velocity[1] = -self.ball.velocity[1]
        if self.ball.rect.y < 0:
            self.ball.velocity[1] = -self.ball.velocity[1]
        reward = self.swept_collision_detection(self.ball, self.paddle1,self.paddle2)
        self.swept_collision_detection_wall(self.ball)
        # Ball-paddle collision detection with swept collision handling
        # self.ball.velocity[0] = -self.ball.velocity[0]
        # self.ball.velocity[0] = -self.ball.velocity[0]
        # if self.swept_collision_detection(self.ball, self.paddle2.rect):
        return reward

    def event_handeling(self):
        """ checks if programm should be terminated condition """
        for event in pygame.event.get():  # User did something
            if event.type == pygame.QUIT:  # If user clicked close
                self.carryOn = False  # Flag that we are done so we exit this loop
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_x:  # Pressing the x Key will quit the game
                    self.carryOn = False

    def event_handeling_learning(self):
        """ checks if programm should be terminated condition during training """
        for event in pygame.event.get():  # User did something
            if event.type == pygame.QUIT:  # If user clicked close
                pygame.quit()  # Flag that we are done so we exit this loop
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:  # Pressing the x Key will quit the game
                    pygame.quit()
                    quit()

    def get_ball_position(self):
        x, y = self.ball.rect.center
        return x, y

    def get_game_state(self):
        """ returns a dictionary about the game state """
        game_state = {
            'ball_x': self.ball.rect.x,
            'ball_y': self.ball.rect.y,
            'paddle1': self.paddle2.rect.y,
        }
        return game_state

    def step(self, ai_action):
        """  take aktion and receive feedback """
#        clock = pygame.time.Clock()
        _, ball_y = self.get_ball_position()
        #self.paddle1.move(ball_y=ball_y)
        self.paddle2.move(ai_action)
        self.all_sprites_list.update()
        reward = self.collision_detection()
        self.draw()
        CarryOn = self.event_handeling_learning()
        done = self.scoreA >= 10 or self.scoreB >= 10
        if done: 
            self.scoreA = 0
            self.scoreB = 0
        next_state = self.get_game_state()  # Implement your method to get the game state
#        clock.tick(60)
        return next_state, reward, done

    def playAIAI(self):
        # Set up the clock
        clock = pygame.time.Clock()
        # Main game loop
        while self.carryOn:
            # Event handling
            CarryOn = self.event_handeling()
            # ball position
            ball_x, ball_y = self.get_ball_position()
            # AI Paddle movement
            self.paddle1.move(ball_y=ball_y)
            self.paddle2.move(ball_y=ball_y)
            # Update game objects
            self.all_sprites_list.update()
            self.collision_detection()
            # drawing
            self.draw()
            # Main event loop
            CarryOn = self.event_handeling()
            # Check for game over
            # if self.scoreA or self.scoreB >= 1000:
            #     pygame.quit()
            #     return
            # Tick the clock
            clock.tick(60)

    def playHumanAI(self):
        # Set up the clock
        clock = pygame.time.Clock()
        # Main game loop
        while self.carryOn:
            # event handeling
            CarryOn = self.event_handeling()
            # ball position
            ball_x, ball_y = self.get_ball_position()
            # Human Paddle movement
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                self.paddle1.move(-5)
            if keys[pygame.K_DOWN]:
                self.paddle1.move(5)
            self.paddle2.move(ball_y=ball_y)
            # self.paddle2.move(AI.ai_move(ball_y=self.ball.rect.y,paddle_y=self.paddle2.rect.y))
            # Update game objects
            self.all_sprites_list.update()
            # Collision detection
            self.collision_detection()
            # Drawing
            self.draw()
            # event handeling
            CarryOn = self.event_handeling()
            # Tick the clock
            clock.tick(60)
