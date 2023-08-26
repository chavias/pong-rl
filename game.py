import pygame
from paddle import Paddle
from ball import Ball
import AI

class PongGame:
    def __init__(self, paddle1, paddle2):
        # Initialize Pygame
        pygame.init()

        # Set up the screen
        self.screen_width = 700
        self.screen_height = 500
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Pong")

        # Set up the game objects
        self.paddle1 = paddle1
        self.paddle2 = paddle2
        self.paddle1.rect.x = 10
        self.paddle1.rect.y = 200
        self.paddle2.rect.x = 680
        self.paddle2.rect.y = 200
        self.ball = Ball()
        #This will be a list that will contain all the sprites we intend to use in our game.
        self.all_sprites_list = pygame.sprite.Group()
    
        # Add the 2 paddles and the ball to the list of objects
        self.all_sprites_list.add(self.paddle1)
        self.all_sprites_list.add(self.paddle2)
        self.all_sprites_list.add(self.ball)

        # Set up the scores
        self.scoreA = 0
        self.scoreB = 0

    def playHumanAI(self):
        # Set up the clock
        clock = pygame.time.Clock()
        carryOn = True
        # Main game loop
        while carryOn:
            # Event handling
            for event in pygame.event.get(): # User did something
                if event.type == pygame.QUIT: # If user clicked close
                    carryOn = False # Flag that we are done so we exit this loop
                elif event.type==pygame.KEYDOWN:
                        if event.key==pygame.K_x: #Pressing the x Key will quit the game
                            carryOn=False
            # ball position
            ball_x,ball_y = self.ball.rect.center
            # Human Paddle movement  
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                self.paddle1.move(-5)
            if keys[pygame.K_DOWN]:
                self.paddle1.move(5)

            self.paddle2.move(ball_y=ball_y)
            #self.paddle2.move(AI.ai_move(ball_y=self.ball.rect.y,paddle_y=self.paddle2.rect.y))
            

            # Update game objects
            self.all_sprites_list.update()
            
            #Check if the ball is bouncing against any of the 4 walls:
            if self.ball.rect.x>=690:            
                self.scoreA+=1
                self.ball.velocity[0] = -self.ball.velocity[0]
                #self.ball.velocityReset()
            if self.ball.rect.x<=0:
                self.scoreB+=1
                self.ball.velocity[0] = -self.ball.velocity[0]
                #self.ball.velocityReset()
            if self.ball.rect.y>490:
                self.ball.velocity[1] = -self.ball.velocity[1]
            if self.ball.rect.y<0:
                self.ball.velocity[1] = -self.ball.velocity[1]     
            # Ball-paddle collision detection
            if pygame.sprite.collide_rect(self.ball, self.paddle1):
                self.ball.bounce()
            elif pygame.sprite.collide_rect(self.ball, self.paddle2):
                self.ball.bounce()

            # Drawing
            self.screen.fill((0, 0, 0))
            pygame.draw.line(self.screen, (255, 255, 255), (self.screen_width/2, 0), (self.screen_width/2, self.screen_height), 2)
            # self.paddle1.draw(self.screen)
            # self.paddle2.draw(self.screen)
            # self.ball.draw(self.screen)
            self.all_sprites_list.draw(self.screen) 
 
            WHITE = (255,255,255)
            #Display scores:
            font = pygame.font.Font(None, 74)
            text = font.render(str(self.scoreA), 1, WHITE)
            self.screen.blit(text, (250,10))
            text = font.render(str(self.scoreB), 1, WHITE)
            self.screen.blit(text, (420,10))
            pygame.display.flip()
            # --- Main event loop
            for event in pygame.event.get(): # User did something
                if event.type == pygame.QUIT: # If user clicked close
                    carryOn = False # Flag that we are done so we exit this loop
                elif event.type==pygame.KEYDOWN:
                        if event.key==pygame.K_x: #Pressing the x Key will quit the game
                            carryOn=False
            # Check for game over
            # if self.scoreA or self.scoreB >= 1000:
            #     pygame.quit()
            #     return
          
            # Tick the clock
            clock.tick(60)


    def playAIAI(self):
        # Set up the clock
        clock = pygame.time.Clock()
        carryOn = True
        # Main game loop
        while carryOn:
            # Event handling
            for event in pygame.event.get(): # User did something
                if event.type == pygame.QUIT: # If user clicked close
                    carryOn = False # Flag that we are done so we exit this loop
                elif event.type==pygame.KEYDOWN:
                        if event.key==pygame.K_x: #Pressing the x Key will quit the game
                            carryOn=False
            # ball position
            ball_x,ball_y = self.ball.rect.center
            
            # AI Paddle movement  
            self.paddle1.move(ball_y=ball_y)
            self.paddle2.move(ball_y=ball_y)
            #self.paddle2.move(AI.ai_move(ball_y=self.ball.rect.y,paddle_y=self.paddle2.rect.y))
            

            # Update game objects
            self.all_sprites_list.update()
            
            #Check if the ball is bouncing against any of the 4 walls:
            if self.ball.rect.x>=690:            
                self.scoreA+=1
                self.ball.velocity[0] = -self.ball.velocity[0]
                #self.ball.velocityReset()
            if self.ball.rect.x<=0:
                self.scoreB+=1
                self.ball.velocity[0] = -self.ball.velocity[0]
                #self.ball.velocityReset()
            if self.ball.rect.y>490:
                self.ball.velocity[1] = -self.ball.velocity[1]
            if self.ball.rect.y<0:
                self.ball.velocity[1] = -self.ball.velocity[1]     
            # Ball-paddle collision detection
            if pygame.sprite.collide_rect(self.ball, self.paddle1):
                self.ball.bounce()
            elif pygame.sprite.collide_rect(self.ball, self.paddle2):
                self.ball.bounce()

            # Drawing
            self.screen.fill((0, 0, 0))
            pygame.draw.line(self.screen, (255, 255, 255), (self.screen_width/2, 0), (self.screen_width/2, self.screen_height), 2)
            # self.paddle1.draw(self.screen)
            # self.paddle2.draw(self.screen)
            # self.ball.draw(self.screen)
            self.all_sprites_list.draw(self.screen) 
 
            WHITE = (255,255,255)
            #Display scores:
            font = pygame.font.Font(None, 74)
            text = font.render(str(self.scoreA), 1, WHITE)
            self.screen.blit(text, (250,10))
            text = font.render(str(self.scoreB), 1, WHITE)
            self.screen.blit(text, (420,10))
            pygame.display.flip()
            # --- Main event loop
            for event in pygame.event.get(): # User did something
                if event.type == pygame.QUIT: # If user clicked close
                    carryOn = False # Flag that we are done so we exit this loop
                elif event.type==pygame.KEYDOWN:
                        if event.key==pygame.K_x: #Pressing the x Key will quit the game
                            carryOn=False
            # Check for game over
            # if self.scoreA or self.scoreB >= 1000:
            #     pygame.quit()
            #     return
          
            # Tick the clock
            clock.tick(60)


