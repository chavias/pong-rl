import pygame
from random import randint
from numpy import sign

BLACK = (0,0,0)
WHITE = (255,255,255)

class Ball(pygame.sprite.Sprite):
    # This class represents the ball. It derives from the "Sprite" class.

    def __init__(self, width=10, height=10, color=WHITE):
        super().__init__()
        self.image = pygame.Surface([width,height])
        self.image.fill(BLACK)
        self.image.set_colorkey(BLACK)
 
        # Draw the ball (a rectangle!)
        pygame.draw.rect(self.image, color, [0, 0, width, height])
        
        self.velocity = [-1,-1] #[randint(4,8),randint(-8,8)]
        
        # Fetch the rectangle object that has the dimensions of the image.
        self.rect = self.image.get_rect()

    def velocityReset(self):
        self.velocity = [sign(self.velocity[0])*5,sign(self.velocity[1])*5] #[randint(4,8),randint(-8,8)]

    def update(self):
        self.rect.x += self.velocity[0]
        self.rect.y += self.velocity[1]
          
    def bounce(self):
        self.velocity[0] = -self.velocity[0] #- 0.1*self.velocity[0]
        self.velocity[1] = self.velocity[1] #+0.1*self.velocity[1]#randint(-8,8)