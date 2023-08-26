import pygame
from ball import Ball

class Paddle(pygame.sprite.Sprite):
    '''This class represents a paddle. It derives from the "Sprite" class in Pygame'''
 
    def __init__(self, color=(255,255,255), width=10, height=100):
        # Call the parent class (Sprite) constructor
        super().__init__()
        # Pass in the color of the Paddle, its width and height.
        # Set the background color and set it to be transparent
        BLACK = (0,20,0)
        self.image = pygame.Surface([width, height])
        self.image.fill(BLACK)
        self.image.set_colorkey(BLACK)
        # Draw the paddle (a rectangle!)
        pygame.draw.rect(self.image, color, [0, 0, width, height])
        # Fetch the rectangle object that has the dimensions of the image.
        self.rect = self.image.get_rect()
        
    def move(self,pixles=5):
        """ Implements moving logic """
        self.rect.y += pixles
        #Check that you are not going too far (off the screen)
        if self.rect.y < 0:
          self.rect.y = 0
        elif self.rect.y > 400:
          self.rect.y = 400
