from pygame.sprite import _Group
import enviroment as env
import pygame

BLACK = (0,0,0)
WHITE = (255,255,255)


class PaddelSprite(pygame.sprite.Sprite):
    def __init__(self, x, y, color=(255,255,255)):
        super().__init__()
        self.paddel = env.Paddle(x,y)
        # Pass in the color of the Paddle, its width and height.
        # Set the background color and set it to be transparent
        BLACK = (0,20,0)
        self.image = pygame.Surface([2, env.PADDLE_HEIGHT])
        self.image.fill(BLACK)
        self.image.set_colorkey(BLACK)
        # Draw the paddle (a rectangle!)
        pygame.draw.rect(
                        self.image,
                        color,
                        [0, self.paddel.y-env.PADDLE_HEIGHT/2, 2, env.PADDLE_HEIGHT]
                        )
        # Fetch the rectangle object that has the dimensions of the image.
        self.rect = self.image.get_rect()
        
class BallSprite(pygame.sprite.Sprite):
        def __init__(self, x, y, vx, vy) -> None:
            super().__init__()
            self.ball = env.Ball(x, y, vx, vy)
            self.image = pygame.Surface([width,height])
            self.image.fill(BLACK)
            self.image.set_colorkey(BLACK)
 
        # Draw the ball (a rectangle!)
        pygame.draw.rect(self.image, color, [0, 0, width, height])
        
        self.velocity = [5,5] #[randint(4,8),randint(-8,8)]
        
        # Fetch the rectangle object that has the dimensions of the image.
        self.rect = self.image.get_rect()
    
class PlayGame:
    def __init__(self) -> None:
        self.game = env.GameEngine
        self.Paddel_Sprite

    