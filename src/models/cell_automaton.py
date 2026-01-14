import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d as convolve2d
import matplotlib.animation as animation

class CellularAutomaton:
    def __init__(self, width, height, a, beta, kappa, threshold):
        self.width = width
        self.height = height
        self.a = a
        self.beta = beta
        self.threshold = threshold
        self.kappa = kappa

        self.bact_grid = np.zeros((height, width))
        self.grid = np.zeros((height, width))
        self.grid[height//4:3*height//4, width//4:3*width//4] = 1
        self.intensity_grid = np.zeros((height, width))
        self.conc_grid = np.zeros((height, width))
        self.conc_grid[height//4:3*height//4, width//4:3*width//4] = 1

    def randomize(self):
        self.grid = np.random.choice([0, 1], size=(self.height, self.width))

    def conc_field_update(self):
        conv_matrix = np.full((3, 3), self.kappa)
        conv_matrix[1,1] = 1
        self.conc_grid = self.conc_grid*(1 - self.a) + convolve2d(self.intensity_grid, conv_matrix, mode='same', boundary='wrap')*self.beta

    def inte_field_update(self):
        self.intensity_grid = 1/(1 + np.exp(- (self.conc_grid)/ self.threshold))
        
    def update(self):
        new_grid = self.grid.copy()

        self.conc_field_update()
        self.inte_field_update()

        self.grid = new_grid

    def draw(self):
        plt.imshow(self.grid, cmap='binary')
        plt.axis('off')
        plt.show()

    def animate(self, frames=10):
        fig = plt.figure()
        ims = []
        for _ in range(frames):
            self.update()
            im = plt.imshow(self.grid, cmap='binary', animated=True)
            ims.append([im])
        ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True)
        plt.show()
