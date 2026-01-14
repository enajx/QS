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
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2
        radius = min(height, width) // 4
        mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
        self.grid[mask] = 1
        self.intensity_grid = np.zeros((height, width, 2))
        self.conc_grid = np.full((height, width, 2), 0.001)
    
    def conc_field_update(self):
        # Diffusion kernel for concentration field
        diffusion_kernel = np.array([[self.a/8, self.a/8, self.a/8],
                                      [self.a/8, 1-self.a, self.a/8],
                                      [self.a/8, self.a/8, self.a/8]])
        
        # Production kernel for intensity field
        production_kernel = np.full((3, 3), self.kappa)
        production_kernel[1,1] = 1
        
        # Update: diffusion of concentration + production from intensity

        mask = self.intensity_grid[:,:,0] > self.intensity_grid[:,:,1]
        
        self.conc_grid[:,:,0] = convolve2d(self.conc_grid[:,:,0], diffusion_kernel, mode='same', boundary='fill') + \
                         convolve2d(self.intensity_grid[:,:,0]*mask, production_kernel, mode='same', boundary='fill')*self.beta
        self.conc_grid[:,:,1] = convolve2d(self.conc_grid[:,:,1], diffusion_kernel, mode='same', boundary='fill') + \
                         convolve2d(self.intensity_grid[:,:,1]*~mask, production_kernel, mode='same', boundary='fill')*self.beta

    def inte_field_update(self):
        self.intensity_grid[:,:,0] = self.grid/(1 + np.exp(- (convolve2d(self.conc_grid[:,:,0], np.ones((3,3)), mode='same', boundary='fill') - self.threshold)))
        self.intensity_grid[:,:,1] = self.grid/(1 + np.exp(- (convolve2d(self.conc_grid[:,:,1], np.ones((3,3)), mode='same', boundary='fill') - self.threshold)))
    
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
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        ims = [[], [], [], [], []]
        for _ in range(frames):
            self.update()
            im0 = axes[0].imshow(self.grid, cmap='binary', animated=True)
            im1 = axes[1].imshow(self.conc_grid[:,:,0], cmap='Reds', animated=True)
            im2 = axes[2].imshow(self.intensity_grid[:,:,0], cmap='Reds', animated=True)
            ims[0].append([im0])
            ims[1].append([im1])
            ims[2].append([im2])
            
            # Additional plots for the second concentration and intensity
            im3 = axes[3].imshow(self.conc_grid[:,:,1], cmap='Greens', animated=True)
            im4 = axes[4].imshow(self.intensity_grid[:,:,1], cmap='Greens', animated=True)
            ims[3].append([im3])
            ims[4].append([im4])

        axes[0].set_title('Grid')
        axes[1].set_title('Concentration RFP')
        axes[2].set_title('Activation RFP')
        axes[3].set_title('Concentration GFP')
        axes[4].set_title('Activation GFP')

        for ax in axes:
            ax.axis('off')
        
        ani = animation.ArtistAnimation(fig, [ims[0][i] + ims[1][i] + ims[2][i] + ims[3][i] + ims[4][i] for i in range(frames)], interval=500, blit=True)
        plt.tight_layout()
        plt.show()
        return ani  # Return the animation object
