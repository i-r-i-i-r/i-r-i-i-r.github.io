

import numpy as np
import matplotlib.pyplot as plt

class Particles:
    def __init__(self, N_particle, N_time, scale):
        pos = np.random.rand(N_particle, 2, 1)*scale
        self.position = np.concatenate([pos for i in range(N_time)], axis=2)
        self.N_time = N_time
        self.scale = scale
        self.N_particle = N_particle
    
    def update_pos(self):
        pos_new = np.zeros_like(self.position)
        pos_new[:,:,0] = self.position[:,:,0].copy() + np.random.randn(self.N_particle,2)
        pos_new[:,:,0] = pos_new[:,:,0] - (pos_new[:,:,0]>self.scale)*self.scale + (pos_new[:,:,0]<0.)*self.scale
        pos_new[:,:,1:]= self.position[:,:,:-1].copy()
        
        self.position = pos_new.copy()
    
    def plot_pos(self):
        alpha_list = [(N_time-1-i)/N_time for i in range(N_time)]
        for i in range(self.N_time-1, -1, -1):
            plt.plot(self.position[:,0,i], self.position[:,1,i], "o", color="r", alpha = alpha_list[i])
        plt.xlim(0, self.scale)
        plt.ylim(0, self.scale)
        plt.show()



temp=25


N_particle = 3
N_time     = 4
scale = 5
p = Particles(N_particle, N_time, scale)
for i in range(60):
    p.plot_pos()
    p.update_pos()




