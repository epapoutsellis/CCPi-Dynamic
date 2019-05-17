from ccpi.framework import ImageData, ImageGeometry

import numpy as np 
import numpy                          
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import PDHG

from ccpi.optimisation.operators import weightedGradient
from ccpi.optimisation.functions import L2NormSquared, MixedL21Norm

import scipy.io

#%%
mat = scipy.io.loadmat('/Users/evangelos/Documents/MATLAB/NewDynamicReconstruction/data/Gaussian_noise/toy_dyn_4regions.mat')
data = ImageData(np.swapaxes(mat['img'],0,2))
C, M, N = data.shape


ig = ImageGeometry(voxel_num_x=N, voxel_num_y=M, channels=C)
ag = ig
#%%

# Create noisy data. Add Gaussian noise
np.random.seed(10)
noisy_data = ImageData( data.as_array() + np.random.normal(0, 0.05, size=ig.shape) )


#%%
# Show Ground Truth and Noisy Data
plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
plt.imshow(data.as_array()[1])
plt.title('Ground Truth')
plt.colorbar()
plt.subplot(2,1,2)
plt.imshow(noisy_data.as_array()[1])
plt.title('Noisy Data')
plt.colorbar()
plt.show()

#%%

# Regularisation Parameter
alpha = 1
alpha_space = 1
alpha_time = 0.1
weight = [alpha_time] + [alpha_space]*2

operator = weightedGradient(ig,correlation='SpaceChannels',weight=weight)
f =  alpha * MixedL21Norm()
g =  0.5 * L2NormSquared(b = noisy_data)
         
# Compute Operator Norm

normK = operator.norm()

# Primal & Dual stepsizes
sigma = 10
tau = 1/(sigma*normK**2)

# Setup and Run the PDHG algorithm
pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma)
pdhg.max_iteration = 6000
pdhg.update_objective_interval = 500
pdhg.run(3000, verbose=True)


#%%
# Show Results
plt.figure(figsize=(10,10))

plt.subplot(2,3,1)
plt.imshow(data.as_array()[1])

plt.subplot(2,3,2)
plt.imshow(data.as_array()[2])

plt.subplot(2,3,3)
plt.imshow(data.as_array()[3])

plt.subplot(2,3,4)
plt.imshow(pdhg.get_output().as_array()[1])

plt.subplot(2,3,5)
plt.imshow(pdhg.get_output().as_array()[2])

plt.subplot(2,3,6)
plt.imshow(pdhg.get_output().as_array()[3])

plt.show()

#%%

import matplotlib.animation as animation


fig = plt.figure()

ims = []
for i in range(data.shape[0]):
    
    im = plt.imshow(pdhg.get_output().as_array()[i], animated=True)
    ims.append([im])
    
ani = animation.ArtistAnimation(fig, ims, interval=500,
                                repeat_delay=10)    

#plt.show()


#ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True,
#                                repeat_delay=1000)
ani.save('sample.mp4', writer='ffmpeg')






