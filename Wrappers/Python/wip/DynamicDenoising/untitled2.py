from ccpi.framework import ImageData, ImageGeometry, AcquisitionGeometry, AcquisitionData

import numpy as np 
import numpy                          
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import PDHG

from ccpi.optimisation.operators import BlockOperator, weightedGradient
from ccpi.optimisation.functions import ZeroFunction, L2NormSquared, \
                      MixedL21Norm, BlockFunction
                      
from ccpi.astra.operators import AstraProjectorMC 
import scipy.io

#%%
mat = scipy.io.loadmat('/Users/evangelos/Documents/MATLAB/Dynamic_reconstruction/data/3shapes.mat')
data = ImageData(np.swapaxes(mat['img'],0,2))
C, M, N = data.shape

ig = ImageGeometry(voxel_num_x=N, voxel_num_y=M, channels=C)

#%%

detectors = 100
angles = np.linspace(0,np.pi,100)

ag = AcquisitionGeometry('parallel','2D', angles, detectors, channels = ig.channels)
Aop = AstraProjectorMC(ig, ag, 'cpu')
sin = Aop.direct(data)

#scale = 2
#n1 = scale * np.random.poisson(sin.as_array()/scale)
#noisy_data = AcquisitionData(n1, ag)

noisy_data = sin + AcquisitionData(np.random.normal(0, 3, sin.shape))


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
weight = [0,1,1]
alpha = 50

# Create operators
#op1 = Gradient(ig)
op1 = weightedGradient(ig,correlation='SpaceChannels',\
                           weight=weight)
op2 = Aop

# Create BlockOperator
operator = BlockOperator(op1, op2, shape=(2,1) ) 

# Create functions
      
f1 = alpha * MixedL21Norm()
f2 = L2NormSquared(b=noisy_data)    
f = BlockFunction(f1, f2)  
                                      
g = ZeroFunction()
    
# Compute operator Norm
normK = operator.norm()

# Primal & dual stepsizes
sigma = 1
tau = 1/(sigma*normK**2)


# Setup and run the PDHG algorithm
pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma)
pdhg.max_iteration = 2000
pdhg.update_objective_interval = 200
pdhg.run(2000)

#%%






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


#plt.subplot(3,2,2)
#plt.imshow(noisy_data.as_array()[1])
#plt.title('Noisy Data')
#plt.colorbar()
#plt.subplot(3,2,3)
#plt.imshow(pdhg.get_output().as_array()[1])
#plt.title('TV Reconstruction')
#plt.colorbar()
plt.show()

