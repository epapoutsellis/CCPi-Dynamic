# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library developed by
#   Visual Analytics and Imaging System Group of the Science Technology
#   Facilities Council, STFC

#   Copyright 2018-2019 STFC,  University of Manchester

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

""" 

Total Variation Denoising using PDHG algorithm:

             min_{x} max_{y} < K x, y > + g(x) - f^{*}(y) 


Problem:     min_x, x>0  \alpha * ||\nabla x||_{1} + \int x - g * log(x)

             \nabla: Gradient operator              
             g: Noisy Data with Poisson Noise
             \alpha: Regularization parameter
             
             Method = 0:  K = [ \nabla,
                                 Identity]
                                                                    
             Method = 1:  K = \nabla    
             
             
"""

from ccpi.framework import ImageData, ImageGeometry

import numpy as np 
import numpy                          
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import PDHG

from ccpi.optimisation.operators import BlockOperator, Identity, Gradient
from ccpi.optimisation.functions import ZeroFunction, KullbackLeibler, \
                      MixedL21Norm, BlockFunction

from skimage.util import random_noise

# Create phantom for TV Poisson denoising
N = 100

data = np.zeros((N,N))
data[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
data[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 1
data = ImageData(data)
ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N)
ag = ig

# Create noisy data. Apply Poisson noise
n1 = random_noise(data.as_array(), mode = 'poisson', seed = 10)
noisy_data = ImageData(n1)

# Regularisation Parameter
alpha = 2

method = '1'

if method == '0':

    # Create operators
    op1 = Gradient(ig)
    op2 = Identity(ig, ag)

    # Create BlockOperator
    operator = BlockOperator(op1, op2, shape=(2,1) ) 

    # Create functions
      
    f1 = alpha * MixedL21Norm()
    f2 = KullbackLeibler(noisy_data)    
    f = BlockFunction(f1, f2)  
                                      
    g = ZeroFunction()
    
else:
    
    # Without the "Block Framework"
    operator = Gradient(ig)
    f =  alpha * MixedL21Norm()
    g =  KullbackLeibler(noisy_data)   
        
    
# Compute operator Norm
normK = operator.norm()

# Primal & dual stepsizes
sigma = 1
tau = 1/(sigma*normK**2)
opt = {'niter':2000, 'memopt': True}

# Setup and run the PDHG algorithm
pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma, memopt=True)
pdhg.max_iteration = 2000
pdhg.update_objective_interval = 50

def pdgap_objectives(niter, objective, solution):
    

     print( "{:04}/{:04} {:<5} {:.4f} {:<5} {:.4f} {:<5} {:.4f}".\
                      format(niter, pdhg.max_iteration,'', \
                             objective[0],'',\
                             objective[1],'',\
                             objective[2]))

pdhg.run(2000, callback = pdgap_objectives)


plt.figure(figsize=(15,15))
plt.subplot(3,1,1)
plt.imshow(data.as_array())
plt.title('Ground Truth')
plt.colorbar()
plt.subplot(3,1,2)
plt.imshow(noisy_data.as_array())
plt.title('Noisy Data')
plt.colorbar()
plt.subplot(3,1,3)
plt.imshow(pdhg.get_output().as_array())
plt.title('TV Reconstruction')
plt.colorbar()
plt.show()
## 
plt.plot(np.linspace(0,N,N), data.as_array()[int(N/2),:], label = 'GTruth')
plt.plot(np.linspace(0,N,N), pdhg.get_output().as_array()[int(N/2),:], label = 'TV reconstruction')
plt.legend()
plt.title('Middle Line Profiles')
plt.show()


#%% Check with CVX solution

from ccpi.optimisation.operators import SparseFiniteDiff

try:
    from cvxpy import *
    cvx_not_installable = True
except ImportError:
    cvx_not_installable = False


if cvx_not_installable:

    ##Construct problem    
    u1 = Variable(ig.shape)
    q = Variable()
    
    DY = SparseFiniteDiff(ig, direction=0, bnd_cond='Neumann')
    DX = SparseFiniteDiff(ig, direction=1, bnd_cond='Neumann')
    
    # Define Total Variation as a regulariser
    regulariser = alpha * sum(norm(vstack([DX.matrix() * vec(u1), DY.matrix() * vec(u1)]), 2, axis = 0))
    
    fidelity = sum( u1 - multiply(noisy_data.as_array(), log(u1)) )
    constraints = [q>= fidelity, u1>=0]
        
    solver = ECOS
    obj =  Minimize( regulariser +  q)
    prob = Problem(obj, constraints)
    result = prob.solve(verbose = True, solver = solver)

    
    diff_cvx = numpy.abs( pdhg.get_output().as_array() - u1.value )
        
    plt.figure(figsize=(15,15))
    plt.subplot(3,1,1)
    plt.imshow(pdhg.get_output().as_array())
    plt.title('PDHG solution')
    plt.colorbar()
    plt.subplot(3,1,2)
    plt.imshow(u1.value)
    plt.title('CVX solution')
    plt.colorbar()
    plt.subplot(3,1,3)
    plt.imshow(diff_cvx)
    plt.title('Difference')
    plt.colorbar()
    plt.show()    
    
    plt.plot(np.linspace(0,N,N), pdhg.get_output().as_array()[int(N/2),:], label = 'PDHG')
    plt.plot(np.linspace(0,N,N), u1.value[int(N/2),:], label = 'CVX')
    plt.legend()
    plt.title('Middle Line Profiles')
    plt.show()
            
    print('Primal Objective (CVX) {} '.format(obj.value))
    print('Primal Objective (PDHG) {} '.format(pdhg.objective[-1][0]))




