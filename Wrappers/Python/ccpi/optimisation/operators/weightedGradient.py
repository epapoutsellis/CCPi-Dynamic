
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 22:50:04 2019

@author: evangelos
"""

from ccpi.optimisation.operators import LinearOperator, ScaledOperator, Gradient
from ccpi.framework import ImageGeometry, BlockGeometry, BlockDataContainer
import numpy 
from ccpi.optimisation.operators import FiniteDiff, SparseFiniteDiff

#%%

class weightedGradient(Gradient):
            
    CORRELATION_SPACE = "Space"
    CORRELATION_SPACECHANNEL = "SpaceChannels"
    # Grad_order = ['channels', 'direction_z', 'direction_y', 'direction_x']
    # Grad_order = ['channels', 'direction_y', 'direction_x']
    # Grad_order = ['direction_z', 'direction_y', 'direction_x']
    # Grad_order = ['channels', 'direction_z', 'direction_y', 'direction_x']
    def __init__(self, gm_domain, bnd_cond = 'Neumann', **kwargs):
        
        super(weightedGradient, self).__init__(gm_domain) 
                
        self.gm_domain = gm_domain # Domain of Grad Operator
        
        self.correlation = kwargs.get('correlation', weightedGradient.CORRELATION_SPACE)
        
        if self.correlation==weightedGradient.CORRELATION_SPACE:
            if self.gm_domain.channels>1:
                self.gm_range = BlockGeometry(*[self.gm_domain for _ in range(self.gm_domain.length-1)] )
                if self.gm_domain.length == 4:
                    # 3D + Channel
                    # expected Grad_order = ['channels', 'direction_z', 'direction_y', 'direction_x']
                    expected_order = [ImageGeometry.CHANNEL, ImageGeometry.VERTICAL, ImageGeometry.HORIZONTAL_Y, ImageGeometry.HORIZONTAL_X]
                else:
                    # 2D + Channel
                    # expected Grad_order = ['channels', 'direction_y', 'direction_x']
                    expected_order = [ImageGeometry.CHANNEL, ImageGeometry.HORIZONTAL_Y, ImageGeometry.HORIZONTAL_X]
                order = self.gm_domain.get_order_by_label(self.gm_domain.dimension_labels, expected_order)
                self.ind = order[1:]
                #self.ind = numpy.arange(1,self.gm_domain.length)
            else:
                # no channel info
                self.gm_range = BlockGeometry(*[self.gm_domain for _ in range(self.gm_domain.length) ] )
                if self.gm_domain.length == 3:
                    # 3D
                    # expected Grad_order = ['direction_z', 'direction_y', 'direction_x']
                    expected_order = [ImageGeometry.VERTICAL, ImageGeometry.HORIZONTAL_Y, ImageGeometry.HORIZONTAL_X]
                else:
                    # 2D
                    expected_order = [ImageGeometry.HORIZONTAL_Y, ImageGeometry.HORIZONTAL_X]    
                self.ind = self.gm_domain.get_order_by_label(self.gm_domain.dimension_labels, expected_order)
                # self.ind = numpy.arange(self.gm_domain.length)
        elif self.correlation==weightedGradient.CORRELATION_SPACECHANNEL:
            if self.gm_domain.channels>1:
                self.gm_range = BlockGeometry(*[self.gm_domain for _ in range(self.gm_domain.length)])
                self.ind = range(self.gm_domain.length)
            else:
                raise ValueError('No channels to correlate')
         
        self.bnd_cond = bnd_cond 
        
        self.weight = kwargs.get('weight', [1] * self.gm_range.shape[0])
        
        if len(self.weight)>self.gm_range.shape[0]:
            raise ValueError('No channels to correlate')
            
            
        # Call FiniteDiff operator
        
        self.FD = FiniteDiff(self.gm_domain, direction = 0, bnd_cond = self.bnd_cond)
                                                         
        
    def direct(self, x, out=None):
        
                
        if out is not None:
            
            for i in range(self.gm_range.shape[0]):
                self.FD.direction = self.ind[i]
                self.FD.direct(self.weight[i] * x, out = out[i])
        else:
            tmp = self.gm_range.allocate()        
            for i in range(tmp.shape[0]):
                self.FD.direction=self.ind[i]
                tmp.get_item(i).fill( self.weight[i] * self.FD.direct(x))
            return tmp    
        
    def adjoint(self, x, out=None):
        
        if out is not None:

            tmp = self.gm_domain.allocate()            
            for i in range(x.shape[0]):
                self.FD.direction=self.ind[i] 
                self.FD.adjoint(x.get_item(i), out = tmp)
                if i == 0:
                    out.fill(tmp) 
                    out *= self.weight[i]
                else:
                    out += self.weight[i] * tmp
        else:            
            tmp = self.gm_domain.allocate()
            for i in range(x.shape[0]):
                self.FD.direction=self.ind[i]
                tmp += self.weight[i] * self.FD.adjoint(x.get_item(i))
            return tmp    
            
                                       
    def norm(self, **kwargs):
        
        return 2 * numpy.sqrt(numpy.sum(numpy.array(self.weight)**2))
    
    def __rmul__(self, scalar):
        return ScaledOperator(self, scalar) 
    
    ###########################################################################
    ###############  For preconditioning ######################################
    ###########################################################################
    def matrix(self):
        
        tmp = self.gm_range.allocate()
        
        mat = []
        for i in range(tmp.shape[0]):
            
            spMat = SparseFiniteDiff(self.gm_domain, direction=self.ind[i], bnd_cond=self.bnd_cond)
            mat.append(spMat.matrix())
    
        return BlockDataContainer(*mat)    


    def sum_abs_col(self):
        
        tmp = self.gm_range.allocate()
        res = self.gm_domain.allocate()
        for i in range(tmp.shape[0]):
            spMat = SparseFiniteDiff(self.gm_domain, direction=self.ind[i], bnd_cond=self.bnd_cond)
            res += spMat.sum_abs_row()
        return res
    
    def sum_abs_row(self):
        
        tmp = self.gm_range.allocate()
        res = []
        for i in range(tmp.shape[0]):
            spMat = SparseFiniteDiff(self.gm_domain, direction=self.ind[i], bnd_cond=self.bnd_cond)
            res.append(spMat.sum_abs_col())
        return BlockDataContainer(*res)
   
       
if __name__ == '__main__':
    
    
    from ccpi.optimisation.operators import Identity, BlockOperator, Gradient
    
    
    M, N = 2, 3
#    ig = ImageGeometry(M, N)
#    arr = ig.allocate('random_int', seed=10)
#    
#    # check direct of Gradient and sparse matrix
#    G = weightedGradient(ig, weight=[1,2])
#    G1 = Gradient(ig)
#    
#    z = G.direct(arr).get_item(0).as_array()
#    z1 = G1.direct(arr).get_item(0).as_array()
#    
#    print(z)
#    print(z1)
#    
#    print(G.norm())
#    print(G1.norm())
    
    
    alpha = [5,2,1]
    norm_alpha = numpy.sqrt(numpy.sum(numpy.array(alpha)**2))
    
    
    ig1 = ImageGeometry(M, N, channels = 2)
    op1 = weightedGradient(ig1,correlation='SpaceChannels', weight=alpha)
    
    
    outDirect = op1.range_geometry().allocate()
    outAdjoint = op1.domain_geometry().allocate()
    
    u = ig1.allocate('random_int')
    w = op1.range_geometry().allocate('random_int')
    
    op1.direct(u, outDirect)
    op1.adjoint(w, outAdjoint)
    
    res1 = op1.direct(u)
    res2 = op1.adjoint(w)
                      
    LHS = (res1 * w).sum()
    RHS = (u * res2).sum()
    
    LHS1 = (outDirect * w).sum()
    RHS1 = (u * outAdjoint).sum()
    
    print(LHS, RHS)
    
    print(LHS1, RHS1)    
    
    

    
    
    
    
    
    
    
#    ig2 = ImageGeometry(M, N)
#    
#    op1 = weightedGradient(ig1,correlation='SpaceChannels', weight=[0,1,1])    
#    op2 = weightedGradient(ig2,correlation='Space') 
#    
#    x1 = ig1.allocate('random_int', seed=10)
#    x2 = ig2.allocate('random_int', seed=10)
#    
#    
#    outDirect_SC = op1.range_geometry().allocate()
#    outAdjoint_SC = op1.domain_geometry().allocate()
#    outDirect_S = op2.range_geometry().allocate()
#    
#    op1.direct(x1, out=outDirect_SC)
#    op2.direct(x2, out=outDirect_S)
#        
#    print(outDirect_SC[0].as_array()) 
#    print(outDirect_SC[1].as_array()) 
#    print(outDirect_SC[2].as_array()) 
#    
#    print(outDirect_S[0].as_array()) 
#    print(outDirect_S[1].as_array()) 
#    
#    
#    # check adjoint
#    
#    y = op1.range_geometry().allocate('random_int')
#    
#    DC = FiniteDiff(ig1, direction = 0)
#    DY = FiniteDiff(ig1, direction = 1)
#    DX = FiniteDiff(ig1, direction = 2)
#    
#    a1, a2, a3 = op1.weight
#    
#    res = a1 * DC.adjoint(y.get_item(0)) + \
#          a2 * DY.adjoint(y.get_item(1)) + \
#          a3 * DX.adjoint(y.get_item(2))
#          
#    print(res.as_array())          
#    
#    op1.adjoint(y, out = outAdjoint_SC)
    
    
    
    
    
    
    

    
#    print(op2.direct(x2)[0].as_array())
#    print(op2.direct(x2)[1].as_array())
        
#    y1 = op1.range_geometry().allocate('random_int', seed=10)
#    y2 = op2.range_geometry().allocate('random_int', seed=10)
#    
##    y2 = BlockDataContainer(y1.get_item(1)[0], y1.get_item(1)[1])
##    y2.get_item(0) = y1.get_item(1)
#    
#    print('########################')
#    print(op1.adjoint(y1).as_array())
#    
#    print(op1.norm())
#    print(op2.norm())
#    
#    
#    
#    outDirect = op1.range_geometry().allocate()
#    op1.direct(x1, out=outDirect)
#    
#    
#    
#    
#    
#    
#    print('########################')
#    print(outDirect.get_item(0).as_array())
#    print(outDirect.get_item(1).as_array())
#    print(outDirect.get_item(2).as_array())
#    
#    res = G.range_geometry().allocate()
#    res1 = G.domain_geometry().allocate()
#    G.direct(u, out = res)          
#    G.adjoint(w, out = res1)    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    