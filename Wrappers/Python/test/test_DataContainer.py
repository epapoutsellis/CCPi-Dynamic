# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 15:08:21 2019

@author: ofn77899
"""
import sys
import unittest
import numpy
from ccpi.framework import DataContainer
from ccpi.framework import ImageData
from ccpi.framework import AcquisitionData
from ccpi.framework import ImageGeometry
from ccpi.framework import AcquisitionGeometry
from timeit import default_timer as timer


def dt(steps):
    return steps[-1] - steps[-2]
def aid(x):
    # This function returns the memory
    # block address of an array.
    return x.__array_interface__['data'][0]



class TestDataContainer(unittest.TestCase):
    def create_simple_ImageData(self):
        N = 64
        ig = ImageGeometry(voxel_num_x=N, voxel_num_y=N)
        Phantom = ImageData(geometry=ig)

        x = Phantom.as_array()

        x[int(round(N/4)):int(round(3*N/4)),
          int(round(N/4)):int(round(3*N/4))] = 0.5
        x[int(round(N/8)):int(round(7*N/8)),
          int(round(3*N/8)):int(round(5*N/8))] = 1

        return (ig, Phantom)

    def create_DataContainer(self, X,Y,Z, value=1):
        steps = [timer()]
        a = value * numpy.ones((X, Y, Z), dtype='float32')
        steps.append(timer())
        t0 = dt(steps)
        #print("a refcount " , sys.getrefcount(a))
        ds = DataContainer(a, False, ['X', 'Y', 'Z'])
        return ds

    def test_creation_nocopy(self):
        shape = (2, 3, 4, 5)
        size = shape[0]
        for i in range(1, len(shape)):
            size = size * shape[i]
        #print("a refcount " , sys.getrefcount(a))
        a = numpy.asarray([i for i in range(size)])
        #print("a refcount " , sys.getrefcount(a))
        a = numpy.reshape(a, shape)
        #print("a refcount " , sys.getrefcount(a))
        ds = DataContainer(a, False, ['X', 'Y', 'Z', 'W'])
        #print("a refcount " , sys.getrefcount(a))
        self.assertEqual(sys.getrefcount(a), 3)
        self.assertEqual(ds.dimension_labels, {0: 'X', 1: 'Y', 2: 'Z', 3: 'W'})

    def testGb_creation_nocopy(self):
        X, Y, Z = 512, 512, 512
        X, Y, Z = 256, 512, 512
        steps = [timer()]
        a = numpy.ones((X, Y, Z), dtype='float32')
        steps.append(timer())
        t0 = dt(steps)
        print("test clone")
        #print("a refcount " , sys.getrefcount(a))
        ds = DataContainer(a, False, ['X', 'Y', 'Z'])
        #print("a refcount " , sys.getrefcount(a))
        self.assertEqual(sys.getrefcount(a), 3)
        ds1 = ds.copy()
        self.assertNotEqual(aid(ds.as_array()), aid(ds1.as_array()))
        ds1 = ds.clone()
        self.assertNotEqual(aid(ds.as_array()), aid(ds1.as_array()))

    def testInlineAlgebra(self):
        print("Test Inline Algebra")
        X, Y, Z = 1024, 512, 512
        X, Y, Z = 256, 512, 512
        steps = [timer()]
        a = numpy.ones((X, Y, Z), dtype='float32')
        steps.append(timer())
        t0 = dt(steps)
        print(t0)
        #print("a refcount " , sys.getrefcount(a))
        ds = DataContainer(a, False, ['X', 'Y', 'Z'])
        #ds.__iadd__( 2 )
        ds += 2
        steps.append(timer())
        print(dt(steps))
        self.assertEqual(ds.as_array()[0][0][0], 3.)
        #ds.__isub__( 2 )
        ds -= 2
        steps.append(timer())
        print(dt(steps))
        self.assertEqual(ds.as_array()[0][0][0], 1.)
        #ds.__imul__( 2 )
        ds *= 2
        steps.append(timer())
        print(dt(steps))
        self.assertEqual(ds.as_array()[0][0][0], 2.)
        #ds.__idiv__( 2 )
        ds /= 2
        steps.append(timer())
        print(dt(steps))
        self.assertEqual(ds.as_array()[0][0][0], 1.)

        ds1 = ds.copy()
        #ds1.__iadd__( 1 )
        ds1 += 1
        #ds.__iadd__( ds1 )
        ds += ds1
        steps.append(timer())
        print(dt(steps))
        self.assertEqual(ds.as_array()[0][0][0], 3.)
        #ds.__isub__( ds1 )
        ds -= ds1
        steps.append(timer())
        print(dt(steps))
        self.assertEqual(ds.as_array()[0][0][0], 1.)
        #ds.__imul__( ds1 )
        ds *= ds1
        steps.append(timer())
        print(dt(steps))
        self.assertEqual(ds.as_array()[0][0][0], 2.)
        #ds.__idiv__( ds1 )
        ds /= ds1
        steps.append(timer())
        print(dt(steps))
        self.assertEqual(ds.as_array()[0][0][0], 1.)

    def test_unary_operations(self):
        print("Test unary operations")
        X, Y, Z = 1024, 512, 512
        X, Y, Z = 256, 512, 512
        steps = [timer()]
        a = -numpy.ones((X, Y, Z), dtype='float32')
        steps.append(timer())
        t0 = dt(steps)
        print(t0)
        #print("a refcount " , sys.getrefcount(a))
        ds = DataContainer(a, False, ['X', 'Y', 'Z'])

        ds.sign(out=ds)
        steps.append(timer())
        print(dt(steps))
        self.assertEqual(ds.as_array()[0][0][0], -1.)

        ds.abs(out=ds)
        steps.append(timer())
        print(dt(steps))
        self.assertEqual(ds.as_array()[0][0][0], 1.)

        ds.__imul__(2)
        ds.sqrt(out=ds)
        steps.append(timer())
        print(dt(steps))
        self.assertEqual(ds.as_array()[0][0][0],
                         numpy.sqrt(2., dtype='float32'))

    def test_binary_operations(self):
        self.binary_add()
        self.binary_subtract()
        self.binary_multiply()
        self.binary_divide()

    def binary_add(self):
        print("Test binary add")
        X, Y, Z = 512, 512, 512
        X, Y, Z = 256, 512, 512
        steps = [timer()]
        a = numpy.ones((X, Y, Z), dtype='float32')
        steps.append(timer())
        t0 = dt(steps)

        #print("a refcount " , sys.getrefcount(a))
        ds = DataContainer(a, False, ['X', 'Y', 'Z'])
        ds1 = ds.copy()

        steps.append(timer())
        ds.add(ds1, out=ds)
        steps.append(timer())
        t1 = dt(steps)
        print("ds.add(ds1, out=ds)", dt(steps))
        steps.append(timer())
        ds2 = ds.add(ds1)
        steps.append(timer())
        t2 = dt(steps)
        print("ds2 = ds.add(ds1)", dt(steps))

        self.assertLess(t1, t2)
        self.assertEqual(ds.as_array()[0][0][0], 2.)

        ds0 = ds
        ds0.add(2, out=ds0)
        steps.append(timer())
        print("ds0.add(2,out=ds0)", dt(steps), 3, ds0.as_array()[0][0][0])
        self.assertEqual(4., ds0.as_array()[0][0][0])

        dt1 = dt(steps)
        ds3 = ds0.add(2)
        steps.append(timer())
        print("ds3 = ds0.add(2)", dt(steps), 5, ds3.as_array()[0][0][0])
        dt2 = dt(steps)
        self.assertLess(dt1, dt2)

    def binary_subtract(self):
        print("Test binary subtract")
        X, Y, Z = 512, 512, 512
        steps = [timer()]
        a = numpy.ones((X, Y, Z), dtype='float32')
        steps.append(timer())
        t0 = dt(steps)

        #print("a refcount " , sys.getrefcount(a))
        ds = DataContainer(a, False, ['X', 'Y', 'Z'])
        ds1 = ds.copy()

        steps.append(timer())
        ds.subtract(ds1, out=ds)
        steps.append(timer())
        t1 = dt(steps)
        print("ds.subtract(ds1, out=ds)", dt(steps))
        self.assertEqual(0., ds.as_array()[0][0][0])

        steps.append(timer())
        ds2 = ds.subtract(ds1)
        self.assertEqual(-1., ds2.as_array()[0][0][0])

        steps.append(timer())
        t2 = dt(steps)
        print("ds2 = ds.subtract(ds1)", dt(steps))

        self.assertLess(t1, t2)

        del ds1
        ds0 = ds.copy()
        steps.append(timer())
        ds0.subtract(2, out=ds0)
        #ds0.__isub__( 2 )
        steps.append(timer())
        print("ds0.subtract(2,out=ds0)", dt(
            steps), -2., ds0.as_array()[0][0][0])
        self.assertEqual(-2., ds0.as_array()[0][0][0])

        dt1 = dt(steps)
        ds3 = ds0.subtract(2)
        steps.append(timer())
        print("ds3 = ds0.subtract(2)", dt(steps), 0., ds3.as_array()[0][0][0])
        dt2 = dt(steps)
        self.assertLess(dt1, dt2)
        self.assertEqual(-2., ds0.as_array()[0][0][0])
        self.assertEqual(-4., ds3.as_array()[0][0][0])

    def binary_multiply(self):
        print("Test binary multiply")
        X, Y, Z = 1024, 512, 512
        X, Y, Z = 256, 512, 512
        steps = [timer()]
        a = numpy.ones((X, Y, Z), dtype='float32')
        steps.append(timer())
        t0 = dt(steps)

        #print("a refcount " , sys.getrefcount(a))
        ds = DataContainer(a, False, ['X', 'Y', 'Z'])
        ds1 = ds.copy()

        steps.append(timer())
        ds.multiply(ds1, out=ds)
        steps.append(timer())
        t1 = dt(steps)
        print("ds.multiply(ds1, out=ds)", dt(steps))
        steps.append(timer())
        ds2 = ds.multiply(ds1)
        steps.append(timer())
        t2 = dt(steps)
        print("ds2 = ds.multiply(ds1)", dt(steps))

        self.assertLess(t1, t2)

        ds0 = ds
        ds0.multiply(2, out=ds0)
        steps.append(timer())
        print("ds0.multiply(2,out=ds0)", dt(
            steps), 2., ds0.as_array()[0][0][0])
        self.assertEqual(2., ds0.as_array()[0][0][0])

        dt1 = dt(steps)
        ds3 = ds0.multiply(2)
        steps.append(timer())
        print("ds3 = ds0.multiply(2)", dt(steps), 4., ds3.as_array()[0][0][0])
        dt2 = dt(steps)
        self.assertLess(dt1, dt2)
        self.assertEqual(4., ds3.as_array()[0][0][0])
        self.assertEqual(2., ds.as_array()[0][0][0])

    def binary_divide(self):
        print("Test binary divide")
        X, Y, Z = 1024, 512, 512
        X, Y, Z = 256, 512, 512
        steps = [timer()]
        a = numpy.ones((X, Y, Z), dtype='float32')
        steps.append(timer())
        t0 = dt(steps)

        #print("a refcount " , sys.getrefcount(a))
        ds = DataContainer(a, False, ['X', 'Y', 'Z'])
        ds1 = ds.copy()

        steps.append(timer())
        ds.divide(ds1, out=ds)
        steps.append(timer())
        t1 = dt(steps)
        print("ds.divide(ds1, out=ds)", dt(steps))
        steps.append(timer())
        ds2 = ds.divide(ds1)
        steps.append(timer())
        t2 = dt(steps)
        print("ds2 = ds.divide(ds1)", dt(steps))

        self.assertLess(t1, t2)
        self.assertEqual(ds.as_array()[0][0][0], 1.)

        ds0 = ds
        ds0.divide(2, out=ds0)
        steps.append(timer())
        print("ds0.divide(2,out=ds0)", dt(steps), 0.5, ds0.as_array()[0][0][0])
        self.assertEqual(0.5, ds0.as_array()[0][0][0])

        dt1 = dt(steps)
        ds3 = ds0.divide(2)
        steps.append(timer())
        print("ds3 = ds0.divide(2)", dt(steps), 0.25, ds3.as_array()[0][0][0])
        dt2 = dt(steps)
        self.assertLess(dt1, dt2)
        self.assertEqual(.25, ds3.as_array()[0][0][0])
        self.assertEqual(.5, ds.as_array()[0][0][0])

    def test_creation_copy(self):
        shape = (2, 3, 4, 5)
        size = shape[0]
        for i in range(1, len(shape)):
            size = size * shape[i]
        #print("a refcount " , sys.getrefcount(a))
        a = numpy.asarray([i for i in range(size)])
        #print("a refcount " , sys.getrefcount(a))
        a = numpy.reshape(a, shape)
        #print("a refcount " , sys.getrefcount(a))
        ds = DataContainer(a, True, ['X', 'Y', 'Z', 'W'])
        #print("a refcount " , sys.getrefcount(a))
        self.assertEqual(sys.getrefcount(a), 2)

    def test_subset(self):
        shape = (4, 3, 2)
        a = [i for i in range(2*3*4)]
        a = numpy.asarray(a)
        a = numpy.reshape(a, shape)
        ds = DataContainer(a, True, ['X', 'Y', 'Z'])
        sub = ds.subset(['X'])
        res = True
        try:
            numpy.testing.assert_array_equal(sub.as_array(),
                                             numpy.asarray([0, 6, 12, 18]))
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

        sub = ds.subset(['X'], Y=2, Z=0)
        res = True
        try:
            numpy.testing.assert_array_equal(sub.as_array(),
                                             numpy.asarray([4, 10, 16, 22]))
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

        sub = ds.subset(['Y'])
        try:
            numpy.testing.assert_array_equal(
                sub.as_array(), numpy.asarray([0, 2, 4]))
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

        sub = ds.subset(['Z'])
        try:
            numpy.testing.assert_array_equal(
                sub.as_array(), numpy.asarray([0, 1]))
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)
        sub = ds.subset(['Z'], X=1, Y=2)
        try:
            numpy.testing.assert_array_equal(
                sub.as_array(), numpy.asarray([10, 11]))
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

        print(a)
        sub = ds.subset(['X', 'Y'], Z=1)
        res = True
        try:
            numpy.testing.assert_array_equal(sub.as_array(),
                                             numpy.asarray([[1,  3,  5],
                                                            [7,  9, 11],
                                                            [13, 15, 17],
                                                            [19, 21, 23]]))
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)
        
    def test_dot(self):
        a0 = numpy.asarray([i for i in range(2*3*4)])
        a1 = numpy.asarray([2*i for i in range(2*3*4)])
        
                
        ds0 = DataContainer(numpy.reshape(a0,(2,3,4)))
        ds1 = DataContainer(numpy.reshape(a1,(2,3,4)))
        
        numpy.testing.assert_equal(ds0.dot(ds1), a0.dot(a1))
        
        a2 = numpy.asarray([2*i for i in range(2*3*5)])
        ds2 = DataContainer(numpy.reshape(a2,(2,3,5)))
        
        # it should fail if the shape is wrong
        try:
            ds2.dot(ds0)
            self.assertTrue(False)
        except ValueError as ve:
            self.assertTrue(True)
        
        

    def test_ImageData(self):
        # create ImageData from geometry
        vgeometry = ImageGeometry(voxel_num_x=4, voxel_num_y=3, channels=2)
        vol = ImageData(geometry=vgeometry)
        self.assertEqual(vol.shape, (2, 3, 4))

        vol1 = vol + 1
        self.assertNumpyArrayEqual(vol1.as_array(), numpy.ones(vol.shape))

        vol1 = vol - 1
        self.assertNumpyArrayEqual(vol1.as_array(), -numpy.ones(vol.shape))

        vol1 = 2 * (vol + 1)
        self.assertNumpyArrayEqual(vol1.as_array(), 2 * numpy.ones(vol.shape))

        vol1 = (vol + 1) / 2
        self.assertNumpyArrayEqual(vol1.as_array(), numpy.ones(vol.shape) / 2)

        vol1 = vol + 1
        self.assertEqual(vol1.sum(), 2*3*4)
        vol1 = (vol + 2) ** 2
        self.assertNumpyArrayEqual(vol1.as_array(), numpy.ones(vol.shape) * 4)

        self.assertEqual(vol.number_of_dimensions, 3)

    def test_AcquisitionData(self):
        sgeometry = AcquisitionGeometry(dimension=2, angles=numpy.linspace(0, 180, num=10),
                                        geom_type='parallel', pixel_num_v=3,
                                        pixel_num_h=5, channels=2)
        sino = AcquisitionData(geometry=sgeometry)
        self.assertEqual(sino.shape, (2, 10, 3, 5))
    def test_ImageGeometry_allocate(self):
        vgeometry = ImageGeometry(voxel_num_x=4, voxel_num_y=3, channels=2)
        image = vgeometry.allocate()
        self.assertEqual(0,image.as_array()[0][0][0])
        image = vgeometry.allocate(1)
        self.assertEqual(1,image.as_array()[0][0][0])
        default_order = ['channel' , 'horizontal_y' , 'horizontal_x']
        self.assertEqual(default_order[0], image.dimension_labels[0])
        self.assertEqual(default_order[1], image.dimension_labels[1])
        self.assertEqual(default_order[2], image.dimension_labels[2])
        order = [ 'horizontal_x' , 'horizontal_y', 'channel' ]
        image = vgeometry.allocate(0,dimension_labels=order)
        self.assertEqual(order[0], image.dimension_labels[0])
        self.assertEqual(order[1], image.dimension_labels[1])
        self.assertEqual(order[2], image.dimension_labels[2])
    def test_AcquisitionGeometry_allocate(self):
        ageometry = AcquisitionGeometry(dimension=2, angles=numpy.linspace(0, 180, num=10),
                                        geom_type='parallel', pixel_num_v=3,
                                        pixel_num_h=5, channels=2)
        sino = ageometry.allocate()
        shape = sino.shape
        print ("shape", shape)
        self.assertEqual(0,sino.as_array()[0][0][0][0])
        self.assertEqual(0,sino.as_array()[shape[0]-1][shape[1]-1][shape[2]-1][shape[3]-1])
        
        sino = ageometry.allocate(1)
        self.assertEqual(1,sino.as_array()[0][0][0][0])
        self.assertEqual(1,sino.as_array()[shape[0]-1][shape[1]-1][shape[2]-1][shape[3]-1])
        print (sino.dimension_labels, sino.shape, ageometry)
        
        default_order = ['channel' , ' angle' ,
                                          'vertical' , 'horizontal']
        self.assertEqual(default_order[0], sino.dimension_labels[0])
        self.assertEqual(default_order[1], sino.dimension_labels[1])
        self.assertEqual(default_order[2], sino.dimension_labels[2])
        self.assertEqual(default_order[3], sino.dimension_labels[3])
        order = ['vertical' , 'horizontal', 'channel' , 'angle' ]
        sino = ageometry.allocate(0,dimension_labels=order)
        print (sino.dimension_labels, sino.shape, ageometry)
        self.assertEqual(order[0], sino.dimension_labels[0])
        self.assertEqual(order[1], sino.dimension_labels[1])
        self.assertEqual(order[2], sino.dimension_labels[2])
        self.assertEqual(order[2], sino.dimension_labels[2])

    def assertNumpyArrayEqual(self, first, second):
        res = True
        try:
            numpy.testing.assert_array_equal(first, second)
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

    def assertNumpyArrayAlmostEqual(self, first, second, decimal=6):
        res = True
        try:
            numpy.testing.assert_array_almost_equal(first, second, decimal)
        except AssertionError as err:
            res = False
            print(err)
            print("expected " , second)
            print("actual " , first)

        self.assertTrue(res)
    def test_DataContainerChaining(self):
        dc = self.create_DataContainer(256,256,256,1)

        dc.add(9,out=dc)\
          .subtract(1,out=dc)
        self.assertEqual(1+9-1,dc.as_array().flatten()[0])
    def test_reduction(self):
        print ("test reductions")
        dc = self.create_DataContainer(2,2,2,value=1)
        sqnorm = dc.squared_norm()
        norm = dc.norm()
        self.assertEqual(sqnorm, 8.0)
        numpy.testing.assert_almost_equal(norm, numpy.sqrt(8.0), decimal=7)



if __name__ == '__main__':
    unittest.main()
 