"""
Script to test if Theano works on CPU or GPU.
This program computes exp() of a group of random numbers.
Includes source code from Theano
Author: Suhas Srinivasan
Date Created: 12/20/2018
Python Version: 2.7
"""

import numpy
from theano import function, config, shared, tensor
import time

vlen = 10 * 30 * 768  # 10 x # cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], tensor.exp(x))
print(f.maker.fgraph.toposort())
t0 = time.time()
for i in range(iters):
    r = f()
t1 = time.time()
print 'Looping %d times took %f seconds' % (iters, t1 - t0)
print 'Result is %s' % (r,)
if numpy.any([isinstance(x.op, tensor.Elemwise) and
              ('Gpu' not in type(x.op).__name__)
              for x in f.maker.fgraph.toposort()]):
    print 'Used the CPU'
else:
    print 'Used the GPU'
