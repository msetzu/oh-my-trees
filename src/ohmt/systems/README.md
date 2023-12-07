# Systems
Systems generalize the notion of `Hyperplane` to organized collections of `Hyperplane`s, i.e., systems.

## `BinaryPath`
A `BinaryPath` is a ordered sequence, i.e., a List, of Hyperplanes.
As such, it can be accessed by index, and also `hash`ed and directly compared for equality.
Like `Hyperplane`s, it also provides a `__call__` interface to directly compute what data lies within the given `Path`

```python
from planes import OHyperplane
from systems import BinaryPath
import numpy


x = numpy.array([[5., 2., 1.],      # 5*1 -0.5*2 <= 10
                 [22., 20., 0.]])   # 22*1 -0.5*21 <= 10 
hyperplane_1 = OHyperplane(numpy.array([1., -0.5, 0.]), 10.)  # 1*x1 -0.5*x2 <= 10.
hyperplane_2 = OHyperplane(numpy.array([-2., +0.5, 1.]), 10.)  # 1*x1 -0.5*x2 <= 10.
path = BinaryPath(hyperplane_1, hyperplane_2)

path(x)  # array([ True, False])
```

As sequences of flippable `Hyperplane`s, `BinaryPath`s can be flipped as well, either in toto:
```python
negated_path = ~path
negated_path(x) # array([ False, True])
```
or with a binary mask:
```python
path.flip([True, False])
```