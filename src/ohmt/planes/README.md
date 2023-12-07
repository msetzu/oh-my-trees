# Hyperplanes
`Hyperplane` is an abstract class representing hyperplanes, and is implemented by
`APHyperplane` and `OHyperplane`, implementing axis-parallel and oblique hyperplanes, respectively.
Both can be:
- copied
- deep-copied
- hashed
- inverted
- compared for equality
- accessed through standard array notation, e.g., `hyperplane[feature]` and `hyperplane[feature] = 0.`
- iterated over, yielding their values, coefficients, e.g., `for coefficient in ohyperplane`
- invoked to know whether some data lies within the hyperplane, e.g., `hyperplane(x)`

## Basic operations
The `OHyperplane` class support basic ring operations, which are defined **only on hyperplanes of the same size**.
Applying them to `Hyperplanes`s of different sizes will raise an exception.

**All mathematical operations are stateless**, that is, they **do not** modify the original `Hyperplane`, rather they return a copy
on which the operation is performed.
```python
from planes import OHyperplane

import numpy

hyperplane_1 = OHyperplane(numpy.array([1., -0.5, 0.]), 10.)  # 1*x1 -0.5*x2 <= 10.
hyperplane_2 = OHyperplane(numpy.array([-2., +0.5, 1.]), 10.)  # 1*x1 -0.5*x2 <= 10.

sum_hyperplanes = hyperplane_1 + hyperplane_2   # sums coefficients and bounds
sub_hyperplanes = hyperplane_1 - hyperplane_2   # subtracts coefficients and bounds
mul_hyperplanes = hyperplane_1 * hyperplane_2   # multiplies coefficients and bounds
inverted_hyperplane = ~hyperplane_1             # negates coefficients and bounds
negated_hyperplane = -hyperplane_1              # negates coefficients and bounds
```
Moreover, we can induce an oblique hyperplane from an axis-parallel one with `OHyperplane.from_aphyperplane`:
```python
from planes import OHyperplane, APHyperplane

import numpy


ohyperplane = OHyperplane(numpy.array([1., -0.5, 0.]), 10.)  # 1*x1 -0.5*x2 <= 10.
aphyperplane = APHyperplane(1, 10., numpy.inf)  # x[1] >= 10.

converted_ohyperplane = OHyperplane.from_aphyperplane(aphyperplane,
                                                      dimensionality=ohyperplane.coefficients.size)
print(converted_ohyperplane)
```


`APHyperplane`s are more limited, as they can only be inverted:
```python
from planes import APHyperplane
import numpy

aphyperplane = APHyperplane(1, 10., numpy.inf)  # x[1] >= 10.
print(aphyperplane)
```

## Data operations
`Hyperplane`s support a functional-like use to split data, that is, we can invoke them directly to see if some data `x`
lies within the space spanned by the hyperplane or not:

```python
from planes import OHyperplane, APHyperplane

import numpy


ohyperplane = OHyperplane(numpy.array([1., -0.5, 0.]), 10.)  # 1*x1 -0.5*x2 <= 10.
aphyperplane = APHyperplane(1, 10., 30.)  # x[1] in [10., 30.)

x = numpy.array([[5., 2., 1.],      # 5*1 -0.5*2 <= 10
                 [22., 20., 0.]])   # 22*1 -0.5*21 <= 10
print(ohyperplane(x))
# [ True False]
print(aphyperplane(x))
# [ False True]
```
We can also "flip" hyperplanes by inverting them: now all instances that were covered by the hyperplane are not, and
vice-versa:
```python
ohyperplane = OHyperplane(numpy.array([1., -0.5, 0.]), 10.)  # 1*x1 -0.5*x2 <= 10.
aphyperplane = APHyperplane(1, 10., 30.)  # x[1] in [10., 30.)

x = numpy.array([[5., 2., 1.],      # 5*1 -0.5*2 <= 10
                 [22., 20., 0.]])   # 22*1 -0.5*21 <= 10
print(ohyperplane(x))
# [ True False]
print(~ohyperplane(x))
# [ False True]
```
Double-bounded `APHyperplane`s (with a non-infinite lower and upper bound) can't be inverted since in order to do so
we would need two `APHyperplane`s, thus trying to invert them would result in an exception:
```python
aphyperplane = APHyperplane(1, 10., 30.)  # x[1] in [10., 30.)
single_bounded_aphyperplane = APHyperplane(1, 10., +numpy.inf)  # x[1] in [10., inf)

x = numpy.array([[5., 2., 1.],      # 5*1 -0.5*2 <= 10
                 [22., 20., 0.]])   # 22*1 -0.5*21 <= 10

print(aphyperplane(x))
# [False True]
print((~aphyperplane)(x))
# ...
# ValueError: Can't invert double-bounded APHyperplane: (10.0, 30.0)

print(single_bounded_aphyperplane(x))
# [False  True]
print(~single_bounded_aphyperplane(x))
# [True  False]
```