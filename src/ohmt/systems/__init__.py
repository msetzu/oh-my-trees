from __future__ import annotations

import copy
from abc import abstractmethod
from typing import List, Tuple, Sequence

import numpy as numpy

from ohmt.planes import APHyperplane, OHyperplane, Hyperplane


class BinaryPath(List[Hyperplane]):
    """A path of hyperplanes, i.e., a sequence of hyperplanes which can be flipped, i.e., inverted.."""
    def __init__(self, *hyperplanes):
        super().__init__(*hyperplanes)

    def __hash__(self):
        return sum([hash(hyperplane) for hyperplane in self])

    def __eq__(self, other):
        if not isinstance(other, APBinaryPath) or len(self) != len(other):
            return False
        return all([hyperplane_this == hyperplane_other for hyperplane_this, hyperplane_other in zip(self, other)])

    def __call__(self, data: numpy.ndarray) -> bool | numpy.array:
        """Coverage vector of this System on the given `data`.

        Args:
            data: The data whose coverage to compute.

        Returns:
            True if the element satisfies the premise, False otherwise.
        """
        if data.ndim == 1:
            # single instance
            try:
                return all([hyperplane(data) for hyperplane in self])
            except ValueError as e:
                raise e
        else:
            return numpy.array([self(record) for record in data])

    def flip(self, mask: Sequence[bool]) -> BinaryPath:
        """Flip the hyperplanes according to the given mask.

        Args:
            mask: Binary mask of the same length of self: if position `i` is True, then self[i] flips.

        Returns:
            A new path with hyperplanes flipped according to the provided mask.
        """
        self_copy = copy.deepcopy(self)

        for i, (hyperplane, should_flip) in enumerate(zip(self_copy, mask)):
            if should_flip:
                self_copy[i] = ~self_copy[i]

        return self_copy

    @abstractmethod
    def system(self, **kwargs) -> Tuple[numpy.ndarray, numpy.array]:
        """Map this System to a numpy system, i.e. a pair (coefficient matrix, bounds vector).

        Returns:
            A coefficient matrix, and a bounds vector.
        """
        pass

    def __invert__(self):
        return self.flip([True] * len(self))


class APBinaryPath(BinaryPath, List[APHyperplane]):
    """System exclusively comprised of APHyperplanes."""
    def __init__(self, *hyperplanes: APHyperplane):
        super().__init__(hyperplanes)

    def __copy__(self):
        # shallow copying: premises are references
        return APBinaryPath(*[copy.copy(hyperplane) for hyperplane in self])

    def __deepcopy__(self, memodict):
        return APBinaryPath(*[copy.copy(hyperplane) for hyperplane in self])

    def system(self, **kwargs) -> Tuple[numpy.ndarray, numpy.array]:
        dimensionality = kwargs.get("dimensionality", max(hyperplane.axis for hyperplane in self))
        ohyperplanes = [OHyperplane.from_aphyperplane(hyperplane, dimensionality=dimensionality) for hyperplane in self]

        return OBinaryPath(*ohyperplanes).system()


class OBinaryPath(BinaryPath, List[OHyperplane]):
    """""System exclusively comprised of APHyperplanes."""
    def __init__(self, *hyperplanes: APHyperplane):
        super().__init__(hyperplanes)

    def __copy__(self):
        # shallow copying: premises are references
        return OBinaryPath(*[copy.copy(hyperplane) for hyperplane in self])

    def __deepcopy__(self, memodict):
        return OBinaryPath(*[copy.deepcopy(hyperplane) for hyperplane in self])

    def system(self, **kwargs) -> Tuple[numpy.ndarray, numpy.array]:
        coefficients = numpy.vstack([hyperplane.coefficients for hyperplane in self])
        bounds = numpy.vstack([hyperplane.bound for hyperplane in self])

        return coefficients, bounds
