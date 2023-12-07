from __future__ import annotations

import copy
from abc import abstractmethod, ABC
from typing import Union, List, Dict

import numpy


class Hyperplane(ABC):
    """A separating hyperplane: splits the given data in two sets."""
    @abstractmethod
    def __call__(self, *args, **kwargs):
        """Check whether the given array is within the premise, i.e., whether the premise covers the array or not."""
        pass


class APHyperplane(Hyperplane):
    """An axis-parallel hyperplane of the form:
        fi in [ai, bi]
    that defines a continuous subspace on dimension `fi` delimited by a lower bound `ai` (included)
    and an upper bound `bi` (excluded).
    """
    def __init__(self, feat: int, low: float = -numpy.inf, upp: float = +numpy.inf):
        """
        Args:
            feat: The feature on which the hyperplane is defined
            low: The lower bound of the hyperplane. For premises of the form X <= a, `low` is `-numpy.inf`.
                    Defaults to `-numpy.inf`.
            upp: The upper bound of the hyperplane. For premises of the form X > a, `low` is `+numpy.inf`.
                    Defaults to `+numpy.inf`.
        """
        self.axis = feat
        self.lower_bound = low
        self.upper_bound = upp

    def __hash__(self):
        return hash((self.axis, self.lower_bound, self.upper_bound))

    def __eq__(self, other):
        return isinstance(other, APHyperplane) and self.axis == other.axis and \
               self.lower_bound == other.lower_bound and self.upper_bound == other.upper_bound

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"{self.axis, self.lower_bound, self.upper_bound}"

    def __call__(self, *args, **kwargs) -> bool | numpy.array:
        """Check whether the given array is within the premise, i.e., whether the premise covers the array or not.

        Args:
            *args:
            **kwargs:

        Returns:
            True if the element satisfies the premise, False otherwise.
        """
        data = args[0]
        if not isinstance(data, numpy.ndarray):
            raise ValueError("Not a numpy.ndarray: {0}".format(str(type(data))))
        if data.ndim == 2 and data.shape[1] < self.axis or data.ndim == 1 and data.size < self.axis:
            raise ValueError(f"Wrong dimensionality: expected vector of size >= {self.axis}, {data.shape} found")

        # multiple records
        # TODO: la macarena
        if data.ndim == 2 and data.shape[1] > 0:
            return numpy.array([self(x) for x in data])
        # single record, two dimensions
        elif data.ndim == 2 and data.shape[0] == 1:
            return self.lower_bound < data[0, self.axis] <= self.upper_bound
        # single record, one dimension
        elif data.ndim == 1:
            return self.lower_bound < data[self.axis] <= self.upper_bound
        else:
            raise ValueError("Expected 2-dimensional or 1-dimensional array")

    def __invert__(self):
        # inversion requires to return a single APHyperplane, which cannot express double bounds, hence we
        # raise exceptions
        if not numpy.isinf(self.lower_bound) and not numpy.isinf(self.upper_bound):
            raise ValueError(f"Can't invert double-bounded APHyperplane: {self.lower_bound, self.upper_bound}")

        if numpy.isinf(self.lower_bound):
            # x <= beta
            return APHyperplane(self.axis,
                                self.upper_bound,
                                +numpy.inf)
        elif numpy.isinf(self.upper_bound):
            # x >= beta
            return APHyperplane(self.axis,
                                -numpy.inf,
                                self.lower_bound)

    def __copy__(self):
        return APHyperplane(self.axis, self.lower_bound, self.upper_bound)

    def __deepcopy__(self, memodict={}):
        return APHyperplane(self.axis, self.lower_bound, self.upper_bound)

    def json(self) -> Dict:
        return {
            "type": "parallel",
            "axis": self.axis,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound
        }

    @staticmethod
    def from_json(json_obj):
        return APHyperplane(json_obj["axis"], json_obj["lower_bound"], json_obj["upper_bound"])


class OHyperplane(Hyperplane):
    """Oblique hyperplane of the form:
        a1*f1 + a2*f2 + ... + am*fm <= b
    where a1, ..., am, and b are scalars. Alternatively, formulated as ax <= b.

    Attributes:
        coefficients (numpy.ndarray): Coefficients of the hyperplane (the a).
        bound (float): Bound of the hyperplane (the b).
    """

    def __init__(self, coefficients: Union[List[float], numpy.ndarray], bound: float):
        """An oblique (multivariate) premise.
        Args:
            coefficients: Coefficients a1, ..., am.
            bound: Bound scalar b.
        """
        self.coefficients = coefficients if isinstance(coefficients, numpy.ndarray) else numpy.array(coefficients)
        self.bound = bound
        self.dim = self.coefficients.size

    def __hash__(self):
        return hash(str(self.coefficients.tolist() + [self.bound]))

    def __eq__(self, other):
        return isinstance(other, OHyperplane) and \
            (self.coefficients == other.coefficients).all() and \
            self.bound == other.bound

    def __repr__(self):
        return f"coefficients: {self.coefficients}\nbound: {self.bound})"

    def __str__(self):
        return f"Hyperplane\n\tcoefficients: {self.coefficients.tolist()}\n\tbound: {self.bound}"

    def __len__(self):
        """Length of the premise as its dimensionality

        Returns:
            The number of coefficients of this hyperplane.
        """
        return self.coefficients.size

    def __getitem__(self, item):
        if item <= self.coefficients.size:
            return self.coefficients[item]
        else:
            raise ValueError(f"Expected value in [0, {self.coefficients.size}], {item} found")

    def __setitem__(self, key, item):
        if key > len(self):
            raise ValueError(f"Expected value in [0, {len(self)}], {key} found")
        self.coefficients[key] = item

        return self

    def __iter__(self):
        for coefficient in self.coefficients:
            yield coefficient

    def __call__(self, data: numpy.ndarray, **kwargs) -> numpy.ndarray:
        """Check whether the given array is within the premise, i.e., whether the premise covers the array or not.

        Args:
            data: The data to check.
        Returns:
            An array of coverage where the i-th entry is True if data[i] lies within this hyperplane, False otherwise.
        """
        if not isinstance(data, numpy.ndarray):
            raise ValueError(f"Not a numpy.ndarray: {type(data)}")
        if (data.ndim == 2 and data.shape[1] != self.coefficients.size) \
                or data.ndim == 1 and data.shape[0] != self.coefficients.size:
            raise ValueError(f"Wrong dimensionality: seen {data.shape}, expected ({self.coefficients.shape[0]},)")
        if data.ndim == 2:
            return numpy.array([self(x) for x in data])

        return numpy.array([numpy.dot(self.coefficients, data.transpose()) <= self.bound]).squeeze()

    def __copy__(self):
        return OHyperplane(copy.copy(self.coefficients), self.bound)

    def __deepcopy__(self, memodict):
        return OHyperplane(copy.deepcopy(self.coefficients), self.bound)

    def __add__(self, other: OHyperplane | float) -> OHyperplane:
        """Sum this OHyperplane to the other.

        Args:
            other: The hyperplane to sum.
        Returns:
            An hyperplane whose coefficients are the sum of the coefficients, and the bound is the sum of the bounds.
        """
        if isinstance(other, OHyperplane):
            if self.coefficients.size != other.coefficients.size:
                raise ValueError("Expected size {self.coefficients.size}, {other.coefficients.size} found")
            return OHyperplane(self.coefficients + other.coefficients, self.bound + other.bound)
        elif isinstance(other, (int, float)):
            return OHyperplane(self.coefficients + other, self.bound + other)
        else:
            raise ValueError("Not an hyperplane")

    def __sub__(self, other: OHyperplane | float) -> OHyperplane:
        """Subtract the other OHyperplane from this.

        Args:
            other: The hyperplane to subtract.
        Returns:
            An hyperplane whose coefficients are the difference of the coefficients, and the bound is the difference
            of the bounds.
        """
        if isinstance(other, OHyperplane):
            if self.coefficients.size != other.coefficients.size:
                raise ValueError("Expected size {self.coefficients.size}, {other.coefficients.size} found")
            return OHyperplane(self.coefficients - other.coefficients, self.bound - other.bound)
        elif isinstance(other, (int, float)):
            return OHyperplane(self.coefficients - other, self.bound - other)
        else:
            raise ValueError("Not an hyperplane")

    def __mul__(self, other: OHyperplane | float) -> OHyperplane:
        """Multiply this OHyperplane by the other, if an OHyperplane, or scale its factors, if a scalar.

        Args:
            other: The hyperplane to multiply, or the scaling factor. Can be a tuple (coefficients, bound), an
                    OHyperplane, or a float.
        Returns:
            An hyperplane whose coefficients are the product of the coefficients, and the bound is the product
            of the bounds. If a scalar is given, it multiplies the coefficients and bound by the scalar.
        """
        if isinstance(other, OHyperplane):
            if self.coefficients.size != other.coefficients.size:
                raise ValueError("Expected size {self.coefficients.size}, {other.coefficients.size} found")
            return OHyperplane(numpy.multiply(self.coefficients, other.coefficients), self.bound * other.bound)
        elif isinstance(other, (int, float)):
            return OHyperplane(self.coefficients * other, self.bound * other)
        else:
            raise ValueError("Not an hyperplane")

    def __invert__(self):
        """Negate this hyperplane by flipping its sign.

        Returns:
            A OHyperplane with negated coefficients and bound.
        """
        return OHyperplane(-self.coefficients,
                           numpy.nextafter(-self.bound, -numpy.inf))

    def __neg__(self):
        """Invert this hyperplane by flipping its sign.

        Returns:
            A OHyperplane with negated coefficients and bound.
        """
        return ~self

    @staticmethod
    def from_aphyperplane(aphyperplane: APHyperplane, dimensionality: int = 1) -> OHyperplane:
        """Create an OHyperplane from an APHyperplane.

        Args:
            aphyperplane: The hyperplane to transform.
            dimensionality: Dimensionality of the desired OHyperplane.
        Returns:
            A sequence of OHyperplane with unitary coefficients for features appearing in the rule, and zero coefficients for other
            features. The bound is given by a sum of the thresholds of the axis-parallel premises.
            A sequence is returned in case the APHyperplane is defined on both lower and upper bound, thus a single
            OHyperplane is insufficient to properly represent it.
        """
        if dimensionality < aphyperplane.axis:
            raise ValueError("Expected dimensionality >= {0}, {1} found".format(aphyperplane.axis, dimensionality))

        base = numpy.zeros(dimensionality, )
        feat, low, upp = aphyperplane.axis, aphyperplane.lower_bound, aphyperplane.upper_bound
        if not numpy.isinf(low) and numpy.isinf(upp):
            # fi >= ai
            try:
                base[feat] = -1
            except IndexError as e:
                raise e
            bound = -low

            return OHyperplane(base, bound)
        elif numpy.isinf(low) and not numpy.isinf(upp):
            # fi < ai
            base[feat] = +1
            bound = upp

            return OHyperplane(base, bound)
        else:
            raise ValueError(f"Can't express an upper- and lower-bounded APHyperplane with a single OHyperplane: {aphyperplane}")

    def json(self) -> Dict:
        return {"type": "oblique",
                "coefficients": self.coefficients.tolist(),
                "bound": self.bound}

    @staticmethod
    def from_json(json_obj) -> OHyperplane:
        return OHyperplane(json_obj["coefficients"], json_obj["bound"])
