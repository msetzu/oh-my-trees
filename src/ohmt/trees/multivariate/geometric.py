from __future__ import annotations

import copy
import random
from typing import Tuple, Optional, Dict, Callable

import numpy
import scipy as scipy
from numpy.linalg import matrix_rank
from scipy.linalg import null_space
from sklearn import svm

from ohmt.planes import OHyperplane
from ohmt.trees.splits.evaluation import gini
from ohmt.trees.structure.trees import Node, InternalNode, ObliqueTree


class GeometricDT(ObliqueTree):
    """Binary Geometric Decision Trees (https://arxiv.org/abs/1009.3604v1)"""
    def __init__(self, root: Optional[InternalNode] = None, store_data: bool = True):
        super(GeometricDT, self).__init__(root, store_data)
        super(GeometricDT, self).__init__(root)

    def step(self, parent_node: Optional[InternalNode],
             data: numpy.ndarray, labels: numpy.ndarray, classes: numpy.ndarray,
             direction: Optional[str] = None, depth: int = 1,
             min_eps: float = 0.000001, max_depth: int = 16, min_samples: int = 10,
             node_fitness_function: Callable = gini,
             node_hyperparameters: Optional[Dict] = None, **step_hyperparameters) -> Optional[Node]:
        # stopping criteria
        should_stop_for_depth = self.should_stop_for_max_depth(depth, max_depth, parent_node, labels, validation_data=data)
        should_stop_for_min_samples = self.should_stop_for_min_samples(data, labels, direction, parent_node,
                                                                       min_samples)

        if not should_stop_for_depth and not should_stop_for_min_samples:
            soft_margin = node_hyperparameters.get("soft_margin", 0.001)
            max_iterations = node_hyperparameters.get("max_iterations", 10000)
            strategy = step_hyperparameters.get("strategy", "svm")

            if strategy == "psvm":
                # split by class
                indices_per_class = [numpy.argwhere(labels == c).squeeze() for c in classes]
                matrices_by_class = [data[indices, :] for indices in indices_per_class]
                distance_matrices = list()
                for i in range(len(matrices_by_class)):
                    # add a column of 1s
                    matrices_by_class[i] = numpy.hstack((matrices_by_class[i],
                                                         numpy.array(numpy.ones(matrices_by_class[i].shape[0]).reshape(1,-1).transpose())))
                    distance_matrices = list()
                    for row in matrices_by_class[i]:
                        x = row.transpose().reshape(1, -1)
                        distance_matrices.append(x * x.transpose())
                    distance_matrices.append(sum(distance_matrices))
                distances_A, distances_B = distance_matrices

                # generalized eigenvalue problem (Eq. 3 of the paper)
                rank_A, rank_B = matrix_rank(distances_A), matrix_rank(distances_B)
                # low-rank matrix, apply null-space projection
                if rank_A < distances_A.shape[1]:
                    null_space_A = null_space(distances_A)
                    distances_A = null_space_A * null_space_A.transpose() * distances_A * null_space_A * null_space_A.transpose()
                if rank_B < distances_B.shape[1]:
                    null_space_B = null_space(distances_B)
                    distances_B = null_space_B * null_space_B.transpose() * distances_B * null_space_B * null_space_B.transpose()
                distances_A, distances_B = self.__equalize_size(distances_A, distances_B)

                # planes computation
                eigenvalues, eigenvectors = scipy.linalg.eigh(distances_A, distances_B)
                clustering_hyperplane, separating_hyperplane = eigenvectors[0], eigenvectors[-1]
                clustering_hyperplane = OHyperplane(clustering_hyperplane[:-1], clustering_hyperplane[-1])
                separating_hyperplane = OHyperplane(separating_hyperplane[:-1], separating_hyperplane[-1])

                # planes selection
                clustering_bisector = clustering_hyperplane + separating_hyperplane
                separating_bisector = clustering_hyperplane - separating_hyperplane
                # bisector computation (Eq. 4 of the paper)
                if clustering_bisector == separating_bisector:
                    hyperplane = OHyperplane(clustering_bisector.coefficients,
                                             (clustering_bisector.bound + separating_bisector.bound) / 2)
                else:
                    # bisector selection (Eq. 6 of the paper)
                    clustering_bisector_fitness = node_fitness_function(clustering_bisector, data=data, labels=labels, classes=classes)
                    separating_bisector_fitness = node_fitness_function(separating_bisector, data=data, labels=labels, classes=classes)
                    hyperplane = clustering_bisector if clustering_bisector_fitness > separating_bisector_fitness else separating_bisector
            elif strategy == "svm":
                dual = data.shape[0] < data.shape[1]
                hyperplane_svm = svm.LinearSVC(penalty="l1" if not dual else "l2", dual=dual, max_iter=max_iterations,
                                               C=soft_margin)
                hyperplane_svm.fit(data, labels)
                hyperplane = OHyperplane(hyperplane_svm.coef_.transpose().squeeze(), hyperplane_svm.intercept_[0])
            else:
                raise ValueError(f"Unknown strategy {strategy}: choose one of (\"svm\", \"psvm\")")

            if parent_node is not None:
                if self.should_stop_for_min_eps(parent_node, InternalNode(hyperplane),
                                                node_fitness_function=node_fitness_function,
                                                validation_data=data, validation_labels=labels, classes=classes,
                                                min_eps=min_eps):
                    fit_node = self.build_leaf(labels)
                else:
                    fit_node = InternalNode(hyperplane)
            else:
                fit_node = InternalNode(hyperplane)

            self.store(fit_node, data, labels)

            return fit_node
        else:
            return None

    def __equalize_size(self, A: numpy.ndarray, B: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Trim `A, B` so that the two square matrices have the same size.
        Args:
            A: A matrix.
            B: Another matrix.

        Returns:
            The two matrices trimmed from `A, B` and of equal size. Returns `A, B` if they already have the
            same size.
        """
        if A.shape == B.shape:
            return A, B
        else:
            trimmed_A, trimmed_B = copy.copy(A), copy.copy(B)
            size_A, size_B = A.shape[0], B.shape[0]
            size_delta = abs(size_A - size_B)
            if size_A > size_B:
                random_indices = random.sample(range(size_A), k=size_delta)
                trimmed_A = numpy.delete(trimmed_A, random_indices, axis=0)
                trimmed_A = numpy.delete(trimmed_A, random_indices, axis=1)
            else:
                random_indices = random.sample(range(size_B), k=size_delta)
                trimmed_B = numpy.delete(trimmed_B, random_indices, axis=0)
                trimmed_B = numpy.delete(trimmed_B, random_indices, axis=1)

            return trimmed_A, trimmed_B
