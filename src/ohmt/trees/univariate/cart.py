from typing import Optional, Dict

import numpy

from ohmt.trees.splits.evaluation import information_gain
from ohmt.trees.structure.trees import InternalNode, ParallelTree, Node


class Cart(ParallelTree):
    """Axis-parallel Tree built using the provided model family for internal node split."""

    def __init__(self, root: Optional[InternalNode] = None):
        super(Cart, self).__init__(root)

    def step(self, parent_node: Optional[InternalNode],
             data: numpy.ndarray, labels: numpy.ndarray, classes: numpy.ndarray,
             direction: Optional[str] = None, depth: int = 1,
             min_eps: float = 0.000001, max_depth: int = 16, min_samples: int = 10,
             node_hyperparameters: Optional[Dict] = None,
             **step_hyperparameters) -> Optional[Node]:
        return super().step(parent_node=parent_node,
                            data=data, labels=labels, classes=classes,
                            direction=direction,
                            depth=depth, min_eps=min_eps, max_depth=max_depth, min_samples=min_samples,
                            node_hyperparameters=None,
                            node_fitness_function=information_gain,
                            **{k: v for k, v in step_hyperparameters.items() if k != "node_fitness_function"})
