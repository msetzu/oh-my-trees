# Pruning
We prune `Tree`s by implementing the `Gardener` interface:
```python
class Gardener(abc.ABC):
    """Prunes a given Tree."""
    @abc.abstractmethod
    def prune(self, tree: Tree, **kwargs) -> Tree:
        pass
```
`Gardener`s **create a copy** of the given Tree, they do not directly modify the original.

As of now, we provide the following `Gardener`s.
- `DepthGardener`, which prunes a `Tree` by cutting at a given `max_depth`
- `GreedyBottomUpGardener`, which prunes a `Tree` in a greedy manner by replacing

### DepthGardener
The `DepthGardener` simply trims the given `Tree` at a maximum depth.
```python
from pruning import DepthGardener


tree = ... # train your Tree

pruner = DepthGardener()
pruned_tree = pruner.prune(tree, max_depth=d)
```

### GreedyBottomUpGardener
The `GreedyBottomUpGardener` prunes greedely in a bottom-up fashion: it tries to replace an internal node with an
alternative one, and, if successful, keeps on going up. 

```python
import numpy
from pruning import GreedyBottomUpGardener

tree = ...  # train your Tree

pruner = GreedyBottomUpGardener()
validation_data = ...
validation_labels = ...
pruned_tree = pruner.prune(tree,
                           validation_data=validation_data,
                           validation_labels=validation_labels,
                           classes=numpy.unique(validation_labels),
                           node_fitness_function=gini)
```

