# Oh, My Trees
**Oh, My Trees** (OMT) is a library for hyperplane-based Decision Tree induction, which allows you to induce
both Univariate (e.g., CART, C4.5) and Multivariate (OC1, Geometric) Decision Trees.
It currently supports single-class classification trees, and does not support categorical variables as they don't play
well with hyperplanes.

# Quickstart
## Installation
Installation through git:
```shell
git clone https://github.com/msetzu/oh-my-trees
mkvirtualenv -p python3.11 omt  # optional, creates virtual environment

cd oh-my-trees
pip install -r src/requirements.txt
```
or directly through `pip`:
```shell
pip install oh-my-trees
```

## Training trees
OMT follows the classic sklearn `fit`/`predict` interface.  
You can find a full example in the examples notebook `notebooks/examples.ipynb`.

```python
from trees import OmnivariateDT

dt = OmnivariateDT()
x = ...
y = ...

# trees all follow a similar sklearn-like training interface, with max_depth, min_samples, and min_eps as available parameters
dt.fit(x, y, max_depth=4)
```

OMT also offers a pruning toolkit, handled by `trees.pruning.Gardener`, which allows you to prune the inducted Tree.

## Induction algorithms
`OMT` offers several Tree induction algorithms

| Algorithm     | Type        | Reference | Info |
| ---------     | ----------- | --------- | -- |
| C4.5          | Univariate  | | |
| CART          | Univariate  | | |
| OC1           | Multivariate| [Paper](https://dl.acm.org/doi/10.5555/1622826.1622827) | |
| Geometric     | Multivariate| [Paper](http://arxiv.org/abs/1009.3604) | Only traditional SVM cut |
| Omnivariate   | Multivariate| | Test all possible splits, pick the best one |
| Model tree    | Multivariate| [Paper](https://sci2s.ugr.es/keel/pdf/algorithm/congreso/1992-Quinlan-AI.pdf) | |
| Linear tree   | Multivariate| [Paper](https://sci2s.ugr.es/keel/pdf/algorithm/congreso/1992-Quinlan-AI.pdf) | |
| Optimal trees* | Univariate and Multivariate| [Paper](https://www.mit.edu/~dbertsim/papers/Machine%20Learning%20under%20a%20Modern%20Optimization%20Lens/Optimal_classification_trees_MachineLearning.pdf) | Mirror of [Interpretable AI's implementation](https://www.interpretable.ai) |

*As mirror of [Interpretable AI's implementation](https://www.interpretable.ai), you need to [install the appropriate license](https://docs.interpretable.ai/stable/installation/) to use Optimal trees

## Using Trees
You can get an explicit view of a `tree` by accessing:
- `tree.nodes: Dict[int, Node]` its nodes,
- `tree.parent: Dict[int, int], tree.ancestors: Dict[int, List[int]]` its parent and ancestors,
- `tree.descendants: Dict[int, List[int]` its descendants,
- `tree.depth: Dict[int, int]`: the depth of its nodes.

`Tree`s can also be JSONized:
```python
tree.json()
```

## Growing your own `Tree`
Greedy trees follow the basic algorithmic core of
- learning step: induce a node
- if shall continue:
  - generate two children
  - recurse on the given children

We incorporate this algorithm in `Tree`, where `step` implements the node induction, thus, most greedy induction
algorithms can implemented by simply overriding the `step` function, e.g., see `trees.cart.Cart`:

```python
    def step(self, parent_node: Optional[InternalNode],
             data: numpy.ndarray, labels: numpy.ndarray, classes: numpy.ndarray,
             direction: Optional[str] = None, depth: int = 1,
             min_eps: float = 0.000001, max_depth: int = 16, min_samples: int = 10,
             node_fitness_function: Optional[Callable] = None,
             node_hyperparameters: Optional[Dict] = None, **step_hyperparameters) -> Optional[Node]
```