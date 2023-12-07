from __future__ import annotations

from typing import List, Dict, Tuple, Optional, Self, Callable

import numpy
import pandas

from interpretableai.optimaltrees import OptimalTreeClassifier
from interpretableai import iai

from ohmt.planes import OHyperplane, APHyperplane
from ohmt.trees.structure.trees import ObliqueTree, InternalNode, Node, Leaf


class OptimalClassificationDT(ObliqueTree):
    def step(self, parent_node: Optional[InternalNode], data: numpy.ndarray, labels: numpy.ndarray,
             classes: numpy.ndarray, direction: Optional[str] = None, depth: int = 1, min_eps: float = 0.000001,
             max_depth: int = 16, min_samples: int = 10, node_fitness_function: Optional[Callable] = None,
             node_hyperparameters: Optional[Dict] = None, **step_hyperparameters) -> Optional[Node]:
        pass

    def __init__(self, root: Optional[InternalNode] = None, store_data: bool = True):
        super(OptimalClassificationDT, self).__init__(root, store_data)
        self.category_to_int = dict()

    def fit(self, data: numpy.ndarray, labels: numpy.ndarray,
            max_depth: int = numpy.inf, min_eps: float = 0.00000001, min_samples: int = 10,
            node_hyperparameters: Optional[Dict] = None,
            **step_hyperparameters) -> Self:
        """Learn a OC1 Decision Tree.

        Args:
            data: The training set
            labels: The training set labels
            min_eps: Minimum improve in the learning metric to keep on creating new nodes
            min_samples: Minimum number of samples in leaves
            node_hyperparameters: Hyperparameters to fit the node
            max_depth: Maximum depth of the Decision Tree

        Returns:
            This OptimalClassificationDT, fit to the given `data` and `labels`.
        """
        # optimalTree implementation requires a header too
        feature_names = [str(i) for i in range(data.shape[1])]
        df = pandas.DataFrame(numpy.hstack((data, labels.reshape(-1, 1))), columns=feature_names + ['y'])
        feature_types = list()
        for feature in feature_names:
            try:
                _ = df[feature].values.astype(float)
                feature_types.append("numerical")
            except ValueError:
                feature_types.append("categorical")
        # convert to categorical
        for feature, (feature_name, feature_type) in enumerate(zip(feature_names, feature_types)):
            if feature_type == 'categorical':
                self.category_to_int[feature_name] = dict()
                df[feature_name] = df[feature_name].astype("category")
                self.category_to_int[feature_name] = df[feature_name].cat.categories.values.tolist()

        x, y = df.iloc[:, :-1], df.iloc[:, -1]
        grid = iai.GridSearch(
            OptimalTreeClassifier(
                random_seed=1,
                max_depth=max_depth,
                hyperplane_config={'sparsity': 'all'}
            ),
        )
        grid.fit(x, y)
        optimal_tree = grid.get_learner()
        # convert to OptimalClassificationDT
        tree = OptimalClassificationDT.julia_tree_to_python_tree(optimal_tree, feature_names, self.category_to_int)

        self.build()

        return tree

    @classmethod
    def julia_tree_to_python_tree(cls, julia_tree: OptimalTreeClassifier, feature_names: List[str], categories: Dict) -> OptimalClassificationDT:
        """Map the given Julia tree to a OptimalClassificationDT.

        Args:
            julia_tree: The tree to convert.
            feature_names: Names of the features (used by Julia)
            categories: Categorical features.

        Returns:
            The converted tree.
        """
        def _leaves_and_planes(optimal_tree: OptimalTreeClassifier, feature_names: List[str], categories: Dict) -> Tuple[Dict, Dict]:
            """Extract leaves and planes from the given `optimal_tree`."""
            nr_nodes = optimal_tree.get_num_nodes()
            nr_features = len(feature_names)
            oblique_splits = dict()
            categorical_splits = dict()
            other_splits = dict()
            leaves = dict()
            for split in range(1, nr_nodes + 1):
                # oblique
                if optimal_tree.is_hyperplane_split(split):
                    weights = optimal_tree.get_split_weights(split)
                    weights = list(weights[0].items())
                    indices = [feature_names.index(feature) for feature, _ in weights]
                    weights = [weight for _, weight in weights]
                    base_weights = numpy.zeros(nr_features, )
                    base_weights[indices] = weights

                    threshold = optimal_tree.get_split_threshold(split)
                    oblique_splits[split] = (base_weights, threshold)
                # axis-parallel
                elif optimal_tree.is_parallel_split(split):
                    feature = int(optimal_tree.get_split_feature(split))
                    threshold = optimal_tree.get_split_threshold(split)
                    oblique = OHyperplane.from_aphyperplane(APHyperplane(feature, upp=threshold),
                                                                          dimensionality=nr_features)
                    oblique_splits[split] = (oblique.coefficients, oblique.bound)
                # categorical
                elif optimal_tree.is_categoric_split(split):
                    category_dict = optimal_tree.get_split_categories(split)
                    category = [category for category, is_split in category_dict.items() if is_split][0]
                    int_mapping = categories[str(split)].index(category)
                    coefficients_lower, coefficients_upper = numpy.zeros(nr_features, ), numpy.zeros(nr_features, )
                    coefficients_lower[split], coefficients_upper[split] = -1., 1.
                    lower_threshold = int_mapping - numpy.finfo(float).eps
                    upper_threshold = -int_mapping + numpy.finfo(float).eps

                    categorical_splits[split] = [(coefficients_lower, lower_threshold),
                                                 (coefficients_upper, upper_threshold)]
                elif optimal_tree.is_leaf(split):
                    leaves[split] = numpy.zeros(2, )
                    leaves[split][int(optimal_tree.get_classification_label(split))] = 1.
                # other
                else:
                    other_splits[split] = None

            return leaves, oblique_splits

        def _rec_obliquetree_from_optimal_tree(optimal_tree: OptimalTreeClassifier, node_id: int, node: Node,
                                               leaves: Dict, oblique_splits: Dict, categories: Dict):
            """Construct the ObliqueTree tree rooted in `node_id`.

            Args:
                optimal_tree: The tree we are trying to convert into a ObliqueTree
                node_id: The id of the subtree we are constructing
                node: The root of the tree we are constructing
            """
            left_child, right_child = optimal_tree.get_lower_child(node_id), optimal_tree.get_upper_child(node_id)
            node.children = list([None, None])
            # leaves check
            if left_child in leaves:
                node.children[0] = Leaf(leaves[left_child])
            else:
                left_child_node = InternalNode(OHyperplane(oblique_splits[left_child][0], oblique_splits[left_child][1]))
                node.children[0] = left_child_node
                _rec_obliquetree_from_optimal_tree(optimal_tree, left_child, left_child_node, leaves, oblique_splits, categories)

            if right_child in leaves:
                node.children[1] = Leaf(leaves[right_child])
            else:
                right_child_node = InternalNode(OHyperplane(oblique_splits[right_child][0], oblique_splits[right_child][1]))
                node.children[1] = right_child_node
                _rec_obliquetree_from_optimal_tree(optimal_tree, right_child, right_child_node, leaves, oblique_splits, categories)

        # extract tree parameters
        leaves, oblique_splits = _leaves_and_planes(julia_tree, feature_names, categories)
        root = InternalNode(OHyperplane(oblique_splits[1][0], oblique_splits[1][1]))
        # recurse on children
        _rec_obliquetree_from_optimal_tree(julia_tree, 1, root, leaves, oblique_splits, categories)

        return OptimalClassificationDT(root)
