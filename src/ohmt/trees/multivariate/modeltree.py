from __future__ import annotations

import logging
import copy
from typing import Optional, Set, Dict, Callable

import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from ohmt.planes import OHyperplane
from ohmt.systems.utils import from_cart
from ohmt.trees.splits.evaluation import label_deviation, gini, accuracy
from ohmt.trees.structure.trees import ObliqueTree, InternalNode, Node, Leaf


class BinaryModelTree(ObliqueTree):
    """Model trees (https://link.springer.com/article/10.1023/A:1007421302149)"""
    def __init__(self, root: Optional[InternalNode] = None, store_data: bool = True):
        super(BinaryModelTree, self).__init__(root, store_data)

    def fit(self, data: numpy.ndarray, labels: numpy.ndarray,
            max_depth: int = numpy.inf, min_eps: float = 0.00000001, min_samples: int = 10,
            node_hyperparameters: Optional[Dict] = None, **step_hyperparameters) -> BinaryModelTree:
        """Learn a Model Decision Tree.

        Args:
            data: The training set
            labels: The training set labels
            max_depth: Maximum depth of the Decision Tree
            min_eps: Minimum improve in the learning metric to keep on creating new nodes
            min_samples: Minimum number of samples in leaves
            node_hyperparameters: Hyperparameters passed to the node construction

        Returns:
            This ModelTree, fit to the given `data` and `labels`.
        """
        # create root
        logging.debug("Fitting tree with:")
        logging.debug(f"\tmax depth: {max_depth}")
        logging.debug(f"\tmin eps: {min_eps}")

        super().fit(data=data, labels=labels,
                    max_depth=max_depth, min_eps=min_eps, min_samples=min_samples,
                    node_hyperparameters=node_hyperparameters, step_hyperparameters=step_hyperparameters)

        self.build()

        # build companion linear models
        one_to_last_nodes = {node for node in self.nodes if len(self.descendants[node]) == 2}
        for node in one_to_last_nodes:
            self.fit_companion_models(self.nodes[node],
                                      linear_model=step_hyperparameters.get("linear_family", "logistic"))

        self.build()

        return self

    def step(self, parent_node: Optional[InternalNode],
             data: numpy.ndarray, labels: numpy.ndarray, classes: numpy.ndarray,
             direction: Optional[str] = None, depth: int = 1,
             min_eps: float = 0.000001, max_depth: int = 16, min_samples: int = 10,
             node_fitness_function: Callable = gini,
             node_hyperparameters: Optional[Dict] = None, **step_hyperparameters) -> Optional[Node]:
        if direction == "left" and parent_node.children[0] is not None:
            return None
        if direction == "right" and parent_node.children[1] is not None:
            return None

        # stopping criteria
        should_stop_for_depth = self.should_stop_for_max_depth(depth, max_depth, parent_node, labels,
                                                               validation_data=data)
        should_stop_for_one_class = self.should_stop_for_one_class(labels, parent_node, direction, validation_data=data)
        should_stop_for_min_samples = self.should_stop_for_min_samples(data, labels, direction, parent_node,
                                                                       min_samples)

        if not should_stop_for_depth and not should_stop_for_min_samples and not should_stop_for_one_class:
            # learn split
            cart = DecisionTreeClassifier(max_depth=1)
            cart.fit(data, labels)
            extracted_rules = from_cart(cart)
            if len(extracted_rules) == 0:
                fit_node = self.build_leaf(labels, data=data)
                self.store(fit_node, data, labels)

                return fit_node
            else:
                best_split = [OHyperplane.from_aphyperplane(hyperplane[0], dimensionality=data.shape[1])
                              for hyperplane, _ in extracted_rules][0]

            # internal node
            if parent_node is not None:
                error_delta = label_deviation(parent_node.hyperplane, parent_node._data, parent_node._labels)
                if error_delta < min_eps:
                    # error stopping criterion
                    logging.debug(f"\t\tReached minimum error delta of {min_eps}")
                    fit_node = self.build_leaf(labels, data=data)
                else:
                    # internal node
                    fit_node = InternalNode(best_split)
            else:
                # tree root
                fit_node = InternalNode(best_split)

            self.store(fit_node, data, labels)

            return fit_node

        else:
            return None

    def fit_companion_models(self, node: InternalNode, linear_model: str = "logistic", max_loss: int = 10):
        """Replace nodes in the tree w/ linear models whose features are found in the splitting criteria of its subtree.

        Args:
            node: The current node whose companion model to fit
            linear_model: The linear model to fit, for now only "logist" is supported
            max_loss: Maximum loss in accuracy accepted when pruning the companion model
        """
        if isinstance(node, Leaf):
            return

        root_features = numpy.argwhere(node.hyperplane.coefficients != 0).squeeze().tolist()
        root_features = {root_features} if isinstance(root_features, int) else set(root_features)
        left_child, right_child = node.children
        descendant_features = self.features_in_subtree(left_child, root_features) | self.features_in_subtree(right_child, root_features)
        nondescendant_features = numpy.array([feature for feature in range(node._data.shape[1])
                                              if feature not in descendant_features])

        if linear_model == "logistic":
            class_model = LogisticRegression()
        else:
            raise ValueError(f"Unknown model: {linear_model}")

        # filter features and create candidate hyperplane
        filtered_data = copy.deepcopy(node._data)
        # filter-out data by setting it to zero to preserve dimensionality later
        if nondescendant_features.size > 0:
            filtered_data[:, nondescendant_features] = 0.
        class_model.fit(filtered_data, node._labels)
        # prune companion model
        class_model = self.prune_companion_model(class_model, filtered_data, node._labels, max_loss=max_loss)
        companion_hyperplane = OHyperplane(class_model.coef_[0], class_model.intercept_[0])

        if accuracy(companion_hyperplane, data=filtered_data, labels=node._labels) > accuracy(node.hyperplane, data=filtered_data, labels=node._labels):
            node.hyperplane = companion_hyperplane
            node.children[0] = self.build_leaf(labels=node._labels[node.hyperplane(filtered_data)],
                                               data=node._data[node.hyperplane(filtered_data)])
            node.children[1] = self.build_leaf(labels=node._labels[~(node.hyperplane(filtered_data))],
                                               data=node._data[~(node.hyperplane(filtered_data))])

    def prune_companion_model(self, model: LogisticRegression, data: numpy.ndarray, labels: numpy.ndarray,
                              max_loss: int = 10) -> LogisticRegression:
        """Prune the given Linear Regression `model` by greedily removing features until at most `max_loss` percent of
        the model accuracy is lost.

        Args:
            model: The Linear Regression model to prune
            data: The data to validate the model loss in accuracy on
            labels: The labels to validate the model loss in accuracy on
            max_loss: The maximum accepted loss in percentage points

        Returns:
            A pruned Linear Regression model
        """
        nonpruned_model_accuracy = accuracy_score(labels, model.predict(data))
        nonzero_features = numpy.argwhere(model.coef_[0] != 0).squeeze()
        pruned_model = copy.deepcopy(model)

        while nonzero_features.size > 1:
            pruning_candidates = [copy.deepcopy(pruned_model) for _ in nonzero_features]
            for feature_to_remove, candidate in zip(nonzero_features, pruning_candidates):
                candidate.coef_[0][feature_to_remove] = 0
                candidates_accuracies = numpy.array([accuracy_score(labels, candidate.predict(data))
                                                     for candidate in pruning_candidates])

            best_candidate_idx = candidates_accuracies.argmax()
            accuracy_delta = nonpruned_model_accuracy - candidates_accuracies[best_candidate_idx]

            if accuracy_delta >= nonpruned_model_accuracy * (max_loss / 100):
                pruned_model = pruning_candidates[best_candidate_idx]
                numpy.delete(nonzero_features, best_candidate_idx)
            else:
                break

        return pruned_model

    def features_in_subtree(self, node: Node, current_features: Set[int]) -> Set[int]:
        """Compute the features used in the splitting criteria of the tree rooted in `node`.

        Args:
            node: The root of the subtree
            current_features: Current features found in the tree
        """
        if isinstance(node, Leaf):
            return current_features

        node_features = numpy.argwhere(node.hyperplane.coefficients != 0).squeeze().tolist()
        node_features = {node_features} if isinstance(node_features, int) else set(node_features)
        cumulated_features = current_features | node_features
        left_child, right_child = node.children
        descendant_features = self.features_in_subtree(left_child, cumulated_features) | self.features_in_subtree(right_child, cumulated_features)

        return cumulated_features | descendant_features
