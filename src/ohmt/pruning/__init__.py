import abc
import copy
from typing import Callable, Optional, Tuple, Dict

import numpy

from ohmt.planes import Hyperplane
from ohmt.trees.splits.evaluation import label_deviation
from ohmt.trees.structure.trees import Tree, InternalNode, Leaf, Node


class Gardener(abc.ABC):
    """Prunes a given Tree."""
    @abc.abstractmethod
    def prune(self, tree: Tree, **kwargs) -> Tree:
        pass

class DepthGardener(Gardener):
    """Prune by maximum depth."""
    def prune(self, tree: Tree, **kwargs) -> Tree:
        """Prune the given `tree` to the given `max_depth`.

        Args:
            tree: The Tree to prune.
            kwargs:
                max_depth: Maximum depth of the resulting Tree

        Returns:
             The pruned Tree.
        """
        if "max_depth" not in kwargs:
            raise ValueError("Must specify max_depth")

        return self._prune(tree, max_depth=kwargs["max_depth"])

    def _prune(self, tree: Tree, max_depth: int) -> Tree:
        """Prune the given `tree` to the given `max_depth`.

        Args:
            tree: The Tree to prune.
            max_depth: Maximum depth of the resulting Tree

        Returns:
             The pruned Tree.
        """
        pruned_tree = copy.deepcopy(tree)
        new_leaves_ids = [node for node in pruned_tree.nodes if tree.depth[node] == max_depth]

        # build new leaves
        for node_id in new_leaves_ids:
            parent_node_id = pruned_tree.parent[node_id]
            parent_node = pruned_tree.nodes[parent_node_id]
            child_index = node_id % 2  # left children have even index, right children odd index
            replacing_leaf = pruned_tree.build_leaf(pruned_tree.nodes[node_id]._labels)
            del pruned_tree.nodes[node_id]

            parent_node.children[child_index] = replacing_leaf

        over_nodes = [node for node in pruned_tree.nodes if len(pruned_tree.ancestors[node]) > max_depth]
        # delete nodes over cutoff
        for node_id in over_nodes:
            del pruned_tree.nodes[node_id]

        pruned_tree.build()

        return pruned_tree


class GreedyBottomUpGardener(Gardener):
    def prune(self, tree: Tree, **kwargs) -> Tree:
        if "validation_data" not in kwargs or "validation_labels" not in kwargs:
            raise ValueError("Must include both validation_data and validation_labels")

        args = {k: v for k, v in dict(kwargs).items() if k != "validation_data" and k != "validation_labels"}

        pruned_tree = copy.deepcopy(tree)
        self.prune_by_replacing(pruned_tree, pruned_tree.root, direction="down",
                                validation_data=kwargs["validation_data"],
                                validation_labels=kwargs["validation_labels"],
                                node_hyperparameters=kwargs.get("node_hyperparameters", dict()),
                                **args)
        pruned_tree.build()

        return pruned_tree

    def should_replace_with_companion(self, hyperplane: Hyperplane,
                                      companion_hyperplane: Hyperplane,
                                      score_hyperplane: Callable[[Hyperplane, numpy.ndarray, numpy.array], float] = label_deviation,
                                      validation_data: Optional[numpy.ndarray] = None,
                                      validation_labels: Optional[numpy.array] = None) -> bool:
        """Should `companion_hyperplane` replace `hyperplane`? Answer by checking performance on `validation_data`,
        `validation_labels`.

        Args:
            hyperplane: The original hyperplane, candidate to be replaced.
            companion_hyperplane: The companion hyperplane, candidate to replace.
            score_hyperplane: Callable to score each hyperplane: higher scores are better.
            validation_data: Validation data to score the hyperplanes.
            validation_labels: Validation labels to score the hyperplanes.

        Returns:
            True if `companion_hyperplane` achieves a higher score than `hyperplane`, False otherwise.
        """
        if companion_hyperplane is None or validation_data is None or validation_labels is None:
            return False

        companion_hyperplane_score = score_hyperplane(companion_hyperplane, validation_data, validation_labels)
        original_hyperplane_score = score_hyperplane(hyperplane, validation_data, validation_labels)

        return companion_hyperplane_score >= original_hyperplane_score

    def generate_companion_node(self, node, tree, validation_data, validation_labels, classes,
                                node_hyperparameters: Dict,
                                **step_hyperparameters) -> Tuple[Optional[Node],
                                                                 Optional[numpy.ndarray],
                                                                 Optional[numpy.ndarray]]:
        """Attempt to generate a node to replace `node` in the given `tree` by inducing on the given
        `validation_data` and `validation_labels`.

        Args:
            node: The node whose companion to generate.
            tree: The tree.
            validation_data: Validation data to guide generation.
            validation_labels: Validation labels to guide generation.
            classes: Set of labels.
            node_hyperparameters: Hyperparameters for the node induction function.
            **step_hyperparameters: Hyperparameters for the step function.

        Returns:
            A node, if one was induced, None otherwise, alongside the used validation data and labels.
        """
        parent_node_id = tree.parent[node.node_id]
        parent_node = tree.nodes[parent_node_id]
        # nodes get the data of the parent...
        path_to_parent = tree.path_to_node(parent_node_id)  # also the node itself is included, thus need to remove the last element
        parent_indices = path_to_parent(validation_data)
        validation_data_of_parent = validation_data[parent_indices]
        validation_labels_of_parent = validation_labels[parent_indices]

        # ...but only according to their position
        if parent_node(validation_data_of_parent).size > 0:
            if node.direction == "left":
                node_validation_data = validation_data_of_parent[parent_node(validation_data_of_parent)]
                node_validation_labels = validation_labels_of_parent[parent_node(validation_data_of_parent)]
            else:
                node_validation_data = validation_data_of_parent[~parent_node(validation_data_of_parent)]
                node_validation_labels = validation_labels_of_parent[~parent_node(validation_data_of_parent)]

            companion_node = tree.step(parent_node=tree.nodes[tree.parent[node.node_id]],
                                       data=node_validation_data, labels=node_validation_labels,
                                       classes=classes, node_hyperparameters=node_hyperparameters,
                                       **step_hyperparameters)

            return companion_node, node_validation_data, node_validation_labels

        else:
            return None, None, None

    def prune_by_replacing(self, tree: Tree, node: Node,
                           score_hyperplane: Callable[[Hyperplane, numpy.ndarray, numpy.array], float] = label_deviation,
                           validation_data: Optional[numpy.ndarray]  = None,
                           validation_labels: Optional[numpy.array] = None,
                           classes: Optional[numpy.array] = None,
                           direction: str = "down",
                           node_hyperparameters: Optional[Dict] = None,
                           **step_hyperparameters):
        """Prune this tree in a bottom-up fashion. When `direction` is "down" reach the one-to-last layer and try to
        replace it, while when `direction` is up traverse the tree back up.

        Args:
            tree: The tree being pruned
            node: The current node
            score_hyperplane: Callable to score each hyperplane: higher scores are better.
            validation_data: Validation data provided to decide whether to trim or not.
            validation_labels: Validation labels provided to decide whether to trim or not.
            classes: Dataset classes used to compute the hyperplane score
            direction: The current tree traversal direction, either "down" or "up".
            node_hyperparameters: Hyperparameters for node induction.

        """
        # node already replaced! Due to the greedy nature of the pruning, even a parent of two leaves is necessarily
        # first replaced by one of the two! if it is cut (not in tree.nodes), then we must skip it
        if node.node_id not in tree.nodes:
            return

        if direction == "down":
            # down direction: looking for the last layer to fit nodes
            if isinstance(node, Leaf):
                self.prune_by_replacing(tree,
                                        node=tree.nodes[tree.parent[node.node_id]],
                                        direction="up",
                                        score_hyperplane=score_hyperplane,
                                        validation_data=validation_data,
                                        validation_labels=validation_labels,
                                        classes=classes,
                                        node_hyperparameters=node_hyperparameters,
                                        **step_hyperparameters)
            else:
                left_child, right_child = node.children

                # one-to-last layer
                if isinstance(left_child, Leaf) and isinstance(right_child, Leaf):
                    self.prune_by_replacing(tree,
                                            node=left_child,
                                            direction="up",
                                            score_hyperplane=score_hyperplane,
                                            validation_data=validation_data,
                                            validation_labels=validation_labels,
                                            classes=classes,
                                            node_hyperparameters=node_hyperparameters,
                                            **step_hyperparameters)
                    self.prune_by_replacing(tree,
                                            node=right_child,
                                            direction="up",
                                            score_hyperplane=score_hyperplane,
                                            validation_data=validation_data,
                                            validation_labels=validation_labels,
                                            classes=classes,
                                            node_hyperparameters=node_hyperparameters,
                                            **step_hyperparameters)

                elif isinstance(left_child, Leaf) and isinstance(right_child, InternalNode):
                    self.prune_by_replacing(tree,
                                            node=right_child,
                                            direction="down",
                                            score_hyperplane=score_hyperplane,
                                            validation_data=validation_data,
                                            validation_labels=validation_labels,
                                            classes=classes,
                                            node_hyperparameters=node_hyperparameters,
                                            **step_hyperparameters)

                elif isinstance(left_child, InternalNode) and isinstance(right_child, Leaf):
                    self.prune_by_replacing(tree,
                                            node=left_child,
                                            direction="down",
                                            score_hyperplane=score_hyperplane,
                                            validation_data=validation_data,
                                            validation_labels=validation_labels,
                                            classes=classes,
                                            node_hyperparameters=node_hyperparameters,
                                            **step_hyperparameters)

                else:
                    # both internal nodes, iterate
                    self.prune_by_replacing(tree,
                                            node=left_child,
                                            direction="down",
                                            score_hyperplane=score_hyperplane,
                                            validation_data=validation_data,
                                            validation_labels=validation_labels,
                                            classes=classes,
                                            node_hyperparameters=node_hyperparameters,
                                            **step_hyperparameters)
                    self.prune_by_replacing(tree,
                                            node=right_child,
                                            direction="down",
                                            score_hyperplane=score_hyperplane,
                                            validation_data=validation_data,
                                            validation_labels=validation_labels,
                                            classes=classes,
                                            node_hyperparameters=node_hyperparameters,
                                            **step_hyperparameters)

        else:
            parent_node_id = tree.parent[node.node_id]
            parent_node = tree.nodes[parent_node_id]
            original_hyperplane = node.hyperplane

            if isinstance(node, Leaf):
                # go back up the tree, if possible: root has node_id == 1
                if parent_node_id > 1:
                    self.prune_by_replacing(tree,
                                            node=parent_node,
                                            direction="up",
                                            score_hyperplane=score_hyperplane,
                                            validation_data=validation_data,
                                            validation_labels=validation_labels,
                                            classes=classes,
                                            node_hyperparameters=node_hyperparameters,
                                            **step_hyperparameters)

            else:
                replacing_node, node_validation_data, node_validation_labels = self.generate_companion_node(node, tree,
                                                                                                            validation_data,
                                                                                                            validation_labels,
                                                                                                            classes,
                                                                                                            node_hyperparameters=node_hyperparameters,
                                                                                                            **step_hyperparameters)
                if isinstance(replacing_node, InternalNode) and self.should_replace_with_companion(original_hyperplane,
                                                                                          replacing_node.hyperplane,
                                                                                                   score_hyperplane=score_hyperplane,
                                                                                                   validation_data=node_validation_data,
                                                                                                   validation_labels=node_validation_labels):
                    # remove old leaves
                    children_node_ids = [node.children[0].node_id, node.children[1].node_id]
                    del tree.nodes[children_node_ids[0]]
                    del tree.nodes[children_node_ids[1]]

                    # construct new leaves
                    children = [tree.build_leaf(node_validation_labels[replacing_node(node_validation_data)]),
                                tree.build_leaf(node_validation_labels[~replacing_node(node_validation_data)])]
                    children[0].node_id = children_node_ids[0]
                    children[1].node_id = children_node_ids[1]
                    node.hyperplane = replacing_node.hyperplane
                    node.children = children

                    tree.nodes[children_node_ids[0]] = children[0]
                    tree.nodes[children_node_ids[1]] = children[1]
                    tree.parent[children_node_ids[0]] = node.node_id
                    tree.parent[children_node_ids[1]] = node.node_id

                elif isinstance(replacing_node, Leaf):
                    try:
                        children_node_ids = [node.children[0].node_id, node.children[1].node_id]
                    except IndexError as e:
                        raise e

                    if node.direction == "left":
                        del tree.nodes[children_node_ids[0]]
                        parent_node.children[0] = replacing_node
                    else:
                        del tree.nodes[children_node_ids[1]]
                        parent_node.children[1] = replacing_node

                    # update the tree information
                    tree.build()

                    # go back up the tree, if possible: root has node_id == 1
                    if parent_node_id > 1:
                        self.prune_by_replacing(tree,
                                                node=parent_node,
                                                direction="up",
                                                score_hyperplane=score_hyperplane,
                                                validation_data=validation_data,
                                                validation_labels=validation_labels,
                                                classes=classes,
                                                node_hyperparameters=node_hyperparameters,
                                                **step_hyperparameters)
