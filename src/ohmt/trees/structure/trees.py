from __future__ import annotations

import copy
import json
import logging
import os
from abc import ABC, abstractmethod
from more_itertools import flatten
from typing import Optional, List, Dict, TypeVar, Set, Self, Callable, Sequence

import numpy

from ohmt.planes import OHyperplane, APHyperplane, Hyperplane
from ohmt.systems import OBinaryPath, BinaryPath

T = TypeVar("T")


class Node(ABC):
    """Node (internal or leaf) of a Decision Tree.

    Attributes:
        hyperplane: The separating hyperplane, if any, None otherwise.
        self._data: Data used to train this Node, if the Tree stores it, None otherwise.
        self._labels: Labels used to train this Node, if the Tree stores it, None otherwise.
        self.node_id: Node id used to reference the Node inside a Tree.
        self.children: Children of the Node, if any. Defaults to the empty list.
    """
    def __init__(self, hyperplane: Optional[OHyperplane] = None):
        self.hyperplane = hyperplane
        self._data = None
        self._labels = None
        self.node_id = -1
        self.children = []
        self.direction = None

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __invert__(self):
        pass

    @abstractmethod
    def json(self):
        pass

    @staticmethod
    def from_json(json_obj) -> List:
        if json_obj["type"] == "leaf":
            return Leaf.from_json(json_obj)
        else:
            return InternalNode.from_json(json_obj)


class InternalNode(Node):
    """Internal node of a Decision Tree."""
    def __init__(self, hyperplane: Optional[Hyperplane] = None):
        super().__init__(hyperplane)
        self.children = [None, None]

    def __hash__(self):
        return hash(self.hyperplane) + 1000

    def __eq__(self, other):
        return isinstance(other, InternalNode) and self.hyperplane == other.hyperplane

    def __call__(self, *args, **kwargs) -> numpy.ndarray:
        """Return a routing array."""
        return self.hyperplane(args[0])

    def __repr__(self):
        return repr(self.hyperplane)

    def __invert__(self):
        node = InternalNode(~self.hyperplane)
        # flipping children position because we flipped the hyperplane
        node.children = [copy.deepcopy(self.children[1]), copy.deepcopy(self.children[0])]

        return node

    def json(self) -> Dict:
        base = dict({"type": "node"})
        base.update({"hyperplane": self.hyperplane.json()})

        return base

    @staticmethod
    def from_json(json_obj):
        # OHyperplane
        if json_obj["type"] == "oblique":
            return InternalNode(OHyperplane.from_json(json_obj))
        # APHyperplane
        else:
            return InternalNode(APHyperplane.from_json(json_obj))

    def __deepcopy__(self, memodict):
        node = InternalNode(copy.deepcopy(self.hyperplane))
        node._data = copy.deepcopy(self._data)
        node._labels = copy.deepcopy(self._labels)
        node.children = [copy.deepcopy(self.children[0]), copy.deepcopy(self.children[1])]

        return node


class LinearInternalNode(InternalNode):
    def __init__(self, hyperplane: Optional[OHyperplane] = None):
        super().__init__(hyperplane)
        self.children = [None, None]
        self.companion_hyperplane = None

    def __eq__(self, other):
        return isinstance(other, LinearInternalNode) and \
            super().__eq__(other) and \
            self.companion_hyperplane == other.companion_hyperplane


class Leaf(Node):
    """Leaf of a Decision Tree.

    Attributes:
        _labels: Probability of each label in this leaf.
    """
    def __init__(self, class_probability: numpy.ndarray, labels: Optional[numpy.array] = None):
        """Create a new Leaf with the given class probabilities.

        Args:
            class_probability: The class probabilities
        """
        super().__init__()
        self.label = class_probability
        self._data = None
        self._labels = labels

    def __hash__(self):
        return hash(self.label) + 1000

    def __eq__(self, other):
        return isinstance(other, Leaf) and (self.label == other.label).all()

    def __call__(self, *args, **kwargs) -> numpy.ndarray:
        """Get the label probability.

        Args:
            *args:
            **kwargs:

        Returns:
            The label probability.
        """
        return self.label

    def __repr__(self):
        return repr(self.label.tolist())

    def __invert__(self):
        return Leaf(1 - self.label,
                    labels=None if self._labels is None else self._labels.copy())

    def __deepcopy__(self, memodict):
        leaf = Leaf(self.label.copy(),
                    None if self._labels is None else self._labels.copy())
        leaf._data = copy.deepcopy(self._data)
        leaf._labels = copy.deepcopy(self._labels)

        return leaf

    def json(self) -> Dict:
        base = dict({"type": "leaf"})
        base.update({"label": self.label.tolist()})

        return base

    @staticmethod
    def from_json(json_obj):
        return Leaf(numpy.array(json_obj["label"]))


class Tree(ABC):
    """A hyperplane-based binary Decision Tree for classification.

    Attributes:
        nodes (Dict[int, Node]): Node dictionary.
        parent (Dict[int, int]): Parent id of the given node.
        children: (Dict[int, List[int, int]]): Children of the given node.
        ancestors: (Dict[int, List[*int]]): Ancestors of the given node.
        descendants: (Dict[int, List[*int]]): Descendants of the given node.
        paths (List[Path]): Paths from root to each Leaf. Leaves not included. Path hyperplanes are already
                             flipped accordingly.
        paths_ids (List[List[int]]): Ids of the nodes composing each path.
        paths_label_ids (List[int]): Ids of the Leaf of each path.
    """
    def __init__(self, store_data: bool = True):
        self.root = None
        self.nodes = {}
        self.parent = {}
        self.children = {}
        self.ancestors = {}
        self.descendants = {}

        self.paths = list()
        self.paths_ids = list()
        self.path_labels = list()
        self.path_labels_ids = list()
        self.store_data = store_data

    def __eq__(self, other):
        if not type(other) is type(self) or self.root != other.root:
            return False
        return self.__rec__eq__(self.root.children[0], other.root.children[0]) and \
            self.__rec__eq__(self.root.children[1], other.root.children[1])

    def __hash__(self):
        return sum(hash(node) for node in self.nodes)

    def __rec__eq__(self, node: Node, other_node: Node):
        if isinstance(node, Leaf) and not isinstance(other_node, Leaf) or \
                not isinstance(node, Leaf) and isinstance(other_node, Leaf):
            return False
        if isinstance(node, Leaf) and isinstance(other_node, Leaf):
            return node == other_node

        return self.__rec__eq__(node.children[0], other_node.children[0]) and \
            self.__rec__eq__(node.children[1], other_node.children[1])

    def should_stop_for_max_depth(self, depth: int, max_depth: int, parent_node: Optional[Node],
                                  validation_labels: numpy.array, validation_data: numpy.ndarray) -> bool:
        """Stop the induction for maximum depth reached?

        Args:
            depth: The current depth
            max_depth: Maximum depth allowed.
            parent_node: The parent node of the node induced.
            validation_labels: Labels of the data on which the node is being induced.
            validation_data: Data on which the node is being induced.

        Returns:
            True if the maximum depth has been reached, False otherwise. If reached, it also creates the appropriate
            leaves.
        """
        if depth >= max_depth - 1:
            logging.debug(f"Reached max depth of {max_depth}")
            children = [self.build_leaf(validation_labels, data=validation_data), self.build_leaf(1 - validation_labels, data=validation_data)]
            parent_node.children = children

            return True

        return False

    def should_stop_for_min_eps(self, parent_node: InternalNode, candidate_node: InternalNode,
                                node_fitness_function: Callable,
                                validation_data: numpy.ndarray, validation_labels: numpy.array,
                                classes: numpy.array,
                                min_eps: float = 0.000001) -> bool:
        """Stop the induction because there's not enough improvement?

        Args:
            parent_node: The parent node of the node induced.
            candidate_node: The induced node.
                        validation_labels: Labels of the data on which the node is being induced.
            validation_data: Data on which the node is being induced.
            validation_labels: Labels of the data on which the node is being induced.
            classes: Set of labels.
            node_fitness_function: Function to evaluate the improvement.
            min_eps: The minimum tolerated improvement. Defaults to 0.000001.

        Returns:
            True if the improvement is below `min_eps`, False otherwise.
        """
        parent_fitness = node_fitness_function(parent_node,
                                               data=parent_node._data, labels=parent_node._labels,
                                               classes=classes)
        node_fitness = node_fitness_function(candidate_node, data=validation_data, labels=validation_labels,
                                             classes=classes)
        return parent_fitness - node_fitness < min_eps

    def should_stop_for_one_class(self, validation_labels: numpy.array, parent_node: Node, direction: str,
                                  validation_data: numpy.ndarray) -> bool:
        """Should induction stop because there's not enough classes to induce on?

        Args:
            validation_labels: Labels of the data on which the node is being induced.
            parent_node: The node parent of the one being induced.
            direction: Direction of the node.
            validation_data: Data on which the node is being induced.

        Returns:
            True if should stop, False otherwise. If True, also creates the appropriate Leaf.
        """
        if numpy.unique(validation_labels).size == 1:
            child_index = 0 if direction == "left" else 1
            parent_node.children[child_index] = self.build_leaf(validation_labels, data=validation_data)

            return True

        return False

    def should_stop_for_min_samples(self, validation_data: numpy.ndarray, validation_labels: numpy.ndarray,
                                    direction: str, parent_node: InternalNode,
                                    min_samples: int) -> bool:
        """Should induction stop because there's not enough data to induce on?

        Args:
            validation_data: Data on which the node is being induced.
            validation_labels: Labels of the data on which the node is being induced.
            parent_node: The node parent of the one being induced.
            direction: Direction of the node.
            min_samples: Minimum number of tolerated samples.

        Returns:
            True if should stop, False otherwise. If True, also creates the appropriate Leaf.
        """
        if validation_data.shape[0] < min_samples:
            logging.debug(f"Reached minimum samples of {min_samples}")
            child_index = 0 if direction == "left" else 1
            parent_node.children[child_index] = self.build_leaf(validation_labels, data=validation_data)

            return True

        return False

    def fit(self, data: numpy.ndarray, labels: numpy.ndarray,
            max_depth: int = numpy.inf, min_eps: float = 0.00000001, min_samples: int = 10,
            node_fitness_function: Optional[Callable] = None,
            node_hyperparameters: Optional[Dict] = None,
            **step_hyperparameters) -> Self:
        """Learn a Tree on the given data.

        Args:
            data: The training set.
            labels: The training set labels.
            max_depth: Maximum depth of the Decision Tree.
            min_eps: Minimum improve in the learning metric to keep on creating new nodes.
            min_samples: Minimum number of samples in leaves.
            node_fitness_function: Fitness function used  to evaluate node candidates, nodes minimizing it are selected.
            node_hyperparameters: Optional training hyperparameters to train the internal nodes.

        Returns:
            This Tree, fit to the given `data` and `labels`.
        """
        # create root
        logging.debug("Fitting tree with:")
        logging.debug(f"\tmax depth: {max_depth}")
        logging.debug(f"\tmin eps: {min_eps}")

        os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"

        logging.debug("Fitting root...")
        classes = numpy.unique(labels)
        self.node_hyperparameters = node_hyperparameters
        root = self.step(parent_node=None,
                         data=data, labels=labels, classes=classes,
                         node_fitness_function=node_fitness_function,
                         min_eps=min_eps, min_samples=min_samples, max_depth=max_depth,
                         direction=None,
                         node_hyperparameters={} if node_hyperparameters is None else node_hyperparameters,
                         **step_hyperparameters)
        self.root = root

        left_child_indices = root(data)
        left_child_data = data[left_child_indices]
        left_child_labels = labels[left_child_indices]
        right_child_data = data[~left_child_indices]
        right_child_labels = labels[~left_child_indices]

        logging.debug("Fitting left child...")
        self.__rec_fit(parent_node=root,
                       data=left_child_data, labels=left_child_labels, classes=classes,
                       node_fitness_function=node_fitness_function,
                       direction="left",
                       depth=1, max_depth=max_depth, min_eps=min_eps, min_samples=min_samples,
                       node_hyperparameters={} if node_hyperparameters is None else node_hyperparameters,
                       **step_hyperparameters)
        logging.debug("Fitting right child...")
        self.__rec_fit(parent_node=root,
                       data=right_child_data, labels=right_child_labels, classes=classes,
                       node_fitness_function=node_fitness_function,
                       direction="right",
                       depth=1, max_depth=max_depth, min_eps=min_eps, min_samples=min_samples,
                       node_hyperparameters={} if node_hyperparameters is None else node_hyperparameters,
                       **step_hyperparameters)
        self.build()

        return self

    def __rec_fit(self, parent_node: Optional[InternalNode],
                  data: numpy.ndarray, labels: numpy.ndarray, classes: numpy.ndarray,
                  direction: str, depth: int = 0, max_depth: int = numpy.inf,
                  min_eps: float = 0.00001, min_samples: int = 100,
                  node_fitness_function: Optional[Callable] = None,
                  node_hyperparameters: Optional[Dict] = None,
                  **step_hyperparameters):
        """Recursively fit the Tree rooted in `parent_node`.

        Args:
            data: The training set available.
            labels: The training set labels for `data`.
            classes: The training set classes.
            parent_node: The parent node.

            depth: Depth of the root of the current subtree.
            max_depth: Maximum depth of the Decision Tree.
            min_eps: Minimum improve in the learning metric to keep on creating new nodes.
            min_samples: Minimum number of samples in leaves.
            node_fitness_function: Fitness function used  to evaluate node candidates, nodes minimizing it are selected.
                                   Defaults to gini.
            node_hyperparameters: Optional training hyperparameters to train the internal nodes.
            step_hyperparameters: Keyword parameters for the `step` function.
        """
        if direction == "left" and parent_node.children[0] is not None:
            return
        if direction == "right" and parent_node.children[1] is not None:
            return

        # stopping criteria
        should_stop_for_depth = self.should_stop_for_max_depth(depth, max_depth, parent_node, labels, validation_data=data)
        should_stop_for_min_samples = self.should_stop_for_min_samples(data, labels, direction, parent_node, min_samples)
        should_stop_for_one_class = self.should_stop_for_one_class(labels, parent_node, direction, validation_data=data)

        if not should_stop_for_depth and not should_stop_for_min_samples and not should_stop_for_one_class:
            step_result = self.step(parent_node=parent_node,
                                    data=data, labels=labels, classes=classes,
                                    min_eps=min_eps, max_depth=max_depth, min_samples=min_samples,
                                    node_fitness_function=node_fitness_function,
                                    node_hyperparameters=node_hyperparameters,
                                    **step_hyperparameters)

            if isinstance(step_result, Leaf):
                if direction == "left":
                    parent_node.children[0] = step_result
                else:
                    parent_node.children[1] = step_result
            else:
                fit_node = step_result
                child_index = 0 if direction == "left" else 1
                parent_node.children[child_index] = fit_node

                # recurse on children
                data_indices = fit_node(data)
                left_child_data, left_child_labels = data[data_indices], labels[data_indices]
                right_child_data, right_child_labels = data[~data_indices], labels[~data_indices]

                logging.debug(f"Fitting child on {left_child_data.shape[0]} nodes...")
                self.__rec_fit(parent_node=fit_node,
                               data=left_child_data, labels=left_child_labels, classes=classes,
                               direction="left",
                               depth=depth + 1, max_depth=max_depth, min_eps=min_eps, min_samples=min_samples,
                               node_fitness_function=node_fitness_function,
                               node_hyperparameters=node_hyperparameters,
                               step_hyperparameters=step_hyperparameters)
                logging.debug(f"Fitting child on {right_child_data.shape[0]} nodes...")
                self.__rec_fit(parent_node=fit_node,
                               data=right_child_data, labels=right_child_labels, classes=classes,
                               direction="right",
                               depth=depth + 1, max_depth=max_depth, min_eps=min_eps, min_samples=min_samples,
                               node_fitness_function=node_fitness_function,
                               node_hyperparameters=node_hyperparameters,
                               step_hyperparameters=step_hyperparameters)

    def __deepcopy__(self, memodict={}):
        copied_nodes = self._deepcopy_tree()
        copied_tree = type(self)(copied_nodes[1])  # type(self) to invoke the subclass constructor
        # copied_tree = ObliqueTree(copied_nodes[0])
        copied_tree.build()

        return copied_tree

    def json(self):
        tree_json = dict()
        tree_json["1"] = {"node": self.root.json(), "parent": None}
        tree_json.update(self.__rec_json(self.root.children[0], index=2, parent=1))
        tree_json.update(self.__rec_json(self.root.children[1], index=3, parent=1))

        return tree_json

    def __rec_json(self, node: Node, index: int, parent: int):
        tree_json = {str(index): {"node": node.json(), "parent": parent}}
        if isinstance(node, InternalNode):
            update_left = self.__rec_json(node.children[0], index * 2, index)
            update_right = self.__rec_json(node.children[1], index * 2 + 1, index)
            tree_json.update(update_left)
            tree_json.update(update_right)

        return tree_json

    def coverage(self, data: numpy.ndarray) -> numpy.ndarray:
        """
        Compute the coverage matrix of the given data.
        Args:
            data: The data.

        Returns:
            A #paths x #records boolean matrix whose `i, j` entry is True if `self.paths[i]` covers record `j`,
            and `0` otherwise.
        """
        return numpy.array([p(data) for p in self.paths])

    def predict(self, data: numpy.ndarray) -> numpy.ndarray:
        """Predict the given data `x` by routing it along the tree."""
        coverage_matrix_transposed = self.coverage(data).transpose()
        hitting_paths = numpy.argwhere(coverage_matrix_transposed)[:, 1]
        path_predictions = [numpy.argmax(self.nodes[leaf].label) for leaf in self.paths_label_ids]
        predictions = numpy.array([path_predictions[i] for i in hitting_paths])

        return predictions

    def store(self, node: Node, data: numpy.ndarray, labels: numpy.array):
        if self.store_data:
            node._data = data
            node._labels = labels
        else:
            node._data = None
            node._labels = None

    @abstractmethod
    def step(self, parent_node: Optional[InternalNode],
             data: numpy.ndarray, labels: numpy.ndarray, classes: numpy.ndarray,
             direction: Optional[str] = None, depth: int = 1,
             min_eps: float = 0.000001, max_depth: int = 16, min_samples: int = 10,
             node_fitness_function: Optional[Callable] = None,
             node_hyperparameters: Optional[Dict] = None, **step_hyperparameters) -> Optional[Node]:
        """Compute a learning step, i.e., learn a hyperplane for the given `data` and `labels`, with parent node
        `parent_node`.

        Args:
            parent_node: The subtree to fit.
            data: The data routed to the node to learn.
            labels: The labels of the data.
            classes: The set of classes of the whole dataset.
            direction: Direction towards which the step is taken. Either "left" or "right".
            depth: Depth of the induced node.
            min_eps: Minimum improve in the learning metric to keep on creating new nodes.
            max_depth: Maximum depth reachable by the Tree.
            min_samples: Minimum number of samples in leaves.
            node_fitness_function: Fitness function used  to evaluate node candidates, nodes minimizing it are selected.
            node_hyperparameters: Optional training hyperparameters to train the internal nodes.
            step_hyperparameters: Additional hyperparameters for the function.

        Returns:
            The learned node, either an InternalNode, if the Tree should be expanded, or a Leaf. If the learning
            procedure is unsuccessful, None is returned instead.
        """
        pass

    def depth_first_accumulate(self, node: InternalNode | Leaf, foo: callable, accumulated: List):
        """Apply `foo` to each node in this tree in a depth-first manner, accumulate its results

        Args:
            node: The current node to which apply `foo`
            foo: The function to apply
            accumulated: The accumulated results
        """
        if isinstance(node, Leaf):
            return accumulated
        else:
            accumulated.append(foo(node))
            accumulated += self.depth_first_accumulate(node.children[0], foo, accumulated)
            accumulated += self.depth_first_accumulate(node.children[1], foo, accumulated)

            return accumulated

    def _deepcopy_tree(self):
        """Deepcopy this Tree's nodes."""
        return {node_id: copy.deepcopy(self.nodes[node_id]) for node_id in self.nodes}

    def path_to_node(self, node_id: int) -> BinaryPath:
        """Get a path to the given node. The node itself is included!

        Args:
            node_id: The ID of the node.

        Returns:
            A path to said node. The node itself is included.
        """
        raw_path = BinaryPath([self.nodes[node] for node in self.ancestors[node_id]])
        is_right_child = [node.direction == "right" for node in raw_path[1:]]  # ancestors, root excluded
        flip_mask = is_right_child + [self.nodes[node_id].direction == "right"]

        return raw_path.flip(flip_mask)

    def build(self):
        """Build internal representation of the Tree, creates attributes `ancestors`, `nodes`, etc."""
        self._compile_node_infos()  # construct self.nodes, self.parent, self.depth, and node direction and node_id
        self._build_structure()
        self._build_paths()  # construct self.path

    def _build_structure(self):
        self.children = {
            node_id: sorted([node.node_id for node in node.children])
            for node_id, node in self.nodes.items()
        }

        # build ancestors
        self.ancestors = {1: set()}
        for node_id in self.nodes:
            self.ancestors[node_id] = {1, self.parent[node_id]}
            current_ancestor_id = node_id
            while current_ancestor_id != 1:
                parent_id = self.parent[current_ancestor_id]
                self.ancestors[node_id].add(parent_id)
                current_ancestor_id = parent_id
        self.ancestors = {key: sorted(value) for key, value in self.ancestors.items()}

        # build descendants
        self.descendants = {1: set(self.nodes.keys())}
        for node_id in self.nodes:
            self.descendants[node_id] = self._descendants(node_id, current_descendants=list())
        self.descendants = {key: sorted(value)[1:] for key, value in self.descendants.items()}

    def _descendants(self, node_id: int, current_descendants: List) -> Set[int]:
        if isinstance(self.nodes[node_id], Leaf):
            return set(current_descendants + [node_id])
        return (self._descendants(node_id * 2, current_descendants + [node_id]) |
                self._descendants(node_id * 2 + 1, current_descendants + [node_id]))

    def _compile_node_infos(self):
        root = self.root
        root.node_id = 1
        root.direction = ""

        self.nodes = {1: root}
        self.depth = {1: 1}
        self.parent = {1: 1}

        self.__rec_gather_nodes(root.children[0], 2, "left", 2, 1)
        self.__rec_gather_nodes(root.children[1], 3, "right", 2, 1)

    def __rec_gather_nodes(self, node: Node, index: int, direction: str = "", depth: int = 1, parent_id: int = 1):
        self.nodes.update({index: node})

        node.node_id = index
        node.direction = direction

        self.depth[index] = depth
        self.parent[index] = parent_id

        if isinstance(node, InternalNode):
            self.__rec_gather_nodes(node.children[0], index * 2, "left", depth + 1, index)
            self.__rec_gather_nodes(node.children[1], index * 2 + 1, "right", depth + 1, index)

    def _build_paths(self):
        """Construct the paths in this tree, starting from the root. These paths will then be used in prediction phase."""
        leaves_ids = [node_id for node_id, node in self.nodes.items() if isinstance(node, Leaf)]
        paths_to_leaves = [list(self.ancestors[leaf]) + [leaf] for leaf in leaves_ids]
        paths_to_leaves = [[self.nodes[node_in_path] for node_in_path in path] for path in paths_to_leaves]
        # by construction left children (direction == "left") lie in the hyperplane, while right children don't,
        # hence need to invert the paths routing towards right children
        adjusted_paths = list()
        adjusted_paths_ids = list()
        for path_id, path in enumerate(paths_to_leaves):
            adjusted_path = list()
            adjusted_path_ids = list()
            for i, node in enumerate(path):
                if i > 0:
                    if node.direction == "right":
                        # right child: the parent hyperplane was false, hence need to invert it
                        adjusted_path.append(~path[i - 1].hyperplane)
                        adjusted_path_ids.append(-path[i - 1].node_id)
                    else:
                        adjusted_path.append(copy.deepcopy(path[i - 1].hyperplane))
                        adjusted_path_ids.append(path[i - 1].node_id)
            adjusted_paths.append(adjusted_path)
            adjusted_paths_ids.append(adjusted_path_ids)

        self.paths = [OBinaryPath(*p) for p in adjusted_paths]
        self.paths_ids = adjusted_paths_ids
        self.paths_label_ids = leaves_ids
        self.paths_labels = numpy.array([numpy.argmax(p[-1].label) for p in paths_to_leaves])

    def build_leaf(self, labels: numpy.array, data: Optional[numpy.ndarray] = None) -> Leaf:
        """Build a Leaf for the given labels."""
        probabilities = numpy.bincount(labels) / labels.size

        if probabilities.size == 0:
            probabilities = numpy.array([0.5, 0.5])
        elif probabilities.size == 1:
            probabilities = numpy.zeros(2, )
            probabilities[int(labels[0])] = 1.

        leaf = Leaf(probabilities, labels=labels)
        leaf._data = data

        return leaf


class ObliqueTree(Tree):
    """Oblique Tree with multivariate splits."""
    def __init__(self, root: Optional[InternalNode] = None, store_data: bool = True):
        super(ObliqueTree, self).__init__(store_data)
        self.root = root
        self.path_matrices = list()
        self.path_labels = list()

    def step(self, parent_node: Optional[InternalNode], data: numpy.ndarray, labels: numpy.ndarray,
             classes: numpy.ndarray, direction: Optional[str] = None, depth: int = 1, min_eps: float = 0.000001,
             max_depth: int = 16, min_samples: int = 10, node_fitness_function: Optional[Callable] = None,
             node_hyperparameters: Optional[Dict] = None, **step_hyperparameters) -> Optional[Node]:
        pass

    @staticmethod
    def from_json(json_file: str) -> ObliqueTree:
        """Extract a tree from the given `json_file`.

        Args:
            json_file: Path to the JSON file encoding the tree.

        Returns:
            The tree encoded in the `json_file`
        """
        with open(json_file, "r") as log:
            json_obj = json.load(log)

        nodes = {int(k): Node.from_json(json_obj[k]["node"]) for k in json_obj.keys()}
        root = nodes[1]
        ObliqueTree.__rec_from_json(root, 1, nodes)
        tree = ObliqueTree(root)

        tree.build()

        return tree

    @staticmethod
    def __rec_from_json(node: InternalNode | Leaf, node_id: int, nodes: Dict):
        left_child, right_child = node_id * 2, node_id * 2 + 1
        if isinstance(node, InternalNode):
            # internal node
            if left_child in nodes:
                node.children[0] = nodes[left_child]
                ObliqueTree.__rec_from_json(node.children[0], left_child, nodes)
            if right_child in nodes:
                node.children[1] = nodes[right_child]
                ObliqueTree.__rec_from_json(node.children[1], right_child, nodes)
        else:
            return


class ParallelTree(ObliqueTree):
    """Axis-parallel tree with univariate splits."""
    def __init__(self, root: Optional[InternalNode] = None, store_data: bool = True):
        super(ParallelTree, self).__init__(store_data)
        self.root = root
        self.path_matrices = list()
        self.path_labels = list()

    def candidates(self, data: numpy.ndarray) -> Sequence[APHyperplane]:
        """Computes a set of candidates to provide to select from.

        Args:
            data: The data to which extract the candidates from.
            exhaustive: True to enumerate all possible candidates, False otherwise. Defaults to True.

        Returns:
            An iterable of `APHyperplanes`.
        """
        unique_values_per_feature = list()
        for feature in range(data.shape[1]):
            values = numpy.unique(data[:, feature])
            if values.size > 10:
                values = numpy.unique(numpy.quantile(data[:, feature], numpy.arange(0., 1., 0.25)))
                try:
                    assert values.size <= 10
                except AssertionError  as e:
                    raise e

            unique_values_per_feature.append(values)

        candidates = [[APHyperplane(feature, -numpy.inf, threshold) for threshold in thresholds]
                      for feature, thresholds in enumerate(unique_values_per_feature)]
        candidates = list(flatten(candidates))

        return candidates


    def step(self, parent_node: Optional[InternalNode],
             data: numpy.ndarray, labels: numpy.ndarray, classes: numpy.ndarray,
             direction: Optional[str] = None, depth: int = 1,
             min_eps: float = 0.000001, max_depth: int = 16, min_samples: int = 10,
             node_fitness_function: Optional[Callable] = None,
             node_hyperparameters: Optional[Dict] = None,
             **step_hyperparameters) -> Optional[Node]:
        # stopping criteria
        should_stop_for_depth = self.should_stop_for_max_depth(depth, max_depth, parent_node, labels,
                                                               validation_data=data)
        should_stop_for_one_class = self.should_stop_for_one_class(labels, parent_node, direction, validation_data=data)
        should_stop_for_min_samples = self.should_stop_for_min_samples(data, labels, direction, parent_node,
                                                                       min_samples)

        if not should_stop_for_depth and not should_stop_for_min_samples and not should_stop_for_one_class:
            candidates = self.candidates(data)
            hyperplane = candidates[numpy.argmin([node_fitness_function(candidate, data, labels, classes)
                                                  for candidate in candidates])]

            if parent_node is not None:
                if self.should_stop_for_min_eps(parent_node, InternalNode(hyperplane),
                                                node_fitness_function=node_fitness_function,
                                                validation_data=data, validation_labels=labels, classes=classes,
                                                min_eps=min_eps):
                    fit_node = self.build_leaf(labels, data=data)
                else:
                    fit_node = InternalNode(hyperplane)
            else:
                fit_node = InternalNode(hyperplane)

        else:
            fit_node = self.build_leaf(labels, data=data)

        self.store(fit_node, data, labels)

        return fit_node

    @staticmethod
    def from_json(json_file: str) -> ParallelTree:
        """Extract a tree from the given `json_file`.

        Args:
            json_file: Path to the JSON file encoding the tree.

        Returns:
            The tree encoded in the `json_file`
        """
        with open(json_file, "r") as log:
            json_obj = json.load(log)

        nodes = {int(k): Node.from_json(json_obj[k]["node"]) for k in json_obj.keys()}
        root = nodes[1]
        ParallelTree.__rec_from_json(root, 1, nodes)
        tree = ParallelTree(root)

        tree.build()

        return tree

    @staticmethod
    def __rec_from_json(node: InternalNode | Leaf, node_id: int, nodes: Dict):
        left_child, right_child = node_id * 2, node_id * 2 + 1
        if isinstance(node, InternalNode):
            # internal node
            if left_child in nodes:
                node.children[0] = nodes[left_child]
                ParallelTree.__rec_from_json(node.children[0], left_child, nodes)
            if right_child in nodes:
                node.children[1] = nodes[right_child]
                ParallelTree.__rec_from_json(node.children[1], right_child, nodes)
        else:
            return