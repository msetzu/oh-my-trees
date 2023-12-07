"""Rules comprising premises. A Rule is defined as a list of Premises, each Premise a Hyperplane."""
from __future__ import annotations

from typing import List, Union, Tuple

import numpy as numpy
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree._tree import Tree

from ohmt.planes import APHyperplane
from . import APBinaryPath


def from_cart(cart: DecisionTreeClassifier) -> List[Tuple[APBinaryPath, int]]:
    """Extract systems from the features of a sklearn.tree.DecisionTreeClassifier.
    Adapted from https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
    Args:
        cart: The Decision Tree whose systems to extract
    Returns:
        The list of axis-parallel systems encoded by the `cart` Decision Tree
    """
    tree_paths = __all_paths(cart.tree_)
    tree_paths = list(filter(lambda path: len(path) > 1, tree_paths))
    features = [list(map(lambda node: cart.tree_.feature[abs(node)], path[:-1])) for path in tree_paths]
    features = [item for sublist in features for item in sublist]  # flatten list of lists
    thresholds = [list(map(lambda node: cart.tree_.threshold[abs(node)], path[:-1])) for path in tree_paths]
    leaves = [i for i in range(cart.tree_.node_count) if cart.tree_.children_left[i] == cart.tree_.children_right[i]]
    labels = {leaf: (cart.tree_.value[leaf][0]).argmax() for leaf in leaves}

    cart_rules = list()
    for feature_list, thresholds_list, path in zip(features, thresholds, tree_paths):
        if abs(path[-1]) not in leaves:
            continue

        rule_premises = {}
        # rule_features = features
        rule_features = [abs(cart.tree_.feature[p]) for p in path[:-1]]
        rule_label = labels[abs(path[-1])]

        # thresholds_ = thresholds[:-1]
        indices_per_feature = {feature: numpy.argwhere(rule_features == feature).flatten() for feature in rule_features}
        directions_per_feature = {f: [numpy.sign(path[k + 1]) for k in indices_per_feature[f] if k < len(path) - 1]
                                  for f in rule_features}

        for feature in rule_features:
            if len(indices_per_feature[feature]) == 1:
                threshold = thresholds[indices_per_feature[feature][0]][0]
                rule_premises[feature] = (-numpy.inf, threshold) if directions_per_feature[feature][0] < 0\
                                                                    else (threshold, numpy.inf)
            else:
                lower_bounds_idx = [index for index, direction in zip(indices_per_feature[feature],
                                                                      directions_per_feature[feature]) if direction > 0]
                upper_bounds_idx = [index for index, direction in zip(indices_per_feature[feature],
                                                                      directions_per_feature[feature]) if direction < 0]
                lower_bounds, upper_bounds = (numpy.array([thresholds[lower_idx] for lower_idx in lower_bounds_idx]),
                                              numpy.array([thresholds[upper_idx] for upper_idx in upper_bounds_idx]))

                if lower_bounds.shape[0] > 0 and upper_bounds.shape[0] > 0:
                    rule_premises[feature] = (max(lower_bounds), min(upper_bounds))
                elif lower_bounds.shape[0] == 0:
                    rule_premises[feature] = (-numpy.inf, min(upper_bounds).item())
                elif upper_bounds.shape[0] == 0:
                    rule_premises[feature] = (max(lower_bounds).item(), +numpy.inf)

        rule_premises = [APHyperplane(feat, low, upp) for (feat, (low, upp)) in rule_premises.items()]
        cart_rules.append((APBinaryPath(*rule_premises), rule_label))

    return cart_rules


def __all_paths(tree: Tree) -> Union[List[Tuple], List[List[int]]]:
    """Retrieve all the possible paths in @tree.

    Arguments:
        tree: The decision tree internals.

    Returns:
        A list of lists of indices:[path_1, path_2, .., path_m] where path_i = [node_1, node_l].
    """
    paths = [[0]]
    left_child = tree.children_left[0]
    right_child = tree.children_right[0]

    if tree.capacity == 1:
        return paths

    paths = paths + \
            __rec_all_paths(tree, right_child, [0], +1) + \
            __rec_all_paths(tree, left_child, [0], -1)
    paths = sorted(set(map(tuple, paths)), key=lambda p: len(p))

    return paths


def __rec_all_paths(tree: Tree, node: int, path: List, direction: int):
    """Recursive call for the @all_paths function.

    Arguments:
        tree: The decision tree internals.
        node: The node whose path to expand.
        path: The path root-> `node`.
        direction:  +1 for right child, -1 for left child. Used to store the actual traversal.

    Returns:
        The enriched path.
    """
    # Leaf
    if tree.children_left[node] == tree.children_right[node]:
        return [path + [node * direction]]
    else:
        path_ = [path + [node * direction]]
        l_child = tree.children_left[node]
        r_child = tree.children_right[node]

        return path_ + \
               __rec_all_paths(tree, r_child, path_[0], +1) + \
               __rec_all_paths(tree, l_child, path_[0], -1)
