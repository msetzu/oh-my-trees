import numpy
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

from ohmt.planes import Hyperplane
from ohmt.trees.structure.trees import InternalNode, Tree


#####################
### Gini impurity ###
#####################
def gini(partitioner: Hyperplane | InternalNode | Tree, data: numpy.ndarray, labels: numpy.ndarray,
         classes: numpy.ndarray) -> float:
    """Compute Gini for the given `partitioner` on the given `data` and `labels`.


    Args:
        partitioner: The object partitioning the given data into partitions. If an Hyperplane, a simple Gini
                     split is computed. If an InternalNode, or a Tree, then the induced partitions are given by fully
                     following the Node/Tree, that is, each Path from the Node/Tree to some leaves induces a partition,
                     and the metric is computed on these partitions.
        data: The data to compute the gini on.
        labels: The labels to compute the gini on.
        classes: Set of classes.

    Returns:
        The gini for the given hyperplane.
    """
    if isinstance(partitioner, Hyperplane):
        return _gini_hyperplane(partitioner, data, labels, classes)
    if isinstance(partitioner, InternalNode):
        return _gini_hyperplane(partitioner.hyperplane, data, labels, classes)
    elif isinstance(partitioner, (InternalNode, Tree)):
        return _gini_tree(partitioner, data, labels, classes)


def _gini_hyperplane(hyperplane: Hyperplane, data: numpy.ndarray, labels: numpy.ndarray,
         classes: numpy.ndarray) -> float:
    """Compute Gini for the given `hyperplane` on the given `data` and `labels`.

    Args:
        hyperplane: The hyperplane whose gini to compute.
        data: The data to compute the gini on.
        labels: The labels to compute the gini on.
        classes: Set of classes.

    Returns:
        The gini for the given hyperplane.
    """
    if labels.size == 0:
        return 0

    partitions = hyperplane(data).astype(int)

    return _gini(partitions, labels, classes)
    # coverage = hyperplane(data)
    # negated_coverage = ~coverage
    # gini_weights = coverage.sum() / labels.size, negated_coverage.sum() / labels.size
    #
    # # gini per split
    # covered_squared_purities = [((labels[coverage] == c).sum() / coverage.size) ** 2
    #                               if coverage.sum() > 0 else 0
    #                             for c in classes]
    # noncovered_squared_purities = [((labels[negated_coverage] == c).sum() / negated_coverage.size) ** 2
    #                                  if negated_coverage.sum() > 0 else 0
    #                                for c in classes]
    #
    # # weighted gini
    # covered_gini = (1 - sum(covered_squared_purities))
    # noncovered_gini = (1 - sum(noncovered_squared_purities))
    # impurity = gini_weights[0] * covered_gini + gini_weights[1] * noncovered_gini
    #
    # return impurity

def _gini_tree(tree: Tree, data: numpy.ndarray, labels: numpy.ndarray,
               classes: numpy.ndarray) -> float:
    """Compute Gini for the given `hyperplane` on the given `data` and `labels`.

    Args:
        tree: The tree whose gini to compute.
        data: The data to compute the gini on.
        labels: The labels to compute the gini on.
        classes: Set of classes.

    Returns:
        The gini for the given hyperplane.
    """
    if labels.size == 0:
        return 0

    paths = tree.paths
    partitions = numpy.repeat(numpy.nan, labels.size)

    for path_id, path in enumerate(paths):
        partitions[path(data)] = path_id

    assert partitions[numpy.isnan(partitions)].size == 0

    return _gini(partitions, labels, classes)

def _gini(partitions: numpy.ndarray, labels: numpy.ndarray, classes: numpy.ndarray) -> float:
    """Compute Gini impurity on the given `partitions`.

    Args:
        partitions: Indices of induced partitions. `partitions[i]` holds the partition number induced on `data[i]`.
        labels: The labels to compute the gini on.
        classes: Set of classes.

    Returns:
        The gini for the given partitions.
    """
    if labels.size == 0:
        return 0

    unique_partitions, partitions_sizes = numpy.unique(partitions, return_counts=True)
    partition_ginis = list()
    for partition, partitions_size in zip(unique_partitions, partitions_sizes):
        partition_ginis.append(numpy.array([(labels[partition == partitions] == class_of_interest).sum() / partitions_size
                                for class_of_interest in classes]) ** 2)
    partition_impurities_per_class = numpy.array(partition_ginis)
    unweighted_partition_ginis = 1 - partition_impurities_per_class.sum(axis=1).squeeze()  # sum over classes
    partition_ginis = (partitions_sizes / partitions.size) * unweighted_partition_ginis
    computed_gini = sum(partition_ginis)

    return computed_gini


########################
### Information gain ###
########################
def information_gain(partitioner: Hyperplane | InternalNode | Tree, data: numpy.ndarray, labels: numpy.ndarray,
         classes: numpy.ndarray) -> float:
    """Compute Information Gain for the given `partitioner` on the given `data` and `labels`.


    Args:
        partitioner: The object partitioning the given data into partitions. If an Hyperplane, a simple Gini
                     split is computed. If an InternalNode, or a Tree, then the induced partitions are given by fully
                     following the Node/Tree, that is, each Path from the Node/Tree to some leaves induces a partition,
                     and the metric is computed on these partitions.
        data: The data to compute the gini on.
        labels: The labels to compute the gini on.
        classes: Set of classes.

    Returns:
        The information gain for the given hyperplane.
    """
    if isinstance(partitioner, Hyperplane):
        return _information_gain_on_hyperplane(partitioner, data, labels, classes)
    if isinstance(partitioner, InternalNode):
        return _information_gain_on_hyperplane(partitioner.hyperplane, data, labels, classes)
    elif isinstance(partitioner, (InternalNode, Tree)):
        return _information_gain_on_tree(partitioner, data, labels, classes)


def _information_gain_on_hyperplane(hyperplane: Hyperplane, data: numpy.ndarray, labels: numpy.ndarray,
         classes: numpy.ndarray) -> float:
    """Compute Gini for the given `hyperplane` on the given `data` and `labels`.

    Args:
        hyperplane: The hyperplane whose gini to compute.
        data: The data to compute the gini on.
        labels: The labels to compute the gini on.
        classes: Set of classes.

    Returns:
        The gini for the given hyperplane.
    """
    if labels.size == 0:
        return 0

    partitions = hyperplane(data).astype(int)

    return _gini(partitions, labels, classes)

def _information_gain_on_tree(tree: Tree, data: numpy.ndarray, labels: numpy.ndarray,
               classes: numpy.ndarray) -> float:
    """Compute Gini for the given `hyperplane` on the given `data` and `labels`.

    Args:
        tree: The tree whose gini to compute.
        data: The data to compute the gini on.
        labels: The labels to compute the gini on.
        classes: Set of classes.

    Returns:
        The gini for the given hyperplane.
    """
    if labels.size == 0:
        return 0

    paths = tree.paths
    partitions = numpy.repeat(numpy.nan, labels.size)

    for path_id, path in enumerate(paths):
        partitions[path(data)] = path_id

    assert partitions[numpy.isnan(partitions)].size == 0

    return _information_gain(partitions, labels, classes)

def _information_gain(partitions: numpy.ndarray, labels: numpy.ndarray, classes: numpy.ndarray) -> float:
    """Compute the Information Gain for the given `hyperplane` on the given `data` and `labels`.

    Args:
        partitions: The induced partitions.
        labels: The labels to compute the gini on.
        classes: Set of classes.

    Returns:
        The gini for the given hyperplane.
    """
    unique_partitions, partitions_sizes = numpy.unique(partitions, return_counts=True)
    partitions_information = list()
    for partition, partitions_size in zip(unique_partitions, partitions_sizes):
        partitions_information.append(
            numpy.array([(labels[partition == partitions] == class_of_interest).sum() / partitions_size
                         for class_of_interest in classes]))
    partition_impurities_per_class = numpy.array(partitions_information)
    unweighted_partition_information = partition_impurities_per_class * numpy.log(partition_impurities_per_class)
    unweighted_partition_information = - unweighted_partition_information.sum(axis=1).squeeze()  # sum over classes
    partitions_information = (partitions_sizes / partitions.size) * unweighted_partition_information
    gain = sum(partitions_information)

    return gain

###############
### Entropy ###
###############
def entropy(partitioner: Hyperplane | InternalNode | Tree, data: numpy.ndarray, labels: numpy.ndarray,
         classes: numpy.ndarray) -> float:
    """Compute Gain Ratio for the given `partitioner` on the given `data` and `labels`.

    Args:
        partitioner: The object partitioning the given data into partitions. If an Hyperplane, a simple Gini
                     split is computed. If an InternalNode, or a Tree, then the induced partitions are given by fully
                     following the Node/Tree, that is, each Path from the Node/Tree to some leaves induces a partition,
                     and the metric is computed on these partitions.
        data: The data to compute the gini on.
        labels: The labels to compute the gini on.
        classes: Set of classes.

    Returns:
        The information gain for the given hyperplane.
    """
    if isinstance(partitioner, Hyperplane):
        return _entropy_on_hyperplane(partitioner, data, labels, classes)
    if isinstance(partitioner, InternalNode):
        return _entropy_on_hyperplane(partitioner.hyperplane, data, labels, classes)
    elif isinstance(partitioner, (InternalNode, Tree)):
        return _entropy_on_tree(partitioner, data, labels, classes)

    if labels.size == 0:
        return 0

def _entropy_on_hyperplane(hyperplane: Hyperplane, data: numpy.ndarray, labels: numpy.ndarray,
                           classes: numpy.ndarray) -> float:
    """Compute Entropy for the given `hyperplane` on the given `data` and `labels`.

    Args:
        hyperplane: The hyperplane whose gini to compute.
        data: The data to compute the gini on.
        labels: The labels to compute the gini on.
        classes: Set of classes.

    Returns:
        The gini for the given hyperplane.
    """
    if labels.size == 0:
        return 0

    partitions = hyperplane(data).astype(int)

    return _entropy(partitions, labels, classes)

def _entropy_on_tree(tree: Tree, data: numpy.ndarray, labels: numpy.ndarray,
               classes: numpy.ndarray) -> float:
    """Compute Entropy for the given `hyperplane` on the given `data` and `labels`.

    Args:
        tree: The tree whose gini to compute.
        data: The data to compute the gini on.
        labels: The labels to compute the gini on.
        classes: Set of classes.

    Returns:
        The gini for the given hyperplane.
    """
    if labels.size == 0:
        return 0

    paths = tree.paths
    partitions = numpy.repeat(numpy.nan, labels.size)

    for path_id, path in enumerate(paths):
        partitions[path(data)] = path_id

    return _entropy(partitions, labels, classes)

def _entropy(partitions: numpy.ndarray, labels: numpy.ndarray, classes: numpy.ndarray) -> float:
    """Compute the entropy for the given `hyperplane` on the given `data` and `labels`.

    Args:
        partitions: The induced partitions.
        labels: The labels to compute the gini on.
        classes: Set of classes.

    Returns:
        The entropy for the given hyperplane.
    """
    unique_partitions, partitions_sizes = numpy.unique(partitions, return_counts=True)
    partitions_information = list()
    for partition, partitions_size in zip(unique_partitions, partitions_sizes):
        partitions_information.append(numpy.array([(labels[partition == partitions] == class_of_interest).sum() / partitions_size
                                                   for class_of_interest in classes]))
    partition_probabilities_per_class = numpy.array(partitions_information)
    unweighted_partition_information = partition_probabilities_per_class * numpy.log(partition_probabilities_per_class)
    unweighted_partition_information = sum(unweighted_partition_information)

    return unweighted_partition_information

##################
### Gain ratio ###
##################
def gain_ratio(partitioner: Hyperplane | InternalNode | Tree, data: numpy.ndarray, labels: numpy.ndarray,
         classes: numpy.ndarray) -> float:
    """Compute Gain Ratio for the given `partitioner` on the given `data` and `labels`.

    Args:
        partitioner: The object partitioning the given data into partitions. If an Hyperplane, a simple Gini
                     split is computed. If an InternalNode, or a Tree, then the induced partitions are given by fully
                     following the Node/Tree, that is, each Path from the Node/Tree to some leaves induces a partition,
                     and the metric is computed on these partitions.
        data: The data to compute the gini on.
        labels: The labels to compute the gini on.
        classes: Set of classes.

    Returns:
        The gain ratio for the given hyperplane.
    """
    return information_gain(partitioner, data, labels, classes) / entropy(partitioner, data, labels, classes)


############
### DKM ###
###########
def dkm(partitioner: Hyperplane | InternalNode | Tree, data: numpy.ndarray, labels: numpy.ndarray,
         classes: numpy.ndarray) -> float:
    """Compute DKM criterion for the given `partitioner` on the given `data` and `labels`.


    Args:
        partitioner: The object partitioning the given data into partitions. If an Hyperplane, a simple Gini
                     split is computed. If an InternalNode, or a Tree, then the induced partitions are given by fully
                     following the Node/Tree, that is, each Path from the Node/Tree to some leaves induces a partition,
                     and the metric is computed on these partitions.
        data: The data to compute the gini on.
        labels: The labels to compute the gini on.
        classes: Set of classes.

    Returns:
        The gini for the given hyperplane.
    """
    if isinstance(partitioner, Hyperplane):
        return _dkm_hyperplane(partitioner, data, labels, classes)
    if isinstance(partitioner, InternalNode):
        return _dkm_hyperplane(partitioner.hyperplane, data, labels, classes)
    elif isinstance(partitioner, (InternalNode, Tree)):
        return _dkm_tree(partitioner, data, labels, classes)


def _dkm_hyperplane(hyperplane: Hyperplane, data: numpy.ndarray, labels: numpy.ndarray,
         classes: numpy.ndarray) -> float:
    """Compute DKM criterion for the given `hyperplane` on the given `data` and `labels`.

    Args:
        hyperplane: The hyperplane whose gini to compute.
        data: The data to compute the gini on.
        labels: The labels to compute the gini on.
        classes: Set of classes.

    Returns:
        The gini for the given hyperplane.
    """
    if labels.size == 0:
        return 0

    partitions = hyperplane(data).astype(int)

    return _dkm(partitions, labels, classes)

def _dkm_tree(tree: Tree, data: numpy.ndarray, labels: numpy.ndarray,
               classes: numpy.ndarray) -> float:
    """Compute DMK criterion for the given `hyperplane` on the given `data` and `labels`.

    Args:
        tree: The tree whose gini to compute.
        data: The data to compute the gini on.
        labels: The labels to compute the gini on.
        classes: Set of classes.

    Returns:
        The gini for the given hyperplane.
    """
    if labels.size == 0:
        return 0

    paths = tree.paths
    partitions = numpy.repeat(numpy.nan, labels.size)

    for path_id, path in enumerate(paths):
        partitions[path(data)] = path_id

    assert partitions[numpy.isnan(partitions)].size == 0

    return _dkm(partitions, labels, classes)

def _dkm(partitions: numpy.ndarray, labels: numpy.ndarray, classes: numpy.ndarray) -> float:
    """Compute DKM criterion on the given `partitions`.

    Args:
        partitions: Indices of induced partitions. `partitions[i]` holds the partition number induced on `data[i]`.
        labels: The labels to compute the gini on.
        classes: Set of classes.

    Returns:
        The gini for the given partitions.
    """
    if labels.size == 0:
        return 0

    unique_partitions, partitions_sizes = numpy.unique(partitions, return_counts=True)
    partition_dkms = list()
    for partition, partitions_size in zip(unique_partitions, partitions_sizes):
        partition_dkms.append(numpy.array([(labels[partition == partitions] == class_of_interest).sum() / partitions_size
                                for class_of_interest in classes]) ** 2)
    partition_impurities_per_class = numpy.array(partition_dkms)
    computed_dkm = 2 * numpy.sqrt(partition_impurities_per_class.prod())

    return computed_dkm


#############
### Mixed ###
#############

def accuracy(hyperplane: Hyperplane | InternalNode, data: numpy.ndarray, labels: numpy.ndarray) -> float:
    """Compute the negated accuracy delta between the split (through `hyperplane`) and not split `data`.

    Args:
       hyperplane: The hyperplane whose gini to compute.
       data: The data to compute the gini on.
       labels: The labels to compute the gini on.

    Returns:
        The label deviation for the given `hyperplane`.
    """
    return 1 - _binary_metric_split(hyperplane, data, labels, "accuracy")


def f1(hyperplane: Hyperplane | InternalNode, data: numpy.ndarray, labels: numpy.ndarray) -> float:
    """Compute the negated f1 delta between the split (through `hyperplane`) and not split `data`.

    Args:
       hyperplane: The hyperplane whose gini to compute.
       data: The data to compute the gini on.
       labels: The labels to compute the gini on.

    Returns:
        The f1 for the given `hyperplane`.
    """
    return 1 - _binary_metric_split(hyperplane, data, labels, "f1")


def auc(hyperplane: Hyperplane | InternalNode, data: numpy.ndarray, labels: numpy.ndarray) -> float:
    """Compute the AUC delta between the split (through `hyperplane`) and not split `data`.

    Args:
       hyperplane: The hyperplane whose gini to compute.
       data: The data to compute the gini on.
       labels: The labels to compute the gini on.

    Returns:
        The AUC for the given `hyperplane`.
    """
    return 1 - _binary_metric_split(hyperplane, data, labels, "auc")


def _binary_metric_split(hyperplane: Hyperplane | InternalNode, data: numpy.ndarray, labels: numpy.ndarray,
                         metric: str) -> float:
    """Compute the accuracy delta between the split (through `hyperplane`) and not split `data`.

    Args:
       hyperplane: The hyperplane whose gini to compute.
       data: The data to compute the gini on.
       labels: The labels to compute the gini on.
       metric: Metric to use.

    Returns:
        The label deviation for the given `hyperplane`.
    """
    if labels.size == 0:
        return 0

    covered_indices = hyperplane(data) if isinstance(hyperplane, Hyperplane) else hyperplane.hyperplane(data)
    noncovered_indices = ~covered_indices
    covered_indices_class = round(labels[covered_indices].mean()) if sum(covered_indices) > 0 else 0
    noncovered_indices_class = round(labels[noncovered_indices].mean()) if sum(noncovered_indices) > 0 else 0
    split_labels = numpy.full(labels.size, numpy.nan)
    split_labels[covered_indices] = covered_indices_class
    split_labels[noncovered_indices] = noncovered_indices_class

    match metric:
        case "accuracy":
            score_function = accuracy_score
        case "f1":
            score_function = f1_score
        case "auc":
            score_function = roc_auc_score
        case _:
            raise ValueError(f"Unknown metric: {metric}")

    score = score_function(labels, split_labels)

    return score


def label_deviation(hyperplane: Hyperplane, data: numpy.ndarray, labels: numpy.ndarray) -> float:
    """Compute the delta in standard deviation of the labels between the split (through `hyperplane`) and
    not split `data`.

     Args:
        hyperplane: The hyperplane whose gini to compute.
        data: The data to compute the gini on.
        labels: The labels to compute the gini on.

    Returns:
        The label deviation for the given `hyperplane`.
    """
    std_labels = labels.std()
    node_size = data.shape[0]
    left_idx = hyperplane(data)
    try:
        left_child_labels, right_child_labels = labels[left_idx], labels[~left_idx]
    except IndexError as e:
        raise e
    weighted_children_errors = [left_child_labels.std() * left_child_labels.size,
                                right_child_labels.std() * right_child_labels.size]
    # correct for no labels
    weighted_children_errors[0] = weighted_children_errors[0] if left_child_labels.size > 0 else 0
    weighted_children_errors[1] = weighted_children_errors[1] if right_child_labels.size > 0 else 0

    return std_labels - (1 / node_size) * sum(weighted_children_errors)
