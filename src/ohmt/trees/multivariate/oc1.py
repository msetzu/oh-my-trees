from __future__ import annotations

import concurrent.futures
import logging
import random
import copy
from typing import Union, Tuple, Optional, Self, Dict, Callable

import numpy
from sklearn.tree import DecisionTreeClassifier


from ohmt.planes import OHyperplane
from ohmt.systems.utils import from_cart
from ohmt.trees.splits.evaluation import gini
from ohmt.trees.structure.trees import ObliqueTree, InternalNode, Node, Leaf


class OC1(ObliqueTree):
    """OC1 Decision Tree (https://dl.acm.org/doi/10.5555/1622826.1622827)"""
    def __init__(self, root: Optional[InternalNode] = None, store_data: bool = True):
        super(OC1, self).__init__(root, store_data)
        self.shift = 0

    def fit(self, data: numpy.ndarray, labels: numpy.ndarray,
            max_depth: int = numpy.inf, min_eps: float = 0.00000001, min_samples: int = 10,
            node_hyperparameters: Optional[Dict] = None,
            **step_hyperparameters) -> Self:
        """Learn a OC1 Decision Tree

        Args:
            data: The training set
            labels: The training set labels
            max_depth: Maximum depth of the Decision Tree
            min_eps: Minimum improve in the learning metric to keep on creating new nodes
            min_samples: Minimum number of instances required to induce a new internal node
            node_hyperparameters: Hyperparameters passed to the node construction:
                step_hyperparameters: keyword arguments:
                    seeding_policy: Policy to select an initial seed:
                        - 'cart' returns the hyperplane given by an axis-parallel CART tree of depth 1.
                        - 'random' returns a random hyperplane
                        - 'random_forest' selects the hyperplane with best gini in a random forest of depth=1 trees
                        Defaults to 'cart'
                    coefficient_policy: Coefficient optimization strategy:
                        - 'sequential': apply 'single_best' incrementally on every feature, stop when single best
                                        yields no more improvements
                        - 'single_best': perturbs only a (random) coefficient
                        - 'best': perform 'single_best' sequentially on all coefficients until no coefficient changes
                        - 'repeat': repeat 'single_best' `perturbations` times
                        Defaults to 'single_best'
                    search_policy: Perturbation strategy. Defaults to 'multi-search'
                    restarts: Number of restarts for the 'multi-search' search policy
                    perturbations: Number of repeats for the 'repeat' coefficient policy, if selected. Defaults to 50
                    max_iter: Maximum iterations for the coefficient optimization policy, if . Defaults to 1000
                    nr_threads: Number of threads. Defaults to 1

        Returns:
            This OC1DT, fit to the given `data` and `labels`.
        """
        # create root
        logging.debug("Fitting tree with:")
        logging.debug(f"\tmax depth: {max_depth}")
        logging.debug(f"\tmin eps: {min_eps}")
        logging.debug(f"\tseeding_policy: {step_hyperparameters.get('seeding_policy', 'cart')}")
        logging.debug(f"\tcoefficient_policy: {step_hyperparameters.get('coefficient_policy', 'single_best')}")
        logging.debug(f"\tsearch_policy: {step_hyperparameters.get('search_policy', 'random')}")
        logging.debug(f"\trestarts: {step_hyperparameters.get('restarts', 100)}")

        tr_data = copy.deepcopy(data)
        # noinspection PyArgumentList
        self.shift = tr_data.min() * (1 if tr_data.min() >= 0 else -1) + 1.
        tr_data = tr_data + self.shift

        super().fit(data=tr_data, labels=labels,
                    max_depth=max_depth, min_eps=min_eps, min_samples=min_samples,
                    node_hyperparameters=node_hyperparameters, **step_hyperparameters)

        return self

    def step(self, parent_node: Optional[InternalNode],
             data: numpy.ndarray, labels: numpy.ndarray, classes: numpy.ndarray,
             direction: Optional[str] = None, depth: int = 1,
             min_eps: float = 0.000001, max_depth: int = 16, min_samples: int = 10,
             node_fitness_function: Callable = gini,
             node_hyperparameters: Optional[Dict] = None, **step_hyperparameters) -> Optional[Node]:
        """Perform a learning step: given the `root` node, learn its two children.

        Args:
            data: The training set
            labels: The training set labels
            depth: Current depth of the tree
            classes: Set of labels
            direction: Direction towards which the step is taken. Either "left" or "right".
            max_depth: Maximum depth allowed for the tree
            parent_node: The subtree to fit
            min_eps: Minimum improve in the learning metric to keep on creating new nodes
            min_samples: Minimum number of samples in leaves
            node_fitness_function: Function to evaluate the goodness of a split. Defaults to gini impurity
            node_hyperparameters: Hyperparameters to fit the node
            step_hyperparameters: Set of hyperparameters:
                seeding_policy: Policy to select an initial seed:
                                - 'cart' returns the hyperplane given by an axis-parallel CART tree of depth 1.
                                - 'random' returns a random hyperplane
                                - 'random_forest' selects the hyperplane with best gini in a random forest of depth=1 trees
                                Defaults to 'cart'
                coefficient_policy: Coefficient optimization strategy:
                            - 'sequential': apply 'single_best' incrementally on every feature, stop when single best
                                            yields no more improvements
                            - 'single_best': perturbs only a (random) coefficient
                            - 'best': perform 'single_best' sequentially on all coefficients until no coefficient changes
                            - 'repeat': repeat 'single_best' opt['perturbations'] times
                            Defaults to 'single_best'.
                search_policy: Perturbation strategy. Defaults to 'multi-search'
                restarts: Number of restarts for the 'multi-search' search policy
                perturbations: Number of repeats for the 'repeat' coefficient policy, if selected. Defaults to 50
                max_iter: Maximum iterations for the coefficient optimization policy, if . Defaults to 1000
                nr_threads: Number of threads. Defaults to 1

        Returns:
            This node's children
        """
        seeding_policy = step_hyperparameters.get("seeding_policy", "cart")
        coefficient_policy = step_hyperparameters.get("coefficient_policy", "single_best")
        search_policy = step_hyperparameters.get("search_policy", "random")
        restarts = step_hyperparameters.get("restarts", 100)
        max_iter = step_hyperparameters.get("max_iter", 1000)
        perturbations = step_hyperparameters.get("perturbations", 50)
        nr_threads = step_hyperparameters.get("nr_threads", 1)

        # stopping criteria
        should_stop_for_depth = self.should_stop_for_max_depth(depth, max_depth, parent_node, labels, data)
        should_stop_for_min_samples = self.should_stop_for_min_samples(data, labels, direction, parent_node, min_samples)

        if not should_stop_for_depth and not should_stop_for_min_samples:
            # learn split
            logging.debug("\tCreating CART hyperplane...")
            cart_hyperplane = self.seed("cart", data=data, labels=labels, classes=classes)
            if cart_hyperplane is None:
                logging.debug("\t\t\tNo CART hyperplane found on {data.shape[0]} records")
                probabilities = numpy.bincount(labels) / labels.size
                fit_node = Leaf(probabilities)
            else:
                cart_hyperplane_fitness = node_fitness_function(cart_hyperplane, data=data, labels=labels, classes=classes)
                logging.debug("\tCreating seed hyperplane...")
                seed = self.seed(seeding_policy, data=data, labels=labels, classes=classes)

                # optimization
                logging.debug(f"\tCoefficient optimization with coefficient policy: {coefficient_policy}...")
                coefficient_optimized_hyperplane = self.optimize_coefficients(seed, coefficient_policy, data, labels,
                                                                              classes, min_eps=min_eps, perturbations=perturbations,
                                                                              max_iter=max_iter, nr_threads=nr_threads)
                logging.debug(f"\tHyperplane optimization with search policy: {search_policy}...")
                multisearch_optimized_hyperplane = self.perturb(coefficient_optimized_hyperplane, data, labels, classes,
                                                                directional_policy='multi-search', search_policy=search_policy,
                                                                restarts=restarts, max_iter=max_iter, min_eps=min_eps,
                                                                nr_threads=nr_threads)
                multisearch_optimized_hyperplane_fitness = node_fitness_function(multisearch_optimized_hyperplane,
                                                                                 data=data, labels=labels, classes=classes)
                # hyperplane choice
                if cart_hyperplane_fitness <= multisearch_optimized_hyperplane_fitness:
                    fit_node = InternalNode(cart_hyperplane)
                else:
                    fit_node = InternalNode(multisearch_optimized_hyperplane)

                if parent_node is not None:
                    if self.should_stop_for_min_eps(parent_node, fit_node,
                                                    node_fitness_function=node_fitness_function,
                                                    validation_data=data, validation_labels=labels, classes=classes,
                                                    min_eps=min_eps):
                        fit_node = self.build_leaf(labels)

            self.store(fit_node, data, labels)

            return fit_node
        else:
            return None

    def seed(self, sampling_policy: str = "cart", candidates_size: int = 1000,
             data: Optional[numpy.ndarray] = None, labels: Optional[numpy.ndarray] = None,
             classes: Optional[numpy.ndarray] = None) -> Optional[OHyperplane]:
        """Generate an initial hyperplane to perturb according to a `sampling_policy`.

        Args:
            data: Data to select the best seed
            labels: Labels for `data` to select the best seed
            classes: Dataset classes to compute gini
            sampling_policy: The sampling policy to choose the hyperplane. Available values are:
                - 'cart' returns the hyperplane given by an axis-parallel CART tree of depth 1.
                - 'random' returns a random hyperplane
                - 'random_forest' selects the hyperplane with best gini in a random forest of depth=1 trees
            candidates_size: number of candidate trees to seed from when `sampling_policy == 'random_forest'`.
                            Defaults to 1000

        Returns:
            A seed hyperplane to perturb, if one is found, None otherwise.
        """
        if sampling_policy not in ["cart", "random", "random_forest"]:
            raise ValueError("Use one of {'cart', 'random', 'random_forest'} sampling policies")
        # cart
        elif sampling_policy == 'cart':
            if data is None or labels is None:
                raise ValueError("Need to provide parameter 'data' and 'labels'  when using the 'cart' sampling policy")
            cart = DecisionTreeClassifier(max_depth=1)
            cart.fit(data, labels)
            extracted_rules = from_cart(cart)
            # # flatten out the extracted systems since a CART tree may have multiple.
            # # Here take only the first since it has depth 1
            extracted_rules = [OHyperplane.from_aphyperplane(hyperplane[0], dimensionality=data.shape[1])
                               for hyperplane, _ in extracted_rules]
        # random
        elif sampling_policy == "random":
            coefficients_and_bound = numpy.random.rand(data.shape[1] + 1,)
            extracted_rules = OHyperplane(coefficients_and_bound[:-1], coefficients_and_bound[-1])
        # random forest
        else:
            extracted_rules = [self.seed("cart", data=data, labels=labels) for _ in range(candidates_size)]

        if len(extracted_rules) == 1:
            seed_rule = extracted_rules[0][0]
        elif len(extracted_rules) > 1:
            rules_gini = [gini(rule, data, labels, classes) for rule in extracted_rules]
            seed_rule = extracted_rules[numpy.argmax(rules_gini)]
        else:
            return None

        return seed_rule

    def perturb(self, hyperplane: OHyperplane, data: numpy.ndarray, labels: numpy.ndarray,
                classes: numpy.ndarray, directional_policy: str = "random", search_policy: str = "single_best",
                restarts: int = 100, perturbations: int = 50, max_iter: int = 1000, min_eps: float = 0.000001,
                nr_threads: int = 1) -> OHyperplane:
        """Perturbation along a direction chosen according to the given `directional_policy`, either a random one, or a one.

        Args:
            hyperplane: The hyperplane to perturb
            data: Data to compute gini.
            labels: Labels of `data` to compute gini.
            classes: Dataset classes to compute gini.
            directional_policy: The directional policy. 'random' to choose a random direction and optimize its weight,
                                'multi-search' to perform a set of directional and optimized perturbations. `restarts`
                                determines the size of this set.
            search_policy: The search policy for random perturbation:
                        - 'sequential': apply 'single_best' incrementally on every feature, stop when single best
                                        yields no more improvements;
                                        yields no more improvements;
                        - 'single_best': perturbs only a (random) coefficient;
                        - 'best': perform 'single_best' sequentially on all coefficients until no coefficient changes;
                        - 'repeat': repeat 'single_best' opt['perturbations'] times.
                        Defaults to 'single_best'.
            restarts: Number of restarts for the 'multisearch' search policy.
            perturbations: Number of repeats for the 'repeat' coefficient policy, if selected. Defaults to 50
            max_iter: Maximum iterations for the coefficient optimization policy, if . Defaults to 1000
            min_eps: minimum improvement to keep on iterating. Defaults to 0.000001
            nr_threads: Number of threads. Defaults to 1

        Returns:
            A perturbed hyperplane with lower gini. Returns a copy of the same hyperplane if no satisfying perturbation
            is found.
        """
        if directional_policy == "random":
            direction = OHyperplane(numpy.random.rand(len(hyperplane),), 0)
            alpha, optimized_gini = self.optimize_perturbation_weight(hyperplane, direction, data, labels, classes)

            return hyperplane + direction * alpha
        elif directional_policy == "multi-search":
            logging.debug(f"\t\tMulti-search perturbation on {restarts} restarts")
            best_gini = gini(hyperplane, data, labels, classes)
            best_perturbation = copy.deepcopy(hyperplane)

            for _ in range(restarts):
                # cycle on random perturbation + optimization
                while True:
                    # random perturbation...
                    random_perturbation = self.perturb(hyperplane, data, labels, classes, directional_policy="random")
                    # optimization
                    optimized_random_baseline = self.optimize_coefficients(random_perturbation, search_policy,
                                                                           data, labels, classes, nr_threads=nr_threads,
                                                                           perturbations=perturbations, max_iter=max_iter,
                                                                           min_eps=min_eps)
                    optimized_random_baseline_gini = gini(optimized_random_baseline,
                                                                      data, labels, classes)
                    if optimized_random_baseline_gini > best_gini:
                        best_gini = optimized_random_baseline_gini
                        best_perturbation = optimized_random_baseline
                    else:
                        break
            return best_perturbation

    def optimize_coefficients(self, hyperplane: OHyperplane, optimization_policy: Union[numpy.ndarray, str],
                              data: numpy.ndarray, labels: numpy.ndarray, classes: numpy.ndarray,
                              perturbations: Optional[int] = None, max_iter: int = 1000, min_eps: float = 0.000001,
                              nr_threads: int = 1) -> OHyperplane:
        """Perturbs the given `hyperplane` along the given `direction`.

        Args:
            hyperplane: The hyperplane to perturb
            optimization_policy: Optimization direction: one of:
                - 'sequential': apply 'single_best' incrementally on every feature, stop when single best
                                yields no more improvements
                - 'single_best': perturbs only a (random) coefficient
                - 'best': perform 'single_best' sequentially on all coefficients until no coefficient changes
                - 'repeat': repeat 'single_best' `perturbations` times
                Defaults to 'single_best'
            data: Data to select the best coefficient
            labels: Labels of `data` to select the best coefficient
            classes: Dataset classes used to compute gini
            perturbations: Number of repeats for the 'repeat' optimization policy, if selected. Defaults to 50
            max_iter: maximum number of iterations. Defaults to 1000
            min_eps: minimum improvement to keep on iterating. Defaults to 0.000001
            nr_threads: Number of threads. Defaults to 1

        Returns:
            The perturbed hyperplane
        """
        def single_best_coefficient(hyperplane_to_perturb: OHyperplane, coefficient: Optional[int] = None, max_workers: int = 1):
            """Find the single-coefficient best perturbation"""
            dimensionality = len(hyperplane)
            if coefficient is None:
                perturbations = [None] * dimensionality
                if max_workers > 1:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        for j in range(dimensionality):
                            future_res = executor.submit(self.optimize_single_coefficient,
                                                         j, hyperplane, data, labels, classes)
                            perturbations[j] = future_res.result()
                else:
                    for j in range(dimensionality):
                        perturbations[j] = self.optimize_single_coefficient(j, hyperplane, data, labels, classes)

                best_coefficient = numpy.argmin([perturbation_gini for _, perturbation_gini in perturbations])
                hyperplane_to_perturb[best_coefficient] = perturbations[best_coefficient][0]
                perturbation_gini = perturbations[best_coefficient][1]

                return hyperplane_to_perturb, best_coefficient, perturbation_gini
            else:
                hyperplane_to_perturb, perturbation_gini = self.optimize_single_coefficient(coefficient, hyperplane, data, labels, classes)

                return hyperplane_to_perturb, coefficient, perturbation_gini

        perturbed_hyperplane = copy.deepcopy(hyperplane)
        # single best: perturb only one coefficient, try many coefficients
        if isinstance(optimization_policy, str) and optimization_policy == "single_best":
            # logging.debug('Single best coefficient optimization...')
            perturbed_hyperplane, coefficient, _ = single_best_coefficient(perturbed_hyperplane, max_workers=nr_threads)

        # repeat: repeat single best multiple times
        elif isinstance(optimization_policy, str) and optimization_policy == 'repeat':
            nr_repetitions = perturbations if perturbations is not None else len(perturbed_hyperplane)
            logging.debug(f"\t\tOptimizing single-best coefficient {nr_repetitions} times...")
            coefficients_to_perturb = random.sample(range(len(hyperplane)), nr_repetitions if nr_repetitions < len(hyperplane) else len(hyperplane))

            for coefficient in coefficients_to_perturb:
                perturbed_hyperplane, _, _ = single_best_coefficient(perturbed_hyperplane, coefficient, max_workers=nr_threads)

        # best: repeat single best until there's no improvement
        elif isinstance(optimization_policy, str) and optimization_policy == 'best':
            logging.debug("\t\tOptimizing single-best coefficient at most {max_iter} times, until there's no improvement >= eps={min_eps}...")
            for i in range(max_iter):
                newly_perturbed_hyperplane, _, _ = single_best_coefficient(perturbed_hyperplane, max_workers=nr_threads)
                # stop for minimum improvement
                if (perturbed_hyperplane - newly_perturbed_hyperplane).coefficients.sum() < min_eps:
                    break

        # sequential
        elif isinstance(optimization_policy, str) and optimization_policy == "sequential":
            # logging.debug("\t\tOptimizing coefficients on a sequence until no coefficient yields improvements...")
            at_least_one_perturbed = True
            while at_least_one_perturbed:
                at_least_one_perturbed = False
                newly_perturbed_hyperplane, _, _ = single_best_coefficient(perturbed_hyperplane, max_workers=nr_threads)
                if (perturbed_hyperplane - newly_perturbed_hyperplane).coefficients.sum() < min_eps:
                    at_least_one_perturbed = True

        return perturbed_hyperplane

    def optimize_single_coefficient(self, coefficient: int, hyperplane: OHyperplane, data: numpy.ndarray, labels: numpy.ndarray,
                                    classes: numpy.ndarray) -> Tuple[float, float]:
        """Find the optimum value for the `coefficient`-th coefficient of `hyperplane`.

        Args:
            coefficient: The coefficient to perturb
            hyperplane: The hyperplane to which the coefficient belongs
            data: Data to select the best coefficient
            labels: Labels of `data` to select the best coefficient
            classes: Dataset classes used to compute gini
        Returns:
            The optimum value for the `coefficient`-th coefficient, and the associated gini.
        """
        # logging.debug("Optimizing coefficient {0}...".format(coefficient))
        projections_on_hyperplane = numpy.dot(data, hyperplane.coefficients) - hyperplane.bound  # V = [V_1, ..., V_n]
        projecting_values = data[:, coefficient]  # a_m
        coefficient_only_projections = projecting_values * hyperplane[coefficient]

        candidate_baselines = ((coefficient_only_projections - projections_on_hyperplane) / projecting_values).squeeze()
        candidate_baselines.sort()
        sorted_candidate_baselines = candidate_baselines
        # use quantiles for long candidate lists
        if sorted_candidate_baselines.size > 100:
            quantiles = numpy.quantile(sorted_candidate_baselines, numpy.arange(0, 1, 0.01))
            quantiles_indices = numpy.array([int(numpy.argwhere((sorted_candidate_baselines >= q)).squeeze()[0])
                                            for q in quantiles])
            sorted_candidate_baselines = [sorted_candidate_baselines[int(i)] for i in quantiles_indices]
        else:
            sorted_candidate_baselines = sorted(candidate_baselines)

        candidate_hyperplanes = [copy.deepcopy(hyperplane) for _ in range(len(sorted_candidate_baselines))]
        for candidate_hyperplane, candidate_perturbation in zip(candidate_hyperplanes, sorted_candidate_baselines):
            candidate_hyperplane[coefficient] = candidate_perturbation
        candidates_impurities = [gini(candidate_hyperplane, data, labels, classes)
                                 for candidate_hyperplane in candidate_hyperplanes]
        best_hyperplane = numpy.argmin(candidates_impurities)

        return sorted_candidate_baselines[best_hyperplane], candidates_impurities[best_hyperplane]

    def optimize_perturbation_weight(self, hyperplane: OHyperplane, direction: OHyperplane, data: numpy.ndarray,
                                     labels: numpy.ndarray, classes: numpy.ndarray):
        """
        Find the optimal value for
        Args:
            hyperplane: The hyperplane to perturb
            direction: The perturbation direction
            data: Data to select the best coefficient
            labels: Labels of `data` to select the best coefficient
            classes: Dataset classes used to compute gini

        Returns:
            The optimal perturbation weight and associated gini
        """
        hyperplane_baselines = numpy.dot(data, hyperplane.coefficients)
        perturbation_baselines = numpy.dot(data, direction.coefficients)
        candidate_baselines = - hyperplane_baselines / perturbation_baselines
        sorted_candidate_alphas = numpy.array(sorted(candidate_baselines))

        if sorted_candidate_alphas.size > 100:
            quantiles = numpy.quantile(sorted_candidate_alphas, numpy.arange(0, 1, 0.01))
            quantiles_indices = numpy.array([int(numpy.argwhere((sorted_candidate_alphas >= q)).squeeze()[0])
                                             for q in quantiles])
            sorted_candidate_alphas = [sorted_candidate_alphas[int(i)] for i in quantiles_indices]
        else:
            sorted_candidate_alphas = sorted(candidate_baselines)

        perturbed_hyperplanes = [hyperplane + direction * alpha for alpha in sorted_candidate_alphas]
        candidates_impurities = [gini(candidate, data, labels, classes)
                                 for candidate in perturbed_hyperplanes]
        best_alpha_index = numpy.argmin(candidates_impurities)

        return sorted_candidate_alphas[best_alpha_index], candidates_impurities[best_alpha_index]

    def predict(self, data: numpy.ndarray) -> numpy.array:
        return super().predict(data + self.shift)
