import numpy as np
import pandas as pd
import tskit
from scipy.linalg import eig
from scipy.optimize import minimize, shgo
from glob import glob
import math
from .world_map import WorldMap
from .likelihood import calc_composite_likelihood_for_parameters


def deconstruct_tree(tree, epochs):
    """Converts a `tskit.Tree` into primitive elements that can be passed to numba

    Parameters
    ----------
    tree : tskit.Tree
        Single tree object
    epochs : numpy.ndarray
        Order timing of epoch starts from argile.WorldMap

    Returns
    -------
    parents : numpy.ndarray
        Parent IDs for each node
    branch_above : numpy.ndarray
        Branch above length (split across epochs) for each node
    time_bin_widths : numpy.ndarray
        Time bin widths for each node (All 1)
    ids_asc_time : numpy.ndarray
        Nodes IDs in time ascending order
    """

    num_nodes = len(tree.postorder())
    parents = np.full(num_nodes, -1, dtype="int64")
    branch_above = np.zeros((num_nodes, len(epochs)), dtype="int64")
    node_epoch = np.full(num_nodes, -1, dtype="int64")
    time_bin_widths = np.full(num_nodes, -1, dtype="int64")
    ids_asc_time = np.full(num_nodes, -1, dtype="int64")
    for i,node in enumerate(tree.nodes(order="timeasc")):
        node_time = tree.time(node)
        parent = tree.parent(node)
        starting_epoch = np.digitize(node_time, epochs)-1
        node_epoch[node] = starting_epoch
        if parent != -1:
            parent_time = tree.time(parent)
            ending_epoch = np.digitize(parent_time, epochs)-1
            if starting_epoch == ending_epoch:
                branch_above[node, starting_epoch] = math.ceil(parent_time - node_time)
            else:
                branch_above[node, starting_epoch] = math.ceil(epochs[starting_epoch+1] - node_time)
                for e in range(starting_epoch+1, ending_epoch):
                    branch_above[node, e] = math.ceil(epochs[e+1] - epochs[e])
                branch_above[node, ending_epoch] = math.ceil(parent_time - epochs[ending_epoch])
        ids_asc_time[i] = node
        parents[node] = parent
        time_bin_widths[node] = 1
    return parents, branch_above, node_epoch, time_bin_widths, ids_asc_time

def deconstruct_trees(trees, epochs):
    """Converts list of `tskit.Tree`s into primitive elements that can be passed to numba

    Parameters
    ----------
    trees : list
        tskit.Tree objects
    epochs : numpy.ndarray
        Order timing of epoch starts from argile.WorldMap

    Returns
    -------
    pl : list
        Arrays containing ID of parent for each node, one array per tree
    bal : list
        Arrays containing branch above length for each node, one array per tree
    tbw : list
        Arrays containaing time bin widths for each node, one array per tree
    iat : list
        Arrays of nodes IDs in time ascending order, one array per tree
    unique_branch_lengths : list
        List of lists containing unique branch lengths in each epoch
    """
    
    pl = []
    bal = []
    nel = []
    tbw = []
    iat = []
    all_branch_lengths = [[] for e in epochs]
    for tree in trees:
        parents, branch_above, node_epoch, time_bin_widths, ids_asc_time = deconstruct_tree(tree, epochs)
        pl.append(parents)
        bal.append(branch_above)
        nel.append(node_epoch)
        tbw.append(time_bin_widths)
        iat.append(ids_asc_time)
        for e in range(len(epochs)):
            all_branch_lengths[e].extend(branch_above[:, e])
    unique_branch_lengths = []
    for e in range(len(epochs)):
        unique_branch_lengths.append(np.unique(all_branch_lengths[e]))
    return pl, bal, nel, tbw, iat, unique_branch_lengths

def _run_from_minimize(
        parameters,
        world_map,
        parents,
        branch_above,
        node_epoch,
        ids_asc_time,
        sample_locations_array_log,
        sample_ids,
        output_file,
        verbose
    ):
    """Switches the sign of composite likelihood so that it can be minimized

    Parameters
    ----------
    parameters : numpy.ndarray
        Combination of parameters used to build the migration surface
    world_map : argile.WorldMap
        Custom object built using the `demes.tsv`, `connections.tsv`, and `samples.tsv` files
    parents : list
        Arrays containing ID of parent for each node, one array per tree
    branch_above : list
        Arrays containing branch above length (split across epochs) for each node, one array per tree
    node_epoch : list
        Arrays containing the epochs of each node, one array per tree
    ids_asc_time : list
        Arrays of nodes IDs in time ascending order, one array per tree
    sample_locations_array_log : numpy.ndarray
        Probability distribution vector for each sample location (generally 0 in all demes except one)
    sample_ids : numpy.ndarray
        Order of sample IDs for `sample_locations_array`
    output_file : str
        Path to an output file to write to (default is `None`, ignored)
    verbose : bool
        Whether to print log-likelihoods to the terminal (default is False)

    Returns
    -------
    composite_likelihood : float
        Negative of log-likelihood of the parameter combination

    """

    new_parameters = parameters.copy()
    for p in range(len(world_map.parameters)):
        if world_map.parameters[p] != "alpha":
            new_parameters[p] = np.exp(new_parameters[p])
    
    return -calc_composite_likelihood_for_parameters(
        new_parameters,
        world_map,
        parents,
        branch_above,
        node_epoch,
        ids_asc_time,
        sample_locations_array_log,
        sample_ids,
        output_file,
        verbose
    )

def run_for_parameters(
        parameters,
        demes_path,
        connections_path,
        samples_path,
        trees_dir_path,
        chop_time=None
    ):

    demes = pd.read_csv(demes_path, sep="\t")
    connections = pd.read_csv(connections_path, sep="\t")
    samples = pd.read_csv(samples_path, sep="\t")
    world_map = WorldMap(demes, connections, samples)

    sample_locations_array = world_map.sample_locations
    sample_ids = np.array(world_map.samples["id"])
    sample_locations_array = np.maximum(sample_locations_array, 1e-99)
    sample_locations_array_log = np.log(sample_locations_array)

    if trees_dir_path[-1] != "/":
        trees_dir_path += "/"
    
    trees = []
    for ts in glob(trees_dir_path+"*"):
        tree = tskit.load(ts)
        if chop_time is not None:
            decap = tree.decapitate(chop_time)
            tree = decap.subset(nodes=np.where(decap.tables.nodes.time <= chop_time)[0])
        trees.append(tree.simplify().first())

    parents, branch_above, node_epoch, time_bin_widths, ids_asc_time, unique_branch_lengths = deconstruct_trees(trees=trees, epochs=world_map.epochs)

    return calc_composite_likelihood_for_parameters(
        parameters,
        world_map,
        parents,
        branch_above,
        node_epoch,
        ids_asc_time,
        sample_locations_array_log,
        sample_ids
    )


def run(
        demes_path,
        connections_path,
        samples_path,
        trees_dir_path,
        chop_time=None,
        output_file=None,
        verbose=False
    ):
    """Identifies the most likely combination of world map parameters

    Uses an optimization algorithm to search the parameter space. 

    Parameters
    ----------
    demes_path : str
        Path to the `demes.tsv` file
    connections_path : str
        Path to the `connections.tsv` file
    samples_path : str
        Path to the `samples.tsv` file
    trees_dir_path : str
        Path to the `trees/` folder
    chop_time : 
        Time at which to chop trees (default is `None`, ignored)
    output_file : str
        Path to an output file to write to (default is `None`, ignored)
    verbose : bool
        Whether to print log-likelihoods to the terminal (default is False)

    Returns
    -------
    final : numpy.ndarray
        Combination of parameters
    -res.fun : float
        Log-likelihood of the parameter combination
    """
    
    if output_file is not None:
        with open(output_file, "w") as outfile:
            outfile.write("parameters\tloglikelihood\n")

    demes = pd.read_csv(demes_path, sep="\t")
    connections = pd.read_csv(connections_path, sep="\t")
    samples = pd.read_csv(samples_path, sep="\t")
    world_map = WorldMap(demes, connections, samples)

    sample_locations_array = world_map.sample_locations
    sample_ids = np.array(world_map.samples["id"])
    sample_locations_array = np.maximum(sample_locations_array, 1e-99)
    sample_locations_array_log = np.log(sample_locations_array)

    if trees_dir_path[-1] != "/":
        trees_dir_path += "/"
    
    trees = []
    for ts in glob(trees_dir_path+"*"):
        tree = tskit.load(ts)
        if chop_time is not None:
            decap = tree.decapitate(chop_time)
            tree = decap.subset(nodes=np.where(decap.tables.nodes.time <= chop_time)[0])
        trees.append(tree.simplify().first())

    parents, branch_above, node_epoch, time_bin_widths, ids_asc_time, unique_branch_lengths = deconstruct_trees(trees=trees, epochs=world_map.epochs)

    start = []
    bounds = []
    for p in world_map.parameters:
        if p == "coefficient":
            start.append(np.log(0.0001))
            bounds.append((-10, 0))
        elif p == "alpha":
            start.append(1)
            bounds.append((0, 1))
        else:
            start.append(0)
            bounds.append((-10, 10))
    
    res = minimize(
        fun=_run_from_minimize,
        x0=np.array(start),
        bounds=bounds,
        args=(
            world_map,
            parents,
            branch_above,
            node_epoch,
            ids_asc_time,
            sample_locations_array_log,
            sample_ids,
            output_file,
            verbose
        ),
        method="L-BFGS-B"
    )

    final = res.x.copy()
    for p in range(len(world_map.parameters)):
        if world_map.parameters[p] != "alpha":
            final[p] = np.exp(final[p])

    return final, -res.fun