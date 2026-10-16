import numpy as np
from numpy.linalg import matrix_power
from numba import njit, prange
import scipy
from .migration_matrix import calc_composite_likelihood_for_parameters


@njit()
def _norm1(A):
    """1-norm (max absolute column sum) — numba's np.linalg.norm
    doesn't reliably support ord=1 for 2D arrays, so do it by hand."""
    n = A.shape[1]
    max_sum = 0.0
    for j in range(n):
        s = 0.0
        for i in range(A.shape[0]):
            s += abs(A[i, j])
        if s > max_sum:
            max_sum = s
    return max_sum

@njit()
def expm(A):
    n = A.shape[0]
    I = np.eye(n, dtype=A.dtype)
    # Pade(13) coefficients (Higham, 2005)
    b0, b1, b2, b3, b4, b5, b6, b7 = (
        64764752532480000.0, 32382376266240000.0, 7771770303897600.0,
        1187353796428800.0, 129060195264000.0, 10559470521600.0,
        670442572800.0, 33522128640.0,
    )
    b8, b9, b10, b11, b12, b13 = (
        1323241920.0, 40840800.0, 960960.0, 16380.0, 182.0, 1.0,
    )
    theta13 = 5.371920351148152
    normA = _norm1(A)
    s = 0
    if normA > theta13:
        s = int(np.ceil(np.log2(normA / theta13)))
        if s < 0:
            s = 0
        A = A / (2.0 ** s)
    A2 = A @ A
    A4 = A2 @ A2
    A6 = A2 @ A4
    U = A @ (A6 @ (b13 * A6 + b11 * A4 + b9 * A2)
             + b7 * A6 + b5 * A4 + b3 * A2 + b1 * I)
    V = (A6 @ (b12 * A6 + b10 * A4 + b8 * A2)
         + b6 * A6 + b4 * A4 + b2 * A2 + b0 * I)
    P = U + V
    Q = V - U
    R = np.ascontiguousarray(np.linalg.solve(Q, P))
    for _ in range(s):
        R = R @ R
    return R

@njit()
def calc_current_pos(id, messages, parents, coal_rate):
    """Calculates current node position as product of child messages

    Parameters
    ----------
    id : int
        ID of node
    messages : np.array
        Messages being passed in tree
    parents : np.array
        Parent IDs for each node
    
    Returns
    -------
    current_pos : np.array
        Probability distribution of node's current position given subtree below
    """

    children = np.where(parents==id)[0]
    if len(children) == 0:
        raise RuntimeError(f"Node {id} does not have any children. This should never happen.")
    current_pos = messages[children[0]]
    if len(children) > 1:
        for i in range(1, len(children)):
            current_pos = np.multiply(current_pos, messages[children[i]])
        current_pos = np.multiply(coal_rate, current_pos)
    return current_pos

@njit()
def calc_branch_message(
        current_pos,
        branch_above,
        transition_matrices
    ):
    """Calculates the message to be passed along a branch above specified node

    Parameters
    ----------
    id : int
        ID of node
    current_pos : np.array
        Probability distribution of node's current position given subtree below
    branch_above : np.array
        Branch lengths above each node split across epochs. Shape is #epochs x #nodes.
    direction : string


    Returns
    -------
    current_pos : np.array
        Probability distribution for location of lineage given subtree below. Length is #demes.
    """

    included_epochs = np.where(branch_above > 0)[0]
    for epoch in included_epochs:
        trans_prob = expm(transition_matrices[epoch]*branch_above[epoch])
        current_pos = np.dot(trans_prob, current_pos)
    return current_pos

@njit()
def calc_backward_messages(
        parents,
        branch_above,
        node_epoch,
        ids_asc_time,
        sample_locations_array,
        sample_ids,
        backward_transition_matrices,
        coal_rates
    ):
    """"""

    num_demes = len(sample_locations_array[0])
    messages = np.ones((len(parents), num_demes), dtype="float64")
    loglikelihood = 0
    for id in ids_asc_time: 
        if id in sample_ids:
            current_pos = sample_locations_array[np.where(sample_ids==id)[0][0]]
        else:
            current_pos = calc_current_pos(
                id,
                messages,
                parents,
                coal_rates[node_epoch[id]]
            )

        # Also rescale so that underflow is not a problem and track scaler
        s = np.sum(current_pos)
        current_pos /= s
        loglikelihood += np.log(s)
        
        parent = parents[id]
        if parent != -1:
            messages[id] = calc_branch_message(
                current_pos,
                branch_above[id],
                backward_transition_matrices
            )
        else:   # collect roots here
            messages[id] = current_pos
    
    return loglikelihood, messages

@njit()
def calc_branch_message_pre(
        current_pos,
        branch_above,
        exponentiated_transition_matrices
    ):
    """Calculates the message to be passed along a branch above specified node

    Parameters
    ----------
    id : int
        ID of node
    current_pos : np.array
        Probability distribution of node's current position given subtree below
    branch_above : np.array
        Branch lengths above each node split across epochs. Shape is #epochs x #nodes.
    direction : string


    Returns
    -------
    current_pos : np.array
        Probability distribution for location of lineage given subtree below. Length is #demes.
    """

    included_epochs = np.where(branch_above > 0)[0]
    for epoch in included_epochs:
        trans_prob = matrix_power(exponentiated_transition_matrices[epoch], branch_above[epoch])
        current_pos = np.dot(trans_prob, current_pos)
    return current_pos

@njit()
def calc_backward_messages_pre(
        parents,
        branch_above,
        node_epoch,
        ids_asc_time,
        sample_locations_array,
        sample_ids,
        exponentiated_backward_transition_matrices,
        coal_rates
    ):
    """"""

    num_demes = len(sample_locations_array[0])
    messages = np.ones((len(parents), num_demes), dtype="float64")
    loglikelihood = 0
    for id in ids_asc_time: 
        if id in sample_ids:
            current_pos = sample_locations_array[np.where(sample_ids==id)[0][0]]
        else:
            current_pos = calc_current_pos(
                id,
                messages,
                parents,
                coal_rates[node_epoch[id]]
            )

        # Also rescale so that underflow is not a problem and track scaler
        s = np.sum(current_pos)
        current_pos /= s
        loglikelihood += np.log(s)
        
        parent = parents[id]
        if parent != -1:
            messages[id] = calc_branch_message_pre(
                current_pos,
                branch_above[id],
                exponentiated_backward_transition_matrices
            )
        else:   # collect roots here
            messages[id] = current_pos
    
    return loglikelihood, messages


@njit()
def _calc_branch_message_pre_log(
        current_pos,
        branch_above,
        exponentiated_transition_matrices,
        epoch_durations,
        precalculated_transitions_log,
    ):
    """Calculates the message to be passed along a branch above specified node

    Parameters
    ----------
    id : int
        ID of node
    current_pos : np.array
        Probability distribution of node's current position given subtree below
    unique_branch_lengths : list
        Arrays containing unique branch lengths in each epoch
    precalculated_transitions : list
        Arrays of transition probabilities corresponding with the unique_branch_lengths, converted to log space
    
    Returns
    -------
    message : np.array
        Probability distribution for location of lineage given subtree below. Length is #demes.
    """

    included_epochs = np.where(branch_above > 0)[0]
    for epoch in included_epochs:
        if branch_above[epoch] == epoch_durations[epoch]:
            trans_prob_log = precalculated_transitions_log[epoch]
        else:
            trans_prob_log = np.log(np.maximum(matrix_power(exponentiated_transition_matrices[epoch], branch_above[epoch]), 1e-99))
        current_pos = logsumexp_custom(trans_prob_log + current_pos, axis=1)[np.newaxis, :]
    return current_pos

@njit()
def likelihood_of_tree_pre_log(
        parents,
        branch_above,
        node_epoch,
        ids_asc_time,
        sample_locations_array_log,
        sample_ids,
        exponentiated_backward_transition_matrices,
        epoch_durations,
        precalculated_transitions_log,
        coal_rates_log
    ):
    """Computes log-likelihood of tree in log-space to avoid underflow

    Parameters
    ----------
    parents : np.array
        Parent IDs for each node. Length is #nodes.
    branch_above : np.array
        Branch lengths above each node split across epochs. Shape is #epochs x #nodes.
    node_epoch : np.array
        Epochs of each node. Length is #nodes.
    ids_asc_time : np.array
        IDs of nodes in tree in time ascending order. Length is #nodes.
    sample_locations_array_log : np.array
        Vector representation of sample locations. Shape is #samples x #demes.
    sample_ids : np.array
        IDs of sample nodes in tree. Length is #samples.
    backward_transition_matrices : np.array

    coal_rates_log : numpy.ndarray
        Recipricol of suitability values for each deme in each epoch, converted to log space. Shape is #epochs x #demes.

    Returns
    -------
    loglikelihood : float
        Log-likelihood of the tree
    """

    num_demes = len(sample_locations_array_log[0])
    messages = np.zeros((len(parents), num_demes), dtype="float64")
    loglikelihood = 0
    for id in ids_asc_time:
        if id in sample_ids:
            current_pos = sample_locations_array_log[np.where(sample_ids==id)[0][0]][np.newaxis, :]
        else:
            current_pos = _calc_current_pos_log(
                id,
                messages,
                parents,
                coal_rates_log[node_epoch[id]]
            )
        messages[id] = _calc_branch_message_pre_log(
            current_pos,
            branch_above[id],
            exponentiated_backward_transition_matrices,
            epoch_durations,
            precalculated_transitions_log
        )
        if parents[id] == -1:
            loglikelihood += logsumexp_custom(current_pos, axis=1)[0]
    return loglikelihood

@njit()
def _calc_current_pos_log(id, messages, parents, coal_rates):
    """Calculates current node position as product of child messages

    Parameters
    ----------
    id : int
        ID of node
    messages : np.array
        Messages being passed in tree. Shape is #nodes x #demes.
    parents : np.array
        Parent IDs for each node. Length is #nodes.
    coal_rates : numpy.ndarray
        Recipricol of suitability values for each deme in the node's corresponding epoch, converted to log space
    
    Returns
    -------
    current_pos : np.array
        Probability distribution of node's current position given subtree below. Length is #demes.
    """

    current_pos = coal_rates + np.sum(messages[np.where(parents==id)[0]], axis=0)[np.newaxis, :]
    return current_pos

@njit()
def logsumexp_custom(x, axis):
    """LogSumExp function that can be accessed from numba

    Parameters
    ----------
    x : numpy.ndarray
        Vector to transform
    axis : int
        Which axis of the numpy.array to transform over
    """

    c = np.max(x)
    return c + np.log(np.sum(np.exp(x - c), axis=axis))

@njit()
def _calc_branch_message_log(
        current_pos,
        branch_above,
        transition_matrices
    ):
    """Calculates the message to be passed along a branch above specified node

    Parameters
    ----------
    id : int
        ID of node
    current_pos : np.array
        Probability distribution of node's current position given subtree below
    unique_branch_lengths : list
        Arrays containing unique branch lengths in each epoch
    precalculated_transitions : list
        Arrays of transition probabilities corresponding with the unique_branch_lengths, converted to log space
    
    Returns
    -------
    message : np.array
        Probability distribution for location of lineage given subtree below. Length is #demes.
    """

    included_epochs = np.where(branch_above > 0)[0]
    for epoch in included_epochs:
        trans_prob_log = np.log(np.maximum(expm(transition_matrices[epoch]*branch_above[epoch]), 1e-99))
        current_pos = logsumexp_custom(trans_prob_log + current_pos, axis=1)[np.newaxis, :]
    return current_pos

@njit()
def likelihood_of_tree_log(
        parents,
        branch_above,
        node_epoch,
        ids_asc_time,
        sample_locations_array_log,
        sample_ids,
        backward_transition_matrices,
        coal_rates_log
    ):
    """Computes log-likelihood of tree in log-space to avoid underflow

    Parameters
    ----------
    parents : np.array
        Parent IDs for each node. Length is #nodes.
    branch_above : np.array
        Branch lengths above each node split across epochs. Shape is #epochs x #nodes.
    node_epoch : np.array
        Epochs of each node. Length is #nodes.
    ids_asc_time : np.array
        IDs of nodes in tree in time ascending order. Length is #nodes.
    sample_locations_array_log : np.array
        Vector representation of sample locations. Shape is #samples x #demes.
    sample_ids : np.array
        IDs of sample nodes in tree. Length is #samples.
    backward_transition_matrices : np.array

    coal_rates_log : numpy.ndarray
        Recipricol of suitability values for each deme in each epoch, converted to log space. Shape is #epochs x #demes.

    Returns
    -------
    loglikelihood : float
        Log-likelihood of the tree
    """

    num_demes = len(sample_locations_array_log[0])
    messages = np.zeros((len(parents), num_demes), dtype="float64")
    loglikelihood = 0
    for id in ids_asc_time:
        if id in sample_ids:
            current_pos = sample_locations_array_log[np.where(sample_ids==id)[0][0]][np.newaxis, :]
        else:
            current_pos = _calc_current_pos_log(
                id,
                messages,
                parents,
                coal_rates_log[node_epoch[id]]
            )
        messages[id] = _calc_branch_message_log(
            current_pos,
            branch_above[id],
            backward_transition_matrices
        )
        if parents[id] == -1:
            loglikelihood += logsumexp_custom(current_pos, axis=1)[0]
    return loglikelihood

@njit(parallel=True)
def _process_trees(
        parents,
        branch_above,
        node_epoch,
        ids_asc_time,
        sample_locations_array_log,
        sample_ids,
        exponentiated_backward_transition_matrices,
        epoch_durations,
        precalculated_transitions_log,
        coal_rates_log
    ):
    """
    Parameters
    ----------
    parents : list
        Arrays containing ID of parent for each node, one array per tree
    branch_above : list
        Arrays containing branch above length (split across epochs) for each node, one array per tree
    node_epoch : list
        Arrays containing the epochs of each node, one array per tree
    ids_asc_time : List
        Arrays of nodes IDs in time ascending order, one array per tree
    sample_locations_array : numpy.ndarray
        Probability distribution vector for each sample location, converted to log space
    sample_ids : numpy.ndarray
        Order of sample IDs for `sample_locations_array`
    backward_transition_matrices : np.array

    coal_rates_log : numpy.ndarray
        Recipricol of suitability values for each deme in each epoch, converted to log space
    
    Returns
    -------
    composite_likelihood : float
        Composite log-likelihood across the trees
    """

    composite_likelihood = 0
    for i in prange(len(branch_above)):
        like = likelihood_of_tree_pre_log(
            parents=parents[i],
            branch_above=branch_above[i],
            node_epoch=node_epoch[i],
            ids_asc_time=ids_asc_time[i],
            sample_locations_array_log=sample_locations_array_log,
            sample_ids=sample_ids,
            exponentiated_backward_transition_matrices=exponentiated_backward_transition_matrices,
            epoch_durations=epoch_durations,
            precalculated_transitions_log=precalculated_transitions_log,
            coal_rates_log=coal_rates_log
        )
        composite_likelihood += like
    return composite_likelihood

def calc_composite_likelihood_for_parameters(
        parameters,
        world_map,
        parents,
        branch_above,
        node_epoch,
        ids_asc_time,
        sample_locations_array_log,
        sample_ids,
        output_file=None,
        verbose=False
    ):
    """
    Parameters
    ----------
    parameters : numpy.ndarray
        Combination of parameters used to build the migration surface
    world_map : terracotta.WorldMap
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
        Log-likelihood of the parameter combination
    """

    backward_transition_matrices = build_transition_matrices(world_map=world_map, parameters=parameters, direction="backward")
    
    exponentiated_backward_transition_matrices = np.zeros(backward_transition_matrices.shape)
    precalculated_full_epoch_branch_lengths = np.zeros(backward_transition_matrices.shape)
    for i in range(len(backward_transition_matrices)):
        exponentiated_backward_transition_matrices[i] = scipy.linalg.expm(backward_transition_matrices[i])
        if world_map.epoch_durations[i] != -1:
            precalculated_full_epoch_branch_lengths[i] = matrix_power(exponentiated_backward_transition_matrices[i], world_map.epoch_durations[i])
    precalculated_full_epoch_branch_lengths_log = np.log(np.maximum(precalculated_full_epoch_branch_lengths, 1e-99))

    pop_sizes = np.maximum(world_map.suitabilities, 1e-99)
    coal_rates_log = np.log(1 / (np.maximum(pop_sizes, 0.1)))

    composite_likelihood = _process_trees(        
        parents=parents,
        branch_above=branch_above,
        node_epoch=node_epoch,
        ids_asc_time=ids_asc_time,
        sample_locations_array_log=sample_locations_array_log,
        sample_ids=sample_ids,
        exponentiated_backward_transition_matrices=exponentiated_backward_transition_matrices,
        epoch_durations=world_map.epoch_durations,
        precalculated_transitions_log=precalculated_full_epoch_branch_lengths_log,
        coal_rates_log=coal_rates_log
    )

    if output_file is not None:
        with open(output_file, "a") as outfile:
            outfile.write(f"{parameters}\t{composite_likelihood}\n")
    if verbose:
        print(parameters, composite_likelihood, flush=True)
    return composite_likelihood