import numpy as np
from scipy import linalg
from .main import deconstruct_tree
from .migration_matrix import build_transition_matrices


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
    current_pos = np.prod(messages[children], axis=0)
    if len(children) > 1:
        current_pos = np.multiply(coal_rate, current_pos)
    current_pos = current_pos / sum(current_pos)
    return current_pos

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
        trans_prob = linalg.expm(transition_matrices[epoch]*branch_above[epoch])
        current_pos = np.matmul(trans_prob, current_pos)
    return current_pos

def ancs(tree, u):
    """Find all of the ancestors above a node for a tree

    Taken directly from https://github.com/tskit-dev/tskit/issues/2706

    Parameters
    ----------
    tree : tskit.Tree
        Tree to be traversed
    u : int
        The ID for the node of interest

    Returns
    -------
    An iterator over the ancestors of u in this tree
    """

    u = tree.parent(u)
    while u != -1:
        yield u
        u = tree.parent(u)

def calc_all_messages(
        parents,
        branch_above,
        node_epoch,
        ids_asc_time,
        sample_locations_array,
        sample_ids,
        backward_transition_matrices,
        forward_transition_matrices,
        coal_rates
    ):
    """"""

    num_demes = len(sample_locations_array[0])
    messages = np.ones((len(parents)*2, num_demes), dtype="float64")
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
        parent = parents[id]
        if parent != -1:
            messages[id] = calc_branch_message(
                current_pos,
                branch_above[id],
                backward_transition_matrices
            )
        else:   # collect roots here
            messages[id] = current_pos
    for id in ids_asc_time[::-1]:
        parent_of = np.where(parents==id)[0]
        for c in range(len(parent_of)):
            alt_children = np.delete(parent_of, c)
            current_pos = np.prod(np.concatenate((messages[[id+len(parents)]], messages[alt_children])), axis=0)
            if len(alt_children) > 0:
                current_pos = np.multiply(coal_rates[node_epoch[id]], current_pos)
            current_pos = current_pos / sum(current_pos)
            messages[parent_of[c]+len(parents)] = calc_branch_message(
                current_pos,
                branch_above[parent_of[c]],
                forward_transition_matrices
            )
    return messages

def trace_ancestors(start, parents):
    """
    """

    lineage = []
    a = start
    while a != -1:
        lineage.append(a)
        a = parents[a]
    return np.array(lineage)

def calc_length_to_node_combo(
        ancestor_time,
        child_time,
        branch_above_child
    ):
    
    branch_length_to_child = int(ancestor_time - child_time)
    bl_child = branch_above_child.copy()
    for e in range(len(branch_above_child)):
        if bl_child[e] >= branch_length_to_child:
            bl_child[e] = branch_length_to_child
        branch_length_to_child -= bl_child[e]
    bl_parent = branch_above_child - bl_child
    return bl_child, bl_parent

def track_lineage_over_time(
        sample,
        times,
        tree,
        world_map,
        parameters,
        uninformed=False
    ):

    ancestors = [sample] + list(ancs(tree=tree, u=sample))

    node_times = []
    for a in ancestors:
        node_times.append(int(tree.time(a)))

    pc_combos = []
    for t in times:
        for i,v in enumerate(node_times):
            if v > t:
                child = ancestors[i-1]
                parent = ancestors[i]
                break
            elif v == t:
                child = ancestors[i]
                parent = ancestors[i]
                break
        pc_combos.append((child, parent))

    if "alpha" in world_map.parameters:
        alpha = parameters[world_map.parameters.index("alpha")]
    else:
        alpha = 1

    
    sample_locations_array = world_map.sample_locations
    sample_ids = np.array(world_map.samples["id"])
    parents, branch_above, node_epoch, time_bin_widths, ids_asc_time = deconstruct_tree(tree, world_map.epochs)
    
    backward_transition_matrices = build_transition_matrices(world_map=world_map, parameters=parameters, direction="backward")
    forward_transition_matrices = build_transition_matrices(world_map=world_map, parameters=parameters, direction="forward")

    pop_sizes = np.maximum(world_map.suitabilities ** alpha, 1e-99)
    coal_rates = 1/(np.maximum(pop_sizes, 0.1))

    messages = calc_all_messages(
        parents,
        branch_above,
        node_epoch,
        ids_asc_time,
        sample_locations_array,
        sample_ids,
        backward_transition_matrices,
        forward_transition_matrices,
        coal_rates
    )

    positions = np.zeros((len(pc_combos), len(world_map.demes)))
    for element, node_combo in enumerate(pc_combos):
        if node_combo[0] == node_combo[1]:
            if node_combo[0] in sample_ids:
                node_pos = sample_locations_array[np.where(sample_ids==node_combo[0])[0][0]]
            else:
                combined = np.multiply(coal_rates[node_epoch[node_combo[0]]], np.prod(np.concatenate((messages[[node_combo[0]+len(parents)]], messages[np.where(parents==node_combo[0])[0]])), axis=0))
                node_pos = combined / sum(combined)
        else:
            if node_combo[0] in sample_ids:
                child_pos = sample_locations_array[np.where(sample_ids==node_combo[0])[0][0]]
            else:
                incoming_child_messages = messages[np.where(parents==node_combo[0])[0]]
                if len(incoming_child_messages) > 0:
                    combined = np.multiply(coal_rates[node_epoch[node_combo[0]]], np.prod(incoming_child_messages, axis=0))
                    child_pos = combined / sum(combined)
                else:
                    child_pos = np.ones((1,len(world_map.demes)))[0]
            if node_combo[1] in sample_ids:
                parent_pos = sample_locations_array[np.where(sample_ids==node_combo[1])[0][0]]
            else:
                backward_messages = np.where(parents==node_combo[1])[0]
                backward_messages = backward_messages[backward_messages != node_combo[0]]
                incoming_parent_messages = np.concatenate((messages[[node_combo[1]+len(parents)]], messages[backward_messages]))
                if len(incoming_parent_messages) > 0:
                    combined = np.multiply(coal_rates[node_epoch[node_combo[1]]], np.prod(incoming_parent_messages, axis=0))
                    parent_pos = combined / sum(combined)
                else:
                    parent_pos = np.ones((1,len(world_map.demes)))[0]

            bl_child, bl_parent = calc_length_to_node_combo(
                times[element],
                tree.time(node_combo[0]),
                branch_above[node_combo[0]]
            )

            outgoing_child_message = calc_branch_message(
                child_pos,
                bl_child,
                backward_transition_matrices
            )
            outgoing_parent_message = calc_branch_message(
                parent_pos,
                bl_parent,
                forward_transition_matrices
            )
        
            node_pos = np.multiply(outgoing_child_message, outgoing_parent_message)
        positions[element] = node_pos / sum(node_pos)

    return positions