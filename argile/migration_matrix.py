import numpy as np


def _calculate_migration_rate(world_map, epoch, connection, deme_0, deme_1, parameters, direction="backward"):
    """Migration rate is default rate * suit(deme_1)/suit(deme_0) * modifier"""

    if "coefficient" in world_map.parameters:
        m = parameters[world_map.parameters.index("coefficient")]
    else:
        m = 1

    suit_0 = world_map.suitabilities[epoch][deme_0]
    suit_1 = world_map.suitabilities[epoch][deme_1]
    if suit_0 == 0 or suit_1 == 0:
        suit_ratio = 0
    else:
        suit_ratio = min(suit_1/suit_0, 1000000)

    mod = world_map.migration_modifiers[epoch][connection]
    if mod in world_map.parameters:
        mod = parameters[world_map.parameters.index(mod)]
    else:
        mod = float(mod)
    if direction == "backward":
        rate = (m * suit_ratio) * mod
    else:
        rate = m * mod
    return rate

def build_transition_matrices(world_map, parameters, direction="backward"):
    """Builds the transition matrix based on the world map and migration rate parameters

    Row is the target deme, column is the source deme backwards in time.

    Parameters
    ----------
    parameters : np.array or list
    
    Returns
    -------
    transition_matrix : np.array
    """

    if len(parameters) != len(world_map.parameters):
        raise RuntimeError("Length of `parameters` does not equal WorldMap.parameters. You must provide values for all parameters, in order.")

    transition_matrix = np.zeros((len(world_map.epochs), len(world_map.demes),len(world_map.demes)))
    for e in range(len(world_map.epochs)):
        for f,connection in world_map.connections.iterrows():
            
            i_0 = world_map.demes.loc[world_map.demes["id"]==connection["deme_0"]].index[0]
            i_1 = world_map.demes.loc[world_map.demes["id"]==connection["deme_1"]].index[0]                
            transition_matrix[e, i_1, i_0] = _calculate_migration_rate(world_map=world_map, epoch=e, connection=f, deme_0=i_0, deme_1=i_1, parameters=parameters, direction=direction)
            transition_matrix[e, i_0, i_1] = _calculate_migration_rate(world_map=world_map, epoch=e, connection=f, deme_0=i_1, deme_1=i_0, parameters=parameters, direction=direction)

        diag = -np.sum(transition_matrix[e], axis=0)
        np.fill_diagonal(transition_matrix[e], diag)
    return transition_matrix