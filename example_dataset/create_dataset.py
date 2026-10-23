import pandas as pd
import numpy as np
import math
import msprime
import tskit
from os import mkdir, remove
import matplotlib.pyplot as plt
import random
import argile



def _populate_vertices(num_rings, gap_between_rings=1):
    if num_rings < 0:
        raise RuntimeError("`num_rings` must be >=0.")

    vertices = [
        {
            "id": 0,
            "xcoord": 0,
            "ycoord": 0
        }
    ]

    counter = 1

    for ring in range(1, num_rings+1):
        for corner in range(6):
            angle = (corner/6)*(2*math.pi)
            x = math.sin(angle) * (gap_between_rings*ring)
            y = math.cos(angle) * (gap_between_rings*ring)
            vertices.append({
                "id": counter,
                "xcoord": x,
                "ycoord": y
            })
            counter += 1
            
            next_angle = ((corner+1)/6)*(2*math.pi)
            next_x = math.sin(next_angle) * (gap_between_rings*ring)
            next_y = math.cos(next_angle) * (gap_between_rings*ring)
            for vertex in range(1, ring):
                new_x = x + (vertex/ring) * (next_x-x)
                new_y = y + (vertex/ring) * (next_y-y)
                vertices.append({
                    "id": counter,
                    "xcoord": new_x,
                    "ycoord": new_y
                })
                counter += 1

    vertices = pd.DataFrame(vertices)
    return vertices

def _populate_faces(num_rings):
    triangles = []
    starting_a = 0
    starting_b = 1
    for ring in range(num_rings+1):
        working_a = starting_a + 0
        working_b = starting_b + 0
        for edge in range(6):
            for i in range(ring):
                if (edge == 5) and (i == ring-1):
                    triangles.append((starting_a, starting_b, working_b))
                    starting_a = working_a + 1
                    starting_b = working_b + 1
                else:
                    triangles.append((working_a, working_b, working_b+1))
                if i < ring-1:
                    if (edge == 5) and (i == ring-2):
                        triangles.append((starting_a, working_a, working_b+1))
                    else:
                        triangles.append((working_a, working_a+1, working_b+1))
                        working_a += 1
                working_b += 1

    edges = []
    for t,tri in enumerate(triangles):
        for i in range(len(tri)):
            if (i < len(tri)-1):
                edges.append((t, tri[i], tri[i+1]))
            else:
                edges.append((t, tri[0], tri[-1]))

    faces = pd.DataFrame(edges, columns=["face", "deme_0", "deme_1"])
    return faces

def create_rectangular_tri_grid(num_rings, output_directory="."):
    vertices = _populate_vertices(num_rings)
    vertices["suitability"] = 1

    faces = _populate_faces(num_rings)
    edges = faces[["deme_0", "deme_1"]].drop_duplicates()
    edges = edges.sort_values(by=["deme_0", "deme_1"]).reset_index(drop=True)
    edges["migration_modifier"] = 1
    edges["id"] = range(len(edges))
    edges = edges[["id", "deme_0", "deme_1", "migration_modifier"]]

    angle = (1/6)*(2*math.pi)
    top_y = math.cos(angle) * num_rings
    angle = (4/6)*(2*math.pi)
    bottom_y = math.cos(angle) * num_rings
    demes_for_removal = vertices.loc[(vertices["ycoord"]>top_y)|(vertices["ycoord"]<bottom_y), "id"]

    vertices = vertices.loc[-(vertices["id"].isin(demes_for_removal))]
    edges = edges.loc[-((edges["deme_0"].isin(demes_for_removal))|(edges["deme_1"].isin(demes_for_removal)))]

    return vertices, edges

def create_random_samples_file(
        demes_path,
        number_of_samples,
        allow_multiple_samples_per_deme=True,
        random_seed=None,
        output_path="samples.tsv"
    ):
    """Creates a samples file associated with a demes file with randomly placed samples

    Parameters
    ----------
    demes_path : str
        Path to the demes file
    number_of_samples : int
        Number of samples to be placed on the map
    allow_multiple_samples_per_deme : bool
        Whether to allow samples to be placed in the same deme. (default=True)
    output_path : str
        Path to directory where file will be written. (default="samples.tsv")
    """
    
    if random_seed is not None:
        np.random.seed(random_seed)

    demes = pd.read_csv(demes_path, sep="\t")
    random_samples = np.random.choice(demes["id"], number_of_samples, replace=allow_multiple_samples_per_deme)
    with open(output_path, "w") as samples_file:
        samples_file.write("\t".join(["id", "deme"]) + "\n")
        for id,sample in enumerate(random_samples):
            samples_file.write(f"{id}\t{sample}\n")

def _set_up_msprime_demography(world_map, pop_size, migration_rate, migration_modifier_variables):
    """Creates the msprime.Demography object for simulating trees

    Parameters
    ----------
    world_map : argile.WorldMap
    pop_size : int
        The population size of each deme
    migration_rates : dict
        Key is the type of connection and value is the migration rate of that connection type
    """

    demography = msprime.Demography()
    for i,row in world_map.demes.iterrows():
        suitability_formatter = row["suitability"].replace(",", ":").split(":")
        demography.add_population(
            name="Pop_"+str(row["id"]),
            initial_size=pop_size*float(suitability_formatter[1])
        )
        for epoch in range(2,len(suitability_formatter),2):
            demography.add_population_parameters_change(
                time=float(suitability_formatter[epoch]),
                population="Pop_"+str(row["id"]),
                initial_size=pop_size*float(suitability_formatter[epoch+1])
            )
    previous_epoch = None
    for epoch in world_map.epochs:

        for _,connection in world_map.connections.iterrows():
            modifier = world_map.get_connection_migration_modifier_at_time(connection["id"], epoch)
            try:
                modifier = float(modifier)
            except:
                modifier = migration_modifier_variables.get(modifier, -1)
                if modifier == -1:
                    raise RuntimeError("Did not find rate.")
            
            N_0 = world_map.get_deme_suitability_at_time(connection["deme_0"], epoch)
            N_1 = world_map.get_deme_suitability_at_time(connection["deme_1"], epoch)
            if N_0 == 0 or N_1 == 0:
                suit_rat_f = 0
                suit_rat_b = 0
            else:
                suit_rat_f = min(N_1/N_0, 1000000)
                suit_rat_b = min(N_0/N_1, 1000000)

            if epoch == world_map.epochs[0]:
                demography.set_migration_rate(
                    source="Pop_"+str(connection["deme_0"]),
                    dest="Pop_"+str(connection["deme_1"]),
                    rate=migration_rate*suit_rat_f*modifier
                )
                demography.set_migration_rate(
                    source="Pop_"+str(connection["deme_1"]),
                    dest="Pop_"+str(connection["deme_0"]),
                    rate=migration_rate*suit_rat_b*modifier
                )
            else:
                demography.add_migration_rate_change(
                    time=epoch,
                    source="Pop_"+str(connection["deme_0"]),
                    dest="Pop_"+str(connection["deme_1"]),
                    rate=migration_rate*suit_rat_f*modifier
                )
                demography.add_migration_rate_change(
                    time=epoch,
                    source="Pop_"+str(connection["deme_1"]),
                    dest="Pop_"+str(connection["deme_0"]),
                    rate=migration_rate*suit_rat_b*modifier
                )                
    demography.sort_events()
    return demography

def _simulate_independent_trees(
        world_map,
        number_of_trees,
        ploidy,
        pop_size,
        migration_rate,
        migration_modifier_variables,
        end_time,
        random_seed=None
    ):
    """Simulates trees under a demographic model set by the world map

    Parameters
    ----------
    world_map : argile.WorldMap
    number_of_trees : int
        The number of independent trees to simulate
    ploidy : int
        The ploidy of the samples
    allow_multiple_samples_per_deme : bool
    pop_size : int
        The population size of each deme
    migration_rates : dict
        Key is the type of connection and value is the migration rate of that connection type
    """

    demography = _set_up_msprime_demography(
        world_map=world_map,
        pop_size=pop_size,
        migration_rate=migration_rate,
        migration_modifier_variables=migration_modifier_variables
    )
    samples = []
    for s in world_map.samples["deme"]:
        samples.append(msprime.SampleSet(1, population="Pop_"+str(s)))
    ts = msprime.sim_ancestry(
        samples=samples,
        ploidy=ploidy,
        demography=demography,
        record_full_arg=True,
        num_replicates=number_of_trees,
        end_time=end_time,
        random_seed=random_seed
    )
    return ts

def create_trees_files(
        demes_path,
        connections_path,
        samples_path,
        number_of_trees,
        pop_size,
        migration_rate,
        migration_modifier_variables=None,
        ploidy=1,
        end_time=None,
        random_seed=None,
        output_directory="."
    ):
    """
    Parameters
    ----------
    demes_path : string
    samples_path : string
    number_of_trees : int
        The number of independent trees to simulate
    pop_size : int
        The population size of each deme
    migration_rates : dict
        Key is the type of connection and value is the migration rate of that connection type.
    ploidy : int
        The ploidy of the individuals. (default=1, haploid)
    migration_rate : float
        Single migration rate between neighboring demes. (default=None, ignored)
    
    output_directory : string
        Path to directory where file will be written. (default=".")
    """

    demes = pd.read_csv(demes_path, sep="\t")
    connections = pd.read_csv(connections_path, sep="\t")
    samples = pd.read_csv(samples_path, sep="\t")
    world_map = argile.WorldMap(demes=demes, connections=connections, samples=samples)

    mkdir(f"{output_directory}/trees")
    trees = _simulate_independent_trees(
        world_map=world_map,
        number_of_trees=number_of_trees,
        ploidy=ploidy,
        pop_size=pop_size,
        migration_rate=migration_rate,
        migration_modifier_variables=migration_modifier_variables,
        end_time=end_time,
        random_seed=random_seed
    )           
    for i,tree in enumerate(trees):
        tree.dump(f"{output_directory}/trees/{i}.trees")


np.random.seed(1000)
random.seed(1000)

demes, connections = create_rectangular_tri_grid(num_rings=7)

demes.to_csv("demes.tsv", sep="\t", index=False)

connections["migration_modifier"] = connections["migration_modifier"].astype(str)

connections.loc[(connections["deme_0"]==2)&(connections["deme_1"]==3), "migration_modifier"] = "r0"
connections.loc[(connections["deme_0"]==0)&(connections["deme_1"]==3), "migration_modifier"] = "r0"
connections.loc[(connections["deme_0"]==0)&(connections["deme_1"]==4), "migration_modifier"] = "r0"
connections.loc[(connections["deme_0"]==4)&(connections["deme_1"]==5), "migration_modifier"] = "r0"
connections.loc[(connections["deme_0"]==4)&(connections["deme_1"]==14), "migration_modifier"] = "r0"
connections.loc[(connections["deme_0"]==13)&(connections["deme_1"]==14), "migration_modifier"] = "r0"
connections.loc[(connections["deme_0"]==14)&(connections["deme_1"]==29), "migration_modifier"] = "r0"
connections.loc[(connections["deme_0"]==14)&(connections["deme_1"]==29), "migration_modifier"] = "r0"
connections.loc[(connections["deme_0"]==14)&(connections["deme_1"]==30), "migration_modifier"] = "r0"
connections.loc[(connections["deme_0"]==15)&(connections["deme_1"]==30), "migration_modifier"] = "r0"
connections.loc[(connections["deme_0"]==30)&(connections["deme_1"]==31), "migration_modifier"] = "r0"
connections.loc[(connections["deme_0"]==31)&(connections["deme_1"]==52), "migration_modifier"] = "r0"
connections.loc[(connections["deme_0"]==31)&(connections["deme_1"]==53), "migration_modifier"] = "r0"
connections.loc[(connections["deme_0"]==53)&(connections["deme_1"]==54), "migration_modifier"] = "r0"
connections.loc[(connections["deme_0"]==54)&(connections["deme_1"]==82), "migration_modifier"] = "r0"
connections.loc[(connections["deme_0"]==54)&(connections["deme_1"]==83), "migration_modifier"] = "r0"
connections.loc[(connections["deme_0"]==55)&(connections["deme_1"]==83), "migration_modifier"] = "r0"
connections.loc[(connections["deme_0"]==83)&(connections["deme_1"]==84), "migration_modifier"] = "r0"
connections.loc[(connections["deme_0"]==84)&(connections["deme_1"]==118), "migration_modifier"] = "r0"
connections.loc[(connections["deme_0"]==118)&(connections["deme_1"]==119), "migration_modifier"] = "r0"
connections.loc[(connections["deme_0"]==119)&(connections["deme_1"]==159), "migration_modifier"] = "r0"
connections.loc[(connections["deme_0"]==159)&(connections["deme_1"]==160), "migration_modifier"] = "r0"

connections.loc[(connections["deme_0"]==63)&(connections["deme_1"]==64), "migration_modifier"] = "r1"
connections.loc[(connections["deme_0"]==39)&(connections["deme_1"]==63), "migration_modifier"] = "r1"
connections.loc[(connections["deme_0"]==38)&(connections["deme_1"]==39), "migration_modifier"] = "r1"
connections.loc[(connections["deme_0"]==20)&(connections["deme_1"]==38), "migration_modifier"] = "r1"
connections.loc[(connections["deme_0"]==19)&(connections["deme_1"]==20), "migration_modifier"] = "r1"
connections.loc[(connections["deme_0"]==7)&(connections["deme_1"]==19), "migration_modifier"] = "r1"
connections.loc[(connections["deme_0"]==7)&(connections["deme_1"]==36), "migration_modifier"] = "r1"
connections.loc[(connections["deme_0"]==7)&(connections["deme_1"]==18), "migration_modifier"] = "r1"
connections.loc[(connections["deme_0"]==1)&(connections["deme_1"]==7), "migration_modifier"] = "r1"
connections.loc[(connections["deme_0"]==1)&(connections["deme_1"]==8), "migration_modifier"] = "r1"
connections.loc[(connections["deme_0"]==2)&(connections["deme_1"]==8), "migration_modifier"] = "r1"

connections.loc[(connections["deme_0"]==73)&(connections["deme_1"]==74), "migration_modifier"] = "r2"
connections.loc[(connections["deme_0"]==47)&(connections["deme_1"]==74), "migration_modifier"] = "r2"
connections.loc[(connections["deme_0"]==48)&(connections["deme_1"]==74), "migration_modifier"] = "r2"
connections.loc[(connections["deme_0"]==48)&(connections["deme_1"]==75), "migration_modifier"] = "r2"
connections.loc[(connections["deme_0"]==48)&(connections["deme_1"]==49), "migration_modifier"] = "r2"
connections.loc[(connections["deme_0"]==28)&(connections["deme_1"]==48), "migration_modifier"] = "r2"
connections.loc[(connections["deme_0"]==27)&(connections["deme_1"]==48), "migration_modifier"] = "r2"
connections.loc[(connections["deme_0"]==27)&(connections["deme_1"]==47), "migration_modifier"] = "r2"
connections.loc[(connections["deme_0"]==26)&(connections["deme_1"]==47), "migration_modifier"] = "r2"
connections.loc[(connections["deme_0"]==26)&(connections["deme_1"]==46), "migration_modifier"] = "r2"
connections.loc[(connections["deme_0"]==25)&(connections["deme_1"]==46), "migration_modifier"] = "r2"
connections.loc[(connections["deme_0"]==25)&(connections["deme_1"]==45), "migration_modifier"] = "r2"
connections.loc[(connections["deme_0"]==25)&(connections["deme_1"]==44), "migration_modifier"] = "r2"
connections.loc[(connections["deme_0"]==24)&(connections["deme_1"]==44), "migration_modifier"] = "r2"
connections.loc[(connections["deme_0"]==43)&(connections["deme_1"]==44), "migration_modifier"] = "r2"
connections.loc[(connections["deme_0"]==43)&(connections["deme_1"]==69), "migration_modifier"] = "r2"
connections.loc[(connections["deme_0"]==68)&(connections["deme_1"]==69), "migration_modifier"] = "r2"
connections.loc[(connections["deme_0"]==69)&(connections["deme_1"]==100), "migration_modifier"] = "r2"
connections.loc[(connections["deme_0"]==100)&(connections["deme_1"]==101), "migration_modifier"] = "r2"

connections.to_csv("connections.tsv", sep="\t", index=False)

create_random_samples_file(
    demes_path="demes.tsv",
    number_of_samples=100,
    allow_multiple_samples_per_deme=True,
    random_seed=100,
    output_path="samples.tsv"
)

with open("demes_with_extra_sample.tsv", "w") as outfile:
    with open("demes.tsv", "r") as demesfile:
        for line in demesfile:
            outfile.write(line)
        outfile.write(f"9999\t60\t60\t1\n")

with open("samples_with_extra_sample.tsv", "w") as outfile:
    with open("samples.tsv", "r") as samplesfile:
        for line in samplesfile:
            outfile.write(line)
        sample = line.split("\t")[0]
        outfile.write(f"{int(sample)+1}\t9999\n")

create_trees_files(
    demes_path="demes_with_extra_sample.tsv",
    connections_path="connections.tsv",
    samples_path="samples_with_extra_sample.tsv",
    number_of_trees=1000,
    pop_size=500,
    migration_rate=0.001,
    migration_modifier_variables={"r0":0.001, "r1":0.01, "r2":0.1},
    end_time=30000
)

remove("demes_with_extra_sample.tsv")
remove("samples_with_extra_sample.tsv")