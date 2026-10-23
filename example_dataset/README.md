# Example dataset for `argile`

This dataset is a minimal example for the inputs of a `argile` world map.

## Input file structure

### demes.tsv

This is a tab separated file with two mandatory columns:
- `id`: integer ID of the deme
- `suitability`: either a float or a string with the format `time:suitability,time:suitability,...` if the suitability changes through time

### connections.tsv

This is a tab separated file with four mandatory columns:
- `id`: integer ID of the connection
- `deme_0`: smaller of the two deme IDs connected by the edge (matching `demes.tsv`)
- `deme_1`: larger of the two deme IDs connected by the edge (matching `demes.tsv`)
- `migration_modifier`: either a float (if known), string (if unknown), or a string with the format `time:modifier,time:modifier,...` if the modifier changes through time 

### samples.tsv

This is a tab separated file with two mandatory columns:
- `id`: integer ID of the sample as it appears in the trees
- `deme`: ID of the deme where the sample was found

### trees/

This folder contains all of the gene trees stored as `tskit` tree files, each with only one tree in it.

## Plotting the map

`terr_cotta` does not provide a method for plotting the world map directly. This is because `argile` does not make assumptions about how the map was built, and many maps will benefit from custom plotting implementations. Two common methods are using either `matplotlib` or `geopandas`, which will require installing these respective packages. These both utilize the fact that `demes.tsv` has two mandatory columns but any number of additional columns can be added.

### `matplotlib`

By including `xcoord` and `ycoord` columns to `demes.tsv`, we can quickly plot the center of each deme and connect them together using `connections.tsv`.

### `geopandas`

Demes are often defined by a spatial geometry. By storing these geometries in a `geopandas`, you can include these geometries in your plots. Under the hood, `geopandas` uses `shapely` and `matplotlib` to generate the plots.

## Regenerating the dataset

`create_dataset.py` contains all of the code needed to regenerate the example dataset, including setting the necessary random seeds. In order to run this script, you will need to have `msprime` installed in your environment as that package is not a dependency of `argile`. `create_dataset.py` includes functions for transforming `argile.WorldMap` into an `msprime.Demography` in order to simulate trees.