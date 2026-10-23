# `argile`

`argile` (formerly `terracotta`) is a Python package for estimating migration surfaces and the locations of genetic ancestors from gene trees. The latest release is available either here on GitHub or through [PyPI](https://pypi.org/project/argile/). See [example_dataset/](https://github.com/kitchensjn/argile/tree/main/example_dataset) for details about file formats.

#### Estimate the most likely migration surface

```
import argile

result, loglikelihood = argile.run(
    demes_path="example_dataset/demes.tsv",
    connections_path="example_dataset/connections.tsv",
    samples_path="example_dataset/samples.tsv",
    trees_dir_path="example_dataset/trees",
    output_file="output.tsv"
)
```

`argile.run()` estimates the most likely migration surface using the `L-BFGS-B` algorithm. It estimates values for the `coefficient` (the default migration rate) and any migration rate modifier variables (here `r0`, `r1`, and `r2`). These variables will be automatically included in this search if present in `connections.tsv`. This fucntion takes the paths to the files as input rather than loading the world map prior.


#### Loading the world map

```
import pandas as pd
import argile

demes = pd.read_csv("example_dataset/demes.tsv", sep="\t")
connections = pd.read_csv("example_dataset/connections.tsv", sep="\t")
samples = pd.read_csv("example_dataset/samples.tsv", sep="\t")
world_map = argile.WorldMap(demes, connections, samples)
```