# ParChain: A Framework for Parallel Hierarchical Agglomerative Clustering using Nearest-Neighbor Chain

This repository contains a cleaned version of the code for [ParChain: A Framework for Parallel Hierarchical Agglomerative Clustering using Nearest-Neighbor Chain](http://arxiv.org/abs/2106.04727).

To get the submodules:
```bash
git pull
git submodule update --init
```

## Installation

Compiler:
* g++ = 7.5.0

## Compilation

```bash
g++ -O3 -std=c++20 -mcx16  -ldl -pthread -I../external/parlaylib/include linkage.cpp -o linkage
```

run the command in `hac` folder to compile the version that uses cache tables and range queries. 

run the command in `hac-matrix` folder to compile the version that uses a distance matrix.



## Run

./linkage -method [METHOD] -cachesize [cache size] -d [dim] -o [output] [dataset]

* `METHOD` can be "complete",  "ward", "avg" (average linkage with Euclidean distance metric), or "avgsq" (average linkage with squared Euclidean distance metric).
* `cache size` is the size of each hash table. if `cache size=1`, no cache will be used. 
* `output` is the output file of the dendrogram

example: ./linkage -method complete -d 2 /path/to/dataset