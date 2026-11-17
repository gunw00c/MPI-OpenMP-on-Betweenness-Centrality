# How to run

## Requirements

- An MPI C++ compiler wrapper `mpicxx` (Open MPI) with OpenMP support, C++17.
- `bc_omp_gpu` is built only when `mpicxx` wraps the NVIDIA HPC SDK compiler (`nvc++`). It runs on NVIDIA GPUs.
- On Vanda: `module load nvhpc-hpcx-cuda12/26.1`. On macOS: Homebrew `open-mpi` and `libomp`.

Run every command from this folder.

## 1. Build

```sh
./run.sh build
```

The programs are compiled into `bin/`: `graphs`, `bc_serial`, `bc_brute`, `validate`, `bc_mpi`, `bc_omp`, and `bc_omp_gpu` when the NVIDIA HPC SDK is used.

## 2. Prepare the graphs

The real-world graphs come from the SNAP ego networks. Download `facebook.tar.gz` and `twitter.tar.gz` from https://snap.stanford.edu/data/ and unpack them. Then convert each folder:

```sh
./run.sh data path/to/facebook      # -> data/facebook.el, data/facebook.ids
./run.sh data path/to/twitter       # -> data/twitter.el,  data/twitter.ids
```

The experiments generate the synthetic graphs (R-MAT, Erdős–Rényi, grid) when they are missing. To make one by hand:

```sh
bin/graphs rmat 20 16 data/rmat_s20.el 1        # R-MAT, n = 2^20, edgefactor 16, seed 1
bin/graphs er 1048576 16 data/er_s20.el 1       # Erdős–Rényi, n = 2^20, average degree 16, seed 1
bin/graphs grid 40 60 data/grid.el              # 40 x 60 grid
```

## 3. Validate

```sh
./run.sh validate
```

This compares `bc_serial` against `bc_brute` (BC from its definition), and `bc_mpi`, `bc_omp` and `bc_omp_gpu` (if a GPU is present) against `bc_serial`. It uses Facebook and three small synthetic graphs. Each comparison prints `PASS` or `FAIL`, and the run ends with `ALL CHECKS PASSED` or with a non-zero exit code.

## 4. Run one program

```sh
bin/bc_serial data/facebook.el -o bc.txt                                     # exact BC
bin/bc_serial data/twitter.el -k 8192 -o bc.txt                              # 8,192 sampled sources
bin/bc_brute  data/facebook.el -o ref.txt                                    # BC from its definition
OMP_NUM_THREADS=1 mpirun -np 64 bin/bc_mpi data/twitter.el -k 8192 -o bc.txt # MPI only
OMP_NUM_THREADS=16 mpirun -np 4 --map-by slot:PE=16 --bind-to core \
    bin/bc_mpi data/twitter.el -k 8192 -o bc.txt                             # MPI + OpenMP, whole sources per thread
OMP_NUM_THREADS=16 mpirun -np 1 --map-by slot:PE=16 --bind-to core \
    bin/bc_omp data/twitter.el -k 1024 -o bc.txt                             # threads share each BFS
mpirun -np 2 bin/bc_omp_gpu data/twitter.el -k 8192 -o bc.txt                # one GPU per rank
bin/validate bc.txt ref.txt                                                  # compare two outputs
```

Options common to every BC program:

| Option | Meaning |
| --- | --- |
| `-k K` | use K sampled sources (fixed seed, so every program uses the same sample); default: all n sources, which gives exact BC |
| `-o file` | write `node raw_bc normalized_bc` per line |

Each program prints one `RESULT,...` line with its timings: load, bcast, compute_max, compute_min, imbalance_wait, reduce, total, in seconds.

## 5. Run the experiments

| Command | Experiment | Default output |
| --- | --- | --- |
| `./run.sh complexity` | time vs sources k (R-MAT scale 16); exact BC vs n, Brandes and brute force | `results/complexity.csv` |
| `./run.sh size` | time vs edges m and vs vertices n (R-MAT and Erdős–Rényi, k = 256) | `results/size.csv` |
| `./run.sh brute data/facebook.el` | `bc_brute` vs `bc_serial`, exact | `results/brute_facebook.csv` |
| `./run.sh strong <graph.el> [K]` | `bc_serial`, then `bc_mpi` on 1, 2, 4, … `P_MAX` cores | `results/strong_<graph>.csv` |
| `./run.sh weak graph` | weak scaling: R-MAT scale grows with the core count | `results/weak_graph.csv` |
| `./run.sh weak sources` | weak scaling: sources grow with the core count | `results/weak_sources.csv` |
| `./run.sh fine <graph.el> [K]` | `bc_omp` on 1 rank with 1, 2, 4, … `T_MAX` threads | `results/fine_<graph>.csv` |
| `./run.sh gpu <graph.el> [K]` | `bc_omp_gpu` on 1, 2, … `G_MAX` GPUs | `results/gpu_<graph>.csv` |
| `./run.sh device <graph.el> [K]` | on one machine: `bc_serial`, then `bc_mpi` and `bc_omp` on all `P_MAX` cores, then `bc_omp_gpu` on 1 … `G_MAX` GPUs | `results/device_<graph>.csv` |

Every run appends one CSV row: `version,graph,n,m,k,ranks,threads,batch,load,bcast,compute_max,compute_min,imbalance_wait,reduce,total,host,rep`.

The commands read these environment variables:

| Variable | Meaning | Default |
| --- | --- | --- |
| `CSV` | output file | see the table above |
| `REPS` | repetitions of every run | 3 |
| `P_MAX` | largest core count for `strong` and `weak` | physical cores of this node |
| `TPR` | OpenMP threads per MPI rank in `strong` and `weak` | 1 |
| `T_MAX` | largest thread count for `fine` | `P_MAX` |
| `G_MAX` | largest GPU count for `gpu` | GPUs of this node |
| `HOSTFILE`, `PPN` | MPI host file and ranks per node, for runs over several nodes | none |
| `SCALE`, `K` | R-MAT scale at 1 core and sources, for `weak` | graph: 15, 256; sources: 17, 32 |
| `RUN_TIMEOUT` | seconds after which a hung MPI run is killed | none |
| `MPI_OPTS` | extra `mpirun` options, for example the network between nodes | none |

Examples:

```sh
CSV=results/mpi/strong.csv P_MAX=64 ./run.sh strong data/twitter.el 8192
CSV=results/hybrid/strong.csv TPR=16 P_MAX=64 ./run.sh strong data/twitter.el 8192
CSV=results/mpi/weak_graph.csv SCALE=14 K=1024 ./run.sh weak graph
CSV=results/omp/fine.csv T_MAX=64 ./run.sh fine data/twitter.el 1024
```

## 6. Run on Vanda (NUS HPC)

Copy this folder to Vanda along with `data/facebook.el` and `data/twitter.el`, then submit the jobs from inside it. Each job builds the programs on its compute node, writes its results under `results/`, and leaves its log file in this folder.

```sh
qsub vanda_serial.pbs    # 1 CPU node, free queue: validation, complexity tests, brute force vs Brandes
qsub vanda.pbs           # 4 CPU nodes, 256 cores, charged queue: bc_mpi strong and weak scaling, MPI + OpenMP
qsub vanda_omp.pbs       # 1 CPU node, free queue: validation, bc_omp on 1 to 64 threads
qsub vanda_gpu_charged.pbs  # 1 A40 GPU + 36 cores, charged queue: GPU validation, CPU baselines and bc_omp_gpu on the same node
qstat -u $USER           # job status
```

| Job script | Log | Results |
| --- | --- | --- |
| `vanda_serial.pbs` | `bc-serial.log` | `results/serial/` |
| `vanda.pbs` | `bc-256.log` | `results/mpi/`, `results/hybrid/` |
| `vanda_omp.pbs` | `bc-omp.log` | `results/omp/` |
| `vanda_gpu_charged.pbs` | `bc-gpu-charged.log` | `results/gpu/device.csv`, `results/gpu/sources.csv` |

Before the experiments, `vanda.pbs` looks for an MPI transport that connects the nodes: it tries 2 ranks on 2 nodes, then checks 128 and 256 ranks against `bc_serial`, and stops if any of these checks fails.
