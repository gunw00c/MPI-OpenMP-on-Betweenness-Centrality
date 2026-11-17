#!/usr/bin/env bash
# Build, validate and benchmark the betweenness-centrality programs.
#
#   ./run.sh build                    compile into bin/ (bc_omp_gpu only with the NVIDIA HPC SDK)
#   ./run.sh data <snap_dir>          SNAP ego-network folder -> data/<folder>.el
#   ./run.sh validate                 serial vs BC from its definition; parallel vs serial
#   ./run.sh brute <graph.el>         bc_brute (BC from its definition) vs bc_serial (Brandes)
#   ./run.sh complexity               serial time vs sources k, and exact BC (Brandes, brute) vs n
#   ./run.sh strong <graph.el> [K]    serial, then bc_mpi on 1, 2, 4, ... P_MAX cores
#   ./run.sh fine <graph.el> [K]      serial, then bc_omp (one BFS shared by T threads), T = 1 .. T_MAX
#   ./run.sh gpu <graph.el> [K]       serial, then bc_omp_gpu on 1, 2, 4, ... G_MAX GPUs (1 rank each)
#   ./run.sh device <graph.el> [K]    same graph and sources on every processor of this machine: serial,
#                                     bc_mpi and bc_omp on all P_MAX cores, then bc_omp_gpu on 1 .. G_MAX GPUs
#   ./run.sh weak graph|sources       R-MAT weak scaling: graph size or source count grows with P
#   ./run.sh size                     serial time vs vertices n and vs edges m
#
# K = number of sampled sources (default: all = exact BC). Environment:
#   P_MAX  cores to scale to (default: physical cores)   TPR  threads per MPI rank (default 1)
#   REPS   repetitions (default 3)                       T_MAX  threads for `fine` (default P_MAX)
#   G_MAX  GPUs for `gpu` (default: GPUs of this node)   HOSTFILE, PPN  host file, slots per node
#   CSV    output file (default results/<command>.csv)   RUN_TIMEOUT  seconds before an MPI run is killed
#   MPI_OPTS  extra mpirun options (e.g. which network to use between nodes)
# Every run appends one row: version,graph,n,m,k,ranks,threads,batch,load,bcast,compute_max,
#   compute_min,imbalance_wait,reduce,total,host,rep   (times in seconds)
set -euo pipefail
cd "$(dirname "$0")"
cmd=${1:-}
shift || true
REPS=${REPS:-3}
TPR=${TPR:-1}
P_MAX=${P_MAX:-$(sysctl -n hw.physicalcpu 2>/dev/null || lscpu -p=CORE | grep -v '^#' | sort -u | wc -l)}
CSV=${CSV:-results/$cmd${1:+_$(basename "$1" .el)}.csv}
mkdir -p bin data results "$(dirname "$CSV")"
fail=0

mpi() {   # mpi <ranks> <threads per rank> <program> <args...>
    local np=$1 nt=$2 place
    shift 2
    place="--map-by slot:PE=$nt --bind-to core"                          # each thread its own core
    if [ -n "${PPN:-}" ] && [ $((np * nt)) -gt "$PPN" ]; then
        place="--map-by ppr:$((PPN / nt)):node:PE=$nt --bind-to core"     # spread over nodes
    fi
    if [ "$(uname)" = Darwin ]; then place="--oversubscribe --bind-to none"; fi   # macOS: no pinning
    # RUN_TIMEOUT (seconds): kill a run that hangs, e.g. on a broken network between nodes
    OMP_NUM_THREADS=$nt OMP_PLACES=cores OMP_PROC_BIND=close \
        ${RUN_TIMEOUT:+timeout "$RUN_TIMEOUT"} mpirun ${HOSTFILE:+--hostfile "$HOSTFILE"} ${MPI_OPTS:-} $place -np "$np" "$@"
}

record() {   # record <rep> <command...>: run it, append its RESULT line to $CSV
    local rep=$1 line
    shift
    [ -f "$CSV" ] || echo "version,graph,n,m,k,ranks,threads,batch,load,bcast,compute_max,compute_min,imbalance_wait,reduce,total,host,rep" > "$CSV"
    if ! line=$("$@" | grep '^RESULT,'); then          # failed or timed out: log it, keep going
        echo "FAILED (no RESULT line): $*" >&2
        return 0
    fi
    echo "${line#RESULT,},$(hostname -s),$rep" | tee -a "$CSV"
}

check() { bin/validate "$@" || fail=1; }

have_gpu() { [ -x bin/bc_omp_gpu ] && nvidia-smi -L > /dev/null 2>&1; }

powers() {   # powers <max>: 1 2 4 ... <= max, then max itself
    local c=1
    while [ $c -le "$1" ]; do echo $c; c=$((c * 2)); done
    if [ $((c / 2)) -ne "$1" ]; then echo "$1"; fi
}

core_counts() { powers "$P_MAX"; }

build() {
    local arch=-march=native omp=-fopenmp nv=0
    if mpicxx --version | grep -q nvc++; then arch=-tp=native; omp=-mp; nv=1      # NVIDIA HPC SDK
    elif [ "$(uname)" = Darwin ]; then
        local L; L=$(brew --prefix libomp)                                       # Apple clang
        arch=-mcpu=native; omp="-Xpreprocessor -fopenmp -I$L/include -L$L/lib -lomp"
    fi
    # one compiler (the MPI wrapper's) for every program, so serial and parallel times compare
    mpicxx -O3 -std=c++17 $arch graphs.cpp -o bin/graphs
    mpicxx -O3 -std=c++17 $arch bc_serial.cpp -o bin/bc_serial
    mpicxx -O3 -std=c++17 $arch bc_brute.cpp -o bin/bc_brute
    mpicxx -O3 -std=c++17 $arch validate.cpp -o bin/validate
    mpicxx -O3 -std=c++17 $arch $omp bc_mpi.cpp -o bin/bc_mpi
    mpicxx -O3 -std=c++17 $arch $omp bc_omp.cpp -o bin/bc_omp
    if [ $nv = 1 ]; then                    # OpenMP offload to NVIDIA GPUs (all compute capabilities)
        mpicxx -O3 -std=c++17 $arch -mp=gpu -gpu=ccall -DGPU bc_omp.cpp -o bin/bc_omp_gpu
    fi
}

validate() {
    local V=results/validation G name
    mkdir -p $V
    bin/graphs rmat 11 8 $V/rmat.el 7      # skewed degrees and isolated vertices
    bin/graphs grid 40 60 $V/grid.el       # diameter 98: many BFS levels
    bin/graphs er 3000 6 $V/er.el 7        # uniform random
    for G in data/facebook.el $V/rmat.el $V/grid.el $V/er.el; do
        name=$V/$(basename $G .el)
        bin/bc_serial $G -o $name.serial > /dev/null
        bin/bc_brute $G -o $name.ref > /dev/null                # BC from its definition
        check $name.serial $name.ref
        for cfg in "1 1" "4 1" "3 2" "2 4"; do                        # ranks x threads
            set -- $cfg
            mpi $1 $2 bin/bc_mpi $G -o $name.mpi > /dev/null
            check $name.mpi $name.serial
        done
        for cfg in "1 1" "1 4" "1 7" "3 2"; do                        # ranks x threads
            set -- $cfg
            mpi $1 $2 bin/bc_omp $G -o $name.omp > /dev/null
            check $name.omp $name.serial
        done
        if have_gpu; then
            for np in 1 2; do
                mpi $np 1 bin/bc_omp_gpu $G -o $name.gpu > /dev/null
                check $name.gpu $name.serial
            done
        fi
    done
    # sampled sources: every program must pick the same sample
    bin/bc_serial data/facebook.el -k 500 -o $V/k500.serial > /dev/null
    mpi 3 2 bin/bc_mpi data/facebook.el -k 500 -o $V/k500.mpi > /dev/null
    check $V/k500.mpi $V/k500.serial
    mpi 2 4 bin/bc_omp data/facebook.el -k 500 -o $V/k500.omp > /dev/null
    check $V/k500.omp $V/k500.serial
    if have_gpu; then
        mpi 2 1 bin/bc_omp_gpu data/facebook.el -k 500 -o $V/k500.gpu > /dev/null
        check $V/k500.gpu $V/k500.serial
    fi
    if [ $fail = 0 ]; then echo "ALL CHECKS PASSED"; else echo "SOME CHECKS FAILED"; exit 1; fi
}

strong() {   # fixed problem, more cores
    local G=$1 K=${2:-0} rep P
    for rep in $(seq "$REPS"); do
        record $rep bin/bc_serial "$G" -k "$K"
        for P in $(core_counts); do
            if [ "$P" -ge "$TPR" ]; then record $rep mpi $((P / TPR)) "$TPR" bin/bc_mpi "$G" -k "$K"; fi
        done
    done
}

fine() {   # one BFS at a time, shared by T threads of one rank
    local G=$1 K=${2:-0} rep T
    for rep in $(seq "$REPS"); do
        record $rep bin/bc_serial "$G" -k "$K"
        for T in $(powers "${T_MAX:-$P_MAX}"); do record $rep mpi 1 "$T" bin/bc_omp "$G" -k "$K"; done
    done
}

gpu() {   # one MPI rank per GPU, each GPU shares one BFS at a time among its threads
    local G=$1 K=${2:-0} rep np
    for rep in $(seq "$REPS"); do
        record $rep bin/bc_serial "$G" -k "$K"
        for np in $(powers "${G_MAX:-$(nvidia-smi -L | wc -l)}"); do
            record $rep mpi "$np" 1 bin/bc_omp_gpu "$G" -k "$K"
        done
    done
}

device() {   # CPU baselines and GPU on the same machine, same graph, same sources
    local G=$1 K=${2:-0} rep np
    for rep in $(seq "$REPS"); do
        record $rep bin/bc_serial "$G" -k "$K"                       # 1 core
        record $rep mpi "$P_MAX" 1 bin/bc_mpi "$G" -k "$K"           # all cores, whole sources each
        record $rep mpi 1 "$P_MAX" bin/bc_omp "$G" -k "$K"           # all cores share one BFS (GPU's algorithm)
        if have_gpu; then
            for np in $(powers "${G_MAX:-$(nvidia-smi -L | wc -l)}"); do
                record $rep mpi "$np" 1 bin/bc_omp_gpu "$G" -k "$K"
            done
        fi
    done
}

weak() {   # work per core fixed. graph: scale SCALE + log2(P), k = K.  sources: scale SCALE, k = K*P
    local mode=$1 P s k G rep lg
    if [ "$mode" = graph ]; then SCALE=${SCALE:-15}; K=${K:-256}; else SCALE=${SCALE:-17}; K=${K:-32}; fi
    for P in $(core_counts); do
        [ $((P & (P - 1))) -eq 0 ] || continue                        # powers of two only
        lg=0; while [ $((1 << lg)) -lt "$P" ]; do lg=$((lg + 1)); done
        if [ "$mode" = graph ]; then s=$((SCALE + lg)); k=$K; else s=$SCALE; k=$((K * P)); fi
        G=data/rmat_s$s.el
        [ -f $G ] || bin/graphs rmat $s 16 $G 1
        for rep in $(seq "$REPS"); do
            if [ "$P" = 1 ]; then record $rep bin/bc_serial $G -k $k; fi
            if [ "$P" -ge "$TPR" ]; then record $rep mpi $((P / TPR)) "$TPR" bin/bc_mpi $G -k $k; fi
        done
    done
}

size() {   # serial cost vs n (average degree 16) and vs m (n = 65,536), k = 256 sources
    local spec kind s e G rep specs=""
    for s in 12 13 14 15 16 17 18 19 20; do specs="$specs rmat:$s:16"; done
    for e in 2 4 8 32 64; do specs="$specs rmat:16:$e"; done
    for s in 12 14 16 18 20; do specs="$specs er:$s:16"; done
    mkdir -p data/syn
    for spec in $specs; do
        IFS=: read -r kind s e <<< "$spec"
        if [ $kind = rmat ]; then G=data/syn/rmat_s${s}_e$e.el; [ -f $G ] || bin/graphs rmat $s $e $G 1
        else G=data/syn/er_s${s}_d$e.el; [ -f $G ] || bin/graphs er $((1 << s)) $e $G 1; fi
        for rep in $(seq "$REPS"); do
            record $rep bin/bc_serial $G -k 256
        done
    done
}

brute() {   # definition-based O(n^3) BC vs Brandes O(n m), both serial and exact
    local rep
    for rep in $(seq "$REPS"); do
        record $rep bin/bc_brute "$1"
        record $rep bin/bc_serial "$1"
    done
}

complexity() {   # checks the O(k (n + m)) and O(n m) cost of Brandes against O(n^3) brute force
    local rep k s G
    G=data/rmat_s16.el                                                 # T1: time vs k, fixed graph
    [ -f $G ] || bin/graphs rmat 16 16 $G 1
    for rep in $(seq "$REPS"); do
        for k in 64 128 256 512 1024 2048 4096; do record $rep bin/bc_serial $G -k $k; done
    done
    mkdir -p data/syn                                                  # T2: exact BC vs n, degree 16
    for s in 10 11 12 13 14; do
        G=data/syn/er_s${s}_d16.el
        [ -f $G ] || bin/graphs er $((1 << s)) 16 $G 1
        for rep in $(seq "$REPS"); do
            record $rep bin/bc_serial $G                               # Brandes, all n sources
            if [ $s -le 13 ]; then record $rep bin/bc_brute $G; fi     # n = 16,384 would take ~20 min
        done
    done
}

case $cmd in
    build | validate | brute | complexity | strong | fine | gpu | device | weak | size) $cmd "$@" ;;
    data) bin/graphs snap "$1" "data/$(basename "$1").el" ;;
    *) sed -n '2,23p' "$0"; exit 1 ;;
esac
