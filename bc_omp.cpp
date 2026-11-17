// bc_omp.cpp - MPI + OpenMP betweenness centrality, parallel INSIDE each BFS (fine-grained).
//
//   CPU:  mpicxx -O3 -std=c++17 -fopenmp bc_omp.cpp -o bc_omp
//   GPU:  mpicxx -O3 -std=c++17 -mp=gpu -DGPU bc_omp.cpp -o bc_omp_gpu      (NVIDIA HPC SDK)
//   run:  OMP_NUM_THREADS=T mpirun -np P ./bc_omp graph.el [-k K] [-o bc.txt]
//
// bc_mpi.cpp gives every thread whole sources, which never interact (embarrassingly parallel).
// Here all threads of a rank - or all threads of its GPU - work on ONE source at a time, so they
// have to cooperate on every BFS level:
//   FORWARD, level d = 0, 1, 2, ...      (the frontier Q[lo, hi) = the vertices at distance d)
//     discover  each frontier vertex v scans its neighbours w. The first thread to change dist[w]
//               from -1 to d + 1 (atomic exchange) appends w to Q (atomic counter).
//     count     each new vertex w sums sigma[v] over its neighbours v at distance d ("pull": every
//               w is written by one thread only, so no floating-point atomics are needed).
//   BACKWARD, deepest level first: each vertex v at level d pulls from its neighbours w at d + 1
//               delta[v] = sigma[v] * sum (1 + delta[w]) / sigma[w],   then bc[v] += delta[v].
// Every step is one parallel loop over one level, and the levels must run in order: one barrier
// (CPU) or one kernel launch (GPU) per step, paid for every source. The same loops run on the CPU
// (parallel for) or the GPU (target teams distribute parallel for), chosen by -DGPU.
// MPI splits the sources over ranks exactly as bc_mpi.cpp does; with GPUs, each rank drives
// GPU (local rank mod GPUs per node). RESULT columns as in bc_mpi.cpp; version mpi_omp_fine (CPU)
// or mpi_omp_gpu (GPU), threads = OpenMP threads per rank (1 for GPU).

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#define PRAGMA(x) _Pragma(#x)
#ifdef GPU
#define PARALLEL_FOR(len) PRAGMA(omp target teams distribute parallel for)
#define ON_DEVICE PRAGMA(omp target)
#define FETCH_COUNT PRAGMA(omp target update from(cnt[0:1]))
#else   // small levels run on one thread: starting a parallel loop costs more than they do
#define PARALLEL_FOR(len) PRAGMA(omp parallel for schedule(dynamic, 64) if(len > 256))
#define ON_DEVICE
#define FETCH_COUNT
#endif

struct Graph {                       // CSR: neighbours of v are adj[off[v] .. off[v+1])
    int n = 0;
    long long m = 0;
    std::vector<int> off, adj;
};

Graph load_graph(const char* path) {
    Graph g;
    FILE* f = std::fopen(path, "r");
    if (!f || std::fscanf(f, "# nodes %d edges %lld", &g.n, &g.m) != 2) {
        std::fprintf(stderr, "cannot read %s\n", path);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::vector<int> eu, ev;
    for (int u, v; std::fscanf(f, "%d %d", &u, &v) == 2;) { eu.push_back(u); ev.push_back(v); }
    std::fclose(f);
    g.m = (long long)eu.size();
    g.off.assign(g.n + 1, 0);
    for (size_t i = 0; i < eu.size(); i++) { g.off[eu[i] + 1]++; g.off[ev[i] + 1]++; }
    std::partial_sum(g.off.begin(), g.off.end(), g.off.begin());
    g.adj.resize(2 * g.m);
    std::vector<int> pos(g.off.begin(), g.off.end() - 1);
    for (size_t i = 0; i < eu.size(); i++) {
        g.adj[pos[eu[i]]++] = ev[i];
        g.adj[pos[ev[i]]++] = eu[i];
    }
    return g;
}

void broadcast_graph(Graph& g) {     // rank 0 has loaded g; every rank gets a full copy
    long long size[2] = {g.n, g.m};
    MPI_Bcast(size, 2, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    g.n = (int)size[0];
    g.m = size[1];
    g.off.resize(g.n + 1);
    g.adj.resize(2 * g.m);
    MPI_Bcast(g.off.data(), g.n + 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(g.adj.data(), (int)(2 * g.m), MPI_INT, 0, MPI_COMM_WORLD);
}

std::vector<int> pick_sources(int n, int k) {           // same sample as bc_serial
    std::vector<int> s(n);
    std::iota(s.begin(), s.end(), 0);
    if (k > 0 && k < n) {
        std::mt19937_64 rng(42);
        std::shuffle(s.begin(), s.end(), rng);
        s.resize(k);
        std::sort(s.begin(), s.end());
    }
    return s;
}

// Work arrays (on the GPU when built with -DGPU). dist = -1, sigma = delta = 0 between sources.
struct State {
    int n;
    const int *off, *adj;
    int *dist, *Q, *cnt;             // Q: vertices in BFS order, grouped by level; cnt[0]: its length
    double *sigma, *delta, *bc;
};

// BFS and ACCUMULATE for one source, each step parallel over the vertices of one level.
void one_source(State& st, int s, std::vector<int>& level) {
    const int *off = st.off, *adj = st.adj;
    int *dist = st.dist, *Q = st.Q, *cnt = st.cnt;
    double *sigma = st.sigma, *delta = st.delta, *bc = st.bc;

    ON_DEVICE
    { dist[s] = 0; sigma[s] = 1.0; Q[0] = s; cnt[0] = 1; }
    level.assign({0, 1});                                // level d = Q[level[d] .. level[d+1])

    for (int d = 0;; d++) {                              // FORWARD
        const int lo = level[d], hi = level[d + 1];
        PARALLEL_FOR(hi - lo)
        for (int i = lo; i < hi; i++) {                  // discover level d + 1
            const int v = Q[i];
            for (int e = off[v]; e < off[v + 1]; e++) {
                const int w = adj[e];
                int seen, old, slot;
                #pragma omp atomic read
                seen = dist[w];
                if (seen >= 0) continue;
                #pragma omp atomic capture
                { old = dist[w]; dist[w] = d + 1; }
                if (old >= 0) continue;                  // another thread claimed w first
                #pragma omp atomic capture
                slot = cnt[0]++;
                Q[slot] = w;
            }
        }
        FETCH_COUNT
        const int next = cnt[0];
        if (next == hi) break;                           // nothing new: BFS finished
        PARALLEL_FOR(next - hi)
        for (int i = hi; i < next; i++) {                // count paths to level d + 1
            const int w = Q[i];
            double sum = 0.0;
            for (int e = off[w]; e < off[w + 1]; e++)
                if (dist[adj[e]] == d) sum += sigma[adj[e]];
            sigma[w] = sum;
        }
        level.push_back(next);
    }

    const int deepest = (int)level.size() - 2;
    for (int d = deepest - 1; d >= 1; d--) {             // BACKWARD (level 0 is s itself)
        const int lo = level[d], hi = level[d + 1];
        PARALLEL_FOR(hi - lo)
        for (int i = lo; i < hi; i++) {
            const int v = Q[i];
            double sum = 0.0;
            for (int e = off[v]; e < off[v + 1]; e++) {
                const int w = adj[e];
                if (dist[w] == d + 1) sum += (1.0 + delta[w]) / sigma[w];
            }
            delta[v] = sigma[v] * sum;
            bc[v] += delta[v];
        }
    }

    const int reached = level.back();
    PARALLEL_FOR(reached)
    for (int i = 0; i < reached; i++) {                  // reset only what this source touched
        const int v = Q[i];
        dist[v] = -1; sigma[v] = 0.0; delta[v] = 0.0;
    }
}

void write_bc(const std::vector<double>& bc, const char* path) {
    const int n = (int)bc.size();
    const double norm = n > 2 ? 2.0 / ((double)(n - 1) * (n - 2)) : 1.0;
    FILE* f = std::fopen(path, "w");
    std::fprintf(f, "# node raw_bc normalized_bc\n");
    for (int v = 0; v < n; v++) std::fprintf(f, "%d %.6f %.10e\n", v, bc[v], bc[v] * norm);
    std::fclose(f);
}

std::string base_name(std::string p) {
    p = p.substr(p.find_last_of('/') + 1);
    return p.substr(0, p.rfind('.'));
}

int main(int argc, char** argv) {
    int provided, rank, P;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    const char *path = nullptr, *out = nullptr;
    int k = 0;
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "-k") && i + 1 < argc) k = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-o") && i + 1 < argc) out = argv[++i];
        else path = argv[i];
    }
    if (!path) {
        if (rank == 0) std::fprintf(stderr, "usage: %s graph.el [-k K] [-o bc.txt]\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
#ifdef GPU
    const char* version = "mpi_omp_gpu";
    const int T = 1;
    MPI_Comm node;                                       // ranks on this node share its GPUs
    int local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
    MPI_Comm_rank(node, &local);
    if (omp_get_num_devices() == 0) {
        std::fprintf(stderr, "rank %d: no GPU found\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    omp_set_default_device(local % omp_get_num_devices());
#else
    const char* version = "mpi_omp_fine";
    const int T = omp_get_max_threads();
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    Graph g;
    if (rank == 0) g = load_graph(path);
    double tl = MPI_Wtime();
    broadcast_graph(g);

    const int n = g.n;
    std::vector<int> dist(n, -1), Q(n), cnt(1, 0);
    std::vector<double> sigma(n, 0.0), delta(n, 0.0), bc_rank(n, 0.0);
    State st{n, g.off.data(), g.adj.data(), dist.data(), Q.data(), cnt.data(),
             sigma.data(), delta.data(), bc_rank.data()};
#ifdef GPU
    const long long m2 = 2 * g.m;
    const int *off = st.off, *adj = st.adj;
    int *d = st.dist, *q = st.Q, *c = st.cnt;
    double *sg = st.sigma, *dl = st.delta, *b = st.bc;
    #pragma omp target enter data map(to: off[0:n + 1], adj[0:m2], d[0:n], sg[0:n], dl[0:n], b[0:n]) \
                                  map(alloc: q[0:n], c[0:1])
#endif
    double t1 = MPI_Wtime();                             // bcast column = broadcast + GPU setup

    std::vector<int> all = pick_sources(n, k), level;
    for (size_t i = rank; i < all.size(); i += P) one_source(st, all[i], level);

#ifdef GPU
    #pragma omp target exit data map(from: b[0:n]) map(delete: off[0:n + 1], adj[0:m2], d[0:n], \
                                 sg[0:n], dl[0:n], q[0:n], c[0:1])
#endif
    double t2 = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);                         // time spent here = load imbalance
    double t3 = MPI_Wtime();

    std::vector<double> bc(rank == 0 ? n : 0);
    MPI_Reduce(bc_rank.data(), bc.data(), n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    double t4 = MPI_Wtime();

    double compute = t2 - t1, wait = t3 - t2, cmax, cmin, wmax;
    MPI_Reduce(&compute, &cmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&compute, &cmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&wait, &wmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        for (double& x : bc) x /= 2.0;                   // FINALIZE
        if (out) write_bc(bc, out);
        std::printf("RESULT,%s,%s,%d,%lld,%zu,%d,%d,0,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    version, base_name(path).c_str(), n, g.m, all.size(), P, T, tl - t0,
                    t1 - tl, cmax, cmin, wmax, t4 - t3, t4 - t0);
    }
    MPI_Finalize();
    return 0;
}
