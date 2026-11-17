// bc_mpi.cpp - MPI (+ optional OpenMP) betweenness centrality, parallel over BFS sources.
//
//   build: mpicxx -O3 -std=c++17 -fopenmp bc_mpi.cpp -o bc_mpi    (without -fopenmp: pure MPI)
//   run:   OMP_NUM_THREADS=T mpirun -np P ./bc_mpi graph.el [-k K] [-o bc.txt]
//
// Same input, options and output as bc_serial.cpp. Parallel decomposition:
//   - rank 0 reads the graph and broadcasts the CSR arrays: every rank holds the whole graph
//   - source i goes to rank i mod P (cyclic: mixes cheap and expensive sources on every rank)
//   - inside a rank, T OpenMP threads take sources one at a time (dynamic schedule), each with
//     its own work arrays and BC array, so no locks or atomics are needed during the search
//   - the per-rank BC arrays are summed on rank 0 with one MPI_Reduce
// The only communication is the broadcast at the start and the reduce at the end.
//
// RESULT columns as in bc_serial.cpp. compute_max / compute_min: slowest / fastest rank;
// imbalance_wait: longest time a rank waited at the barrier for the slowest one.

#include <mpi.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

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

// Rank 0 has loaded g; send its size and CSR arrays so every rank holds a full copy.
void broadcast_graph(Graph& g) {
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

// BFS and ACCUMULATE for one source: identical to bc_serial.cpp.
// Per-source state, reused for every source.
struct State {
    std::vector<int> dist;       // d[v]: BFS distance from s, -1 = not reached
    std::vector<double> sigma;   // sigma[v]: number of shortest s-v paths (double: can exceed 2^64)
    std::vector<double> delta;   // delta[v]: dependency of s on v
    std::vector<int> Q;          // BFS queue
    std::vector<int> S;          // stack: vertices in the order BFS finished them
    explicit State(int n) : dist(n), sigma(n), delta(n), Q(n), S(n) {}
};

// BFS(G, s): shortest-path distances and path counts from s; fills the stack S.
// The predecessors of w are the neighbours v with d[v] = d[w] - 1, so no lists are stored.
int bfs(const Graph& g, int s, State& st) {
    std::fill(st.dist.begin(), st.dist.end(), -1);
    std::fill(st.sigma.begin(), st.sigma.end(), 0.0);
    st.dist[s] = 0;
    st.sigma[s] = 1.0;
    int head = 0, tail = 0, top = 0;
    st.Q[tail++] = s;
    while (head < tail) {
        int v = st.Q[head++];
        st.S[top++] = v;
        for (int i = g.off[v]; i < g.off[v + 1]; i++) {
            int w = g.adj[i];
            if (st.dist[w] < 0) {                        // w found for the first time
                st.dist[w] = st.dist[v] + 1;
                st.Q[tail++] = w;
            }
            if (st.dist[w] == st.dist[v] + 1)            // v is a predecessor of w
                st.sigma[w] += st.sigma[v];
        }
    }
    return top;                                          // number of vertices on the stack
}

// ACCUMULATE(G, s, bc): pop vertices farthest-first; each w passes its dependency back to
// its predecessors v:  delta[v] += sigma[v] / sigma[w] * (1 + delta[w]).  Then bc[w] += delta[w].
void accumulate(const Graph& g, int s, State& st, int top, std::vector<double>& bc) {
    std::fill(st.delta.begin(), st.delta.end(), 0.0);
    while (top > 0) {
        int w = st.S[--top];
        double c = (1.0 + st.delta[w]) / st.sigma[w];
        for (int i = g.off[w]; i < g.off[w + 1]; i++) {
            int v = g.adj[i];
            if (st.dist[v] == st.dist[w] - 1) st.delta[v] += st.sigma[v] * c;
        }
        if (w != s) bc[w] += st.delta[w];
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
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);   // only the main thread calls MPI
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
    int T = 1;
#ifdef _OPENMP
    T = omp_get_max_threads();
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    Graph g;
    if (rank == 0) g = load_graph(path);
    double tl = MPI_Wtime();
    broadcast_graph(g);
    double t1 = MPI_Wtime();

    std::vector<int> all = pick_sources(g.n, k), mine;
    for (size_t i = rank; i < all.size(); i += P) mine.push_back(all[i]);

    std::vector<double> bc_rank(g.n, 0.0);
    #pragma omp parallel
    {
        State st(g.n);                                  // private to this thread
        std::vector<double> bc_thread(g.n, 0.0);
        #pragma omp for schedule(dynamic, 1) nowait
        for (int i = 0; i < (int)mine.size(); i++) {
            int top = bfs(g, mine[i], st);
            accumulate(g, mine[i], st, top, bc_thread);
        }
        #pragma omp critical
        for (int v = 0; v < g.n; v++) bc_rank[v] += bc_thread[v];
    }
    double t2 = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);                        // time spent here = load imbalance
    double t3 = MPI_Wtime();

    std::vector<double> bc(rank == 0 ? g.n : 0);
    MPI_Reduce(bc_rank.data(), bc.data(), g.n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    double t4 = MPI_Wtime();

    double compute = t2 - t1, wait = t3 - t2, cmax, cmin, wmax;
    MPI_Reduce(&compute, &cmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&compute, &cmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&wait, &wmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        for (double& x : bc) x /= 2.0;                  // FINALIZE
        if (out) write_bc(bc, out);
        std::printf("RESULT,%s,%s,%d,%lld,%zu,%d,%d,0,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    T == 1 ? "mpi_only" : "mpi_omp", base_name(path).c_str(), g.n, g.m, all.size(),
                    P, T, tl - t0, t1 - tl, cmax, cmin, wmax, t4 - t3, t4 - t0);
    }
    MPI_Finalize();
    return 0;
}
