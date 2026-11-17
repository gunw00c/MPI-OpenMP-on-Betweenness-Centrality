// bc_serial.cpp - serial betweenness centrality, Brandes' algorithm, written as in the
// pseudocode:
//
//   BETWEENNESS(G):  bc[v] = 0 for all v
//                    for each source s:  BFS(G, s);  ACCUMULATE(G, s, bc)
//                    FINALIZE(bc)
//
//   build: g++ -O3 -std=c++17 bc_serial.cpp -o bc_serial
//   run:   ./bc_serial graph.el [-k K] [-o bc.txt]
//
//   graph.el  "# nodes N edges M", then one undirected edge "u v" per line (0 <= u, v < N)
//   -k K      K sampled sources (fixed seed, the same sample in every program); default all N
//             sources = exact BC. Sampled BC is not rescaled.
//   -o file   write "node raw_bc normalized_bc" per line, normalized = raw * 2 / ((N-1)(N-2))
//
// Prints one line  RESULT,serial,graph,n,m,k,ranks,threads,batch,load,bcast,compute_max,
// compute_min,imbalance_wait,reduce,total  (times in seconds; MPI-only columns are 0).

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// Undirected graph in CSR form: the neighbours of v are adj[off[v] .. off[v+1]).
struct Graph {
    int n = 0;
    long long m = 0;
    std::vector<int> off, adj;
};

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

// FINALIZE(bc): the graph is undirected, so every pair (s, t) was counted from both ends.
void finalize(std::vector<double>& bc) {
    for (double& x : bc) x /= 2.0;
}

std::vector<double> betweenness(const Graph& g, const std::vector<int>& sources) {
    std::vector<double> bc(g.n, 0.0);
    State st(g.n);
    for (int s : sources) {
        int top = bfs(g, s, st);
        accumulate(g, s, st, top, bc);
    }
    finalize(bc);
    return bc;
}

// ---- input / output ----

Graph load_graph(const char* path) {
    Graph g;
    FILE* f = std::fopen(path, "r");
    if (!f || std::fscanf(f, "# nodes %d edges %lld", &g.n, &g.m) != 2) {
        std::fprintf(stderr, "cannot read %s\n", path);
        std::exit(1);
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

// All n vertices, or k of them chosen by a fixed-seed shuffle.
std::vector<int> pick_sources(int n, int k) {
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

void write_bc(const std::vector<double>& bc, const char* path) {
    const int n = (int)bc.size();
    const double norm = n > 2 ? 2.0 / ((double)(n - 1) * (n - 2)) : 1.0;
    FILE* f = std::fopen(path, "w");
    std::fprintf(f, "# node raw_bc normalized_bc\n");
    for (int v = 0; v < n; v++) std::fprintf(f, "%d %.6f %.10e\n", v, bc[v], bc[v] * norm);
    std::fclose(f);
}

std::string base_name(std::string p) {                   // "data/twitter.el" -> "twitter"
    p = p.substr(p.find_last_of('/') + 1);
    return p.substr(0, p.rfind('.'));
}

double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    const char *path = nullptr, *out = nullptr;
    int k = 0;
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "-k") && i + 1 < argc) k = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-o") && i + 1 < argc) out = argv[++i];
        else path = argv[i];
    }
    if (!path) { std::fprintf(stderr, "usage: %s graph.el [-k K] [-o bc.txt]\n", argv[0]); return 1; }

    double t0 = now();
    Graph g = load_graph(path);
    double t1 = now();
    std::vector<int> sources = pick_sources(g.n, k);
    std::vector<double> bc = betweenness(g, sources);
    double t2 = now();

    if (out) write_bc(bc, out);
    std::printf("RESULT,serial,%s,%d,%lld,%zu,1,1,0,%.6f,0,%.6f,%.6f,0,0,%.6f\n",
                base_name(path).c_str(), g.n, g.m, sources.size(), t1 - t0, t2 - t1, t2 - t1, t2 - t0);
    return 0;
}
