// bc_brute.cpp - betweenness centrality straight from its definition (no Brandes).
//
//   bc(v) = sum over all vertex pairs s < t (s != v != t) of  sigma_st(v) / sigma_st
//
//   sigma_st     number of shortest paths between s and t
//   sigma_st(v)  how many of them pass through v
//
// Two steps:
//   1. ALL-PAIRS BFS: a BFS from every vertex gives dist[s][t] and sigma[s][t] for all pairs.
//   2. PAIR SUM: v lies on a shortest s-t path exactly when dist[s][v] + dist[v][t] = dist[s][t];
//      the number of those paths is sigma[s][v] * sigma[v][t] (any shortest s-v path followed
//      by any shortest v-t path).
// Cost: O(n m) for step 1 and O(n^3) for step 2, with two n x n tables (12 n^2 bytes:
// Facebook 0.2 GB; Twitter would need 79 GB). Brandes needs O(n m) time and O(n + m) memory.
//
//   build: g++ -O3 -std=c++17 bc_brute.cpp -o bc_brute
//   run:   ./bc_brute graph.el [-o bc.txt]        (always exact: every pair)
// Output file and RESULT line as in bc_serial.cpp.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using Table = std::vector<std::vector<double>>;

std::vector<std::vector<int>> load_graph(const char* path, long long& m) {
    FILE* f = std::fopen(path, "r");
    int n;
    if (!f || std::fscanf(f, "# nodes %d edges %lld", &n, &m) != 2) {
        std::fprintf(stderr, "cannot read %s\n", path);
        std::exit(1);
    }
    std::vector<std::vector<int>> nbr(n);               // adjacency lists
    for (int u, v; std::fscanf(f, "%d %d", &u, &v) == 2;) {
        nbr[u].push_back(v);
        nbr[v].push_back(u);
    }
    std::fclose(f);
    return nbr;
}

// Step 1: BFS from s, filling row s of dist (-1 = unreachable) and sigma.
void bfs(const std::vector<std::vector<int>>& nbr, int s, std::vector<int>& dist, std::vector<double>& sigma) {
    std::vector<int> queue = {s};
    dist[s] = 0;
    sigma[s] = 1.0;
    for (size_t head = 0; head < queue.size(); head++) {
        int v = queue[head];
        for (int w : nbr[v]) {
            if (dist[w] < 0) { dist[w] = dist[v] + 1; queue.push_back(w); }
            if (dist[w] == dist[v] + 1) sigma[w] += sigma[v];
        }
    }
}

// Step 2: for every pair s < t, give each vertex v on a shortest s-t path its share.
// (The graph is undirected, so dist[v][t] = dist[t][v]; using row t keeps the inner loop
// reading memory in order.)
std::vector<double> pair_sum(const std::vector<std::vector<int>>& dist, const Table& sigma) {
    const int n = (int)dist.size();
    std::vector<double> bc(n, 0.0);
    for (int s = 0; s < n; s++)
        for (int t = s + 1; t < n; t++) {
            const int d_st = dist[s][t];
            if (d_st < 2) continue;                      // unreachable or adjacent: no middle vertex
            const double paths_st = sigma[s][t];
            for (int v = 0; v < n; v++)
                if (dist[s][v] > 0 && dist[t][v] > 0 && dist[s][v] + dist[t][v] == d_st)
                    bc[v] += sigma[s][v] * sigma[t][v] / paths_st;
        }
    return bc;
}

double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    const char *path = nullptr, *out = nullptr;
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "-o") && i + 1 < argc) out = argv[++i];
        else path = argv[i];
    }
    if (!path) { std::fprintf(stderr, "usage: %s graph.el [-o bc.txt]\n", argv[0]); return 1; }

    double t0 = now();
    long long m;
    std::vector<std::vector<int>> nbr = load_graph(path, m);
    const int n = (int)nbr.size();
    if (12.0 * n * n > 16e9) { std::fprintf(stderr, "n = %d needs %.0f GB of tables\n", n, 12.0 * n * n / 1e9); return 1; }
    double t1 = now();

    std::vector<std::vector<int>> dist(n, std::vector<int>(n, -1));
    Table sigma(n, std::vector<double>(n, 0.0));
    for (int s = 0; s < n; s++) bfs(nbr, s, dist[s], sigma[s]);
    std::vector<double> bc = pair_sum(dist, sigma);
    double t2 = now();

    if (out) {
        const double norm = n > 2 ? 2.0 / ((double)(n - 1) * (n - 2)) : 1.0;
        FILE* f = std::fopen(out, "w");
        std::fprintf(f, "# node raw_bc normalized_bc\n");
        for (int v = 0; v < n; v++) std::fprintf(f, "%d %.6f %.10e\n", v, bc[v], bc[v] * norm);
        std::fclose(f);
    }
    std::string name = path;
    name = name.substr(name.find_last_of('/') + 1);
    name = name.substr(0, name.rfind('.'));
    std::printf("RESULT,brute,%s,%d,%lld,%d,1,1,0,%.6f,0,%.6f,%.6f,0,0,%.6f\n",
                name.c_str(), n, m, n, t1 - t0, t2 - t1, t2 - t1, t2 - t0);
    return 0;
}
