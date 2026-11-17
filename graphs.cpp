// graphs.cpp - writes the edge-list files read by the BC programs.
//
//   build: g++ -O3 -std=c++17 graphs.cpp -o graphs
//   ./graphs snap <ego_dir> <out.el>         SNAP ego networks (facebook/, twitter/) -> one graph
//   ./graphs rmat <scale> <ef> <out.el> [seed]  R-MAT: n = 2^scale, ef*n edge samples
//   ./graphs er   <n> <avg_deg> <out.el> [seed] Erdos-Renyi G(n, m), m = n*avg_deg/2
//   ./graphs grid <rows> <cols> <out.el>        2-D grid (large diameter: many BFS levels)
//
// Output: "# nodes N edges M", then "u v" per edge with u < v, no self loops, no duplicates.
//
// snap: every "u v" line of X.edges becomes an undirected edge, and ego X is joined to every
//   vertex listed in X.feat (the ego itself is not in X.edges). Original ids are relabelled
//   0..N-1 in numeric order; out.ids lists the original id of each new id.
// rmat: each edge picks one quadrant of the adjacency matrix per bit with probabilities
//   a, b, c, d = 0.57, 0.19, 0.19, 0.05 (Graph500), giving a skewed, social-network-like
//   degree distribution. Vertex ids are then randomly permuted so high-degree vertices are not
//   clustered at small ids. The same arguments and seed always give the same graph.

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using Edges = std::vector<std::pair<int, int>>;

Edges snap(const std::string& dir, const std::string& out, long long& n) {
    // ids can exceed 64 bits, so keep them as strings; numeric order = shorter first, then digits
    auto numeric_less = [](const std::string& x, const std::string& y) {
        return x.size() != y.size() ? x.size() < y.size() : x < y;
    };
    std::map<std::string, int, decltype(numeric_less)> id(numeric_less);
    std::vector<std::pair<std::string, std::string>> raw;
    DIR* d = opendir(dir.c_str());                       // POSIX: works with any C++17 compiler
    if (!d) { std::fprintf(stderr, "cannot open %s\n", dir.c_str()); std::exit(1); }
    for (dirent* e; (e = readdir(d));) {
        std::string file = e->d_name;
        if (file.size() <= 6 || file.compare(file.size() - 6, 6, ".edges")) continue;
        std::string ego = file.substr(0, file.size() - 6);
        if (!std::all_of(ego.begin(), ego.end(), ::isdigit)) continue;   // skips "._123.edges"
        id[ego];
        std::ifstream fe(dir + "/" + file);
        for (std::string a, b; fe >> a >> b;) raw.emplace_back(a, b);
        std::ifstream ff(dir + "/" + ego + ".feat");
        for (std::string line, a; std::getline(ff, line);)
            if (std::istringstream(line) >> a) raw.emplace_back(ego, a);
    }
    closedir(d);
    for (auto& [a, b] : raw) { id[a]; id[b]; }
    std::ofstream fid(out.substr(0, out.rfind('.')) + ".ids");
    n = 0;
    for (auto& [name, v] : id) { v = (int)n++; fid << name << '\n'; }
    Edges e;
    for (auto& [a, b] : raw) e.emplace_back(id[a], id[b]);
    return e;
}

Edges rmat(int scale, double ef, unsigned long seed) {
    const int n = 1 << scale;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    Edges e((long long)(ef * n));
    const double a = 0.57, b = 0.19, c = 0.19;                   // d = 1 - a - b - c = 0.05
    for (auto& [u, v] : e) {
        u = v = 0;
        for (int bit = 0; bit < scale; bit++) {
            double r = U(rng);
            u = (u << 1) | (r >= a + b);                             // lower half: c or d
            v = (v << 1) | ((r >= a && r < a + b) || r >= a + b + c);  // right half: b or d
        }
    }
    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    for (auto& [u, v] : e) { u = perm[u]; v = perm[v]; }
    return e;
}

Edges erdos_renyi(int n, double deg, unsigned long seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> V(0, n - 1);
    Edges e((long long)(deg * n / 2.0));
    for (auto& [u, v] : e) { u = V(rng); v = V(rng); }
    return e;
}

Edges grid(int rows, int cols) {
    Edges e;
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++) {
            if (c + 1 < cols) e.emplace_back(r * cols + c, r * cols + c + 1);
            if (r + 1 < rows) e.emplace_back(r * cols + c, (r + 1) * cols + c);
        }
    return e;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s snap <dir> <out.el> | rmat <scale> <ef> <out.el> [seed] | "
                             "er <n> <deg> <out.el> [seed] | grid <rows> <cols> <out.el>\n", argv[0]);
        return 1;
    }
    const std::string kind = argv[1];
    const std::string out = kind == "snap" ? argv[3] : argc > 4 ? argv[4] : "";
    const unsigned long seed = argc > 5 ? std::strtoul(argv[5], nullptr, 10) : 1;
    Edges e;
    long long n = 0;
    if (kind == "snap") e = snap(argv[2], out, n);
    else if (kind == "rmat") { n = 1LL << std::atoi(argv[2]); e = rmat(std::atoi(argv[2]), std::atof(argv[3]), seed); }
    else if (kind == "er") { n = std::atoll(argv[2]); e = erdos_renyi((int)n, std::atof(argv[3]), seed); }
    else if (kind == "grid") { n = std::atoll(argv[2]) * std::atoll(argv[3]); e = grid(std::atoi(argv[2]), std::atoi(argv[3])); }
    else { std::fprintf(stderr, "unknown graph type %s\n", argv[1]); return 1; }

    for (auto& [u, v] : e) { if (u > v) std::swap(u, v); n = std::max(n, (long long)v + 1); }
    e.erase(std::remove_if(e.begin(), e.end(), [](auto& x) { return x.first == x.second; }), e.end());
    std::sort(e.begin(), e.end());
    e.erase(std::unique(e.begin(), e.end()), e.end());

    FILE* f = std::fopen(out.c_str(), "w");
    std::fprintf(f, "# nodes %lld edges %zu\n", n, e.size());
    for (auto& [u, v] : e) std::fprintf(f, "%d %d\n", u, v);
    std::fclose(f);
    std::printf("%s: %lld nodes, %zu edges, average degree %.2f\n", out.c_str(), n, e.size(), 2.0 * e.size() / n);
    return 0;
}
