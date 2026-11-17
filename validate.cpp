// validate.cpp - compares two BC output files ("node raw_bc normalized_bc" per line).
//
//   build: g++ -O3 -std=c++17 validate.cpp -o validate
//   run:   ./validate test.txt ref.txt [tol]
// PASS (exit 0) if every vertex has |test - ref| / max(1, |ref|) <= tol (default 1e-9);
// values below 1 are thus compared absolutely, larger ones relatively.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

std::vector<double> read_bc(const char* path) {
    std::vector<double> bc;
    FILE* f = std::fopen(path, "r");
    if (!f) { std::fprintf(stderr, "cannot read %s\n", path); std::exit(1); }
    char line[256];
    while (std::fgets(line, sizeof line, f)) {
        int v;
        double x;
        if (line[0] != '#' && std::sscanf(line, "%d %lf", &v, &x) == 2) {
            if (v >= (int)bc.size()) bc.resize(v + 1, 0.0);
            bc[v] = x;
        }
    }
    std::fclose(f);
    return bc;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s test.txt ref.txt [tol]\n", argv[0]); return 2; }
    std::vector<double> a = read_bc(argv[1]), b = read_bc(argv[2]);
    const double tol = argc > 3 ? std::atof(argv[3]) : 1e-9;
    double worst = a.size() == b.size() ? 0.0 : INFINITY;     // different vertex counts: fail
    int at = -1;
    for (size_t v = 0; v < std::min(a.size(), b.size()); v++) {
        double err = std::abs(a[v] - b[v]) / std::max(1.0, std::abs(b[v]));
        if (err > worst) { worst = err; at = (int)v; }
    }
    const bool ok = worst <= tol;
    std::printf("%s  %s vs %s: max relative error %.1e", ok ? "PASS" : "FAIL", argv[1], argv[2], worst);
    if (at >= 0) std::printf(" (vertex %d)", at);
    std::printf(", tol %g\n", tol);
    return ok ? 0 : 1;
}
