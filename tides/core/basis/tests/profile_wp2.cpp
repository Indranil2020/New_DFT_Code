// WP2 CPU profiling harness. Times each WP2 operation across 3 repeats
// (cold + warm) and emits structured JSON-lines logs consumable by the
// benchmark dashboard (per 30-architecture/30: "structured logs (JSON lines)
// -> benchmark harness reads these").
//
// Per 60-benchmarks/60: 3 repeats; cold and warm reported separately.
// All times in milliseconds. JSON line per measurement:
//   {"op": "<name>", "device": "cpu", "variant": "cold|warm", "repeat": <n>,
//    "ms": <time>, "params": {...}, "ts": "<iso8601>"}

#include "basis/atomgen/atomic_lda.hpp"
#include "basis/atomgen/radial_solver.hpp"
#include "basis/atomgen/symmetric_eigensolver.hpp"
#include "basis/atomgen/lda_xc.hpp"
#include "basis/nao_generator.hpp"
#include "basis/two_center_integrals.hpp"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using ms_t = std::chrono::duration<double, std::milli>;

std::string IsoNow() {
  auto now = std::chrono::system_clock::now();
  std::time_t t = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
  gmtime_r(&t, &tm);
  std::ostringstream os;
  os << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
  return os.str();
}

// Emit a JSON line. Minimal escaping for the op name.
void Log(const std::string& op, const std::string& variant, int repeat,
         double ms, const std::string& params = "{}") {
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "{\"op\":\"" << op << "\",\"device\":\"cpu\",\"variant\":\""
            << variant << "\",\"repeat\":" << repeat << ",\"ms\":" << ms
            << ",\"params\":" << params << ",\"ts\":\"" << IsoNow() << "\"}\n";
}

double TimeMs(Clock::time_point t0, Clock::time_point t1) {
  return ms_t(t1 - t0).count();
}

void ProfileRadialSolve() {
  const std::string params = "{\"task\":\"T2.1\",\"desc\":\"hydrogenic radial solve\"}";
  for (int n_r : {2000, 4000, 8000, 16000}) {
    for (int rep = 0; rep < 3; ++rep) {
      auto t0 = Clock::now();
      auto st = tides::atomgen::RadialSolver::SolveHydrogenic(1, 0, 3, 80.0,
                                                              static_cast<std::size_t>(n_r));
      auto t1 = Clock::now();
      std::ostringstream p;
      p << "{\"n_r\":" << n_r << ",\"n_states\":3,\"err_1s\":"
        << std::scientific << std::setprecision(3)
        << std::fabs(st[0].epsilon + 0.5) << "}";
      Log("radial_solve", rep == 0 ? "cold" : "warm", rep, TimeMs(t0, t1), p.str());
    }
  }
}

void ProfileAtomicLDA() {
  const std::string params = "{\"task\":\"T2.1-obs2\",\"desc\":\"atomic LDA SCF\"}";
  for (int Z : {2, 10}) {  // He, Ne
    for (int rep = 0; rep < 3; ++rep) {
      tides::atomgen::AtomConfig cfg;
      cfg.Z = Z;
      int rem = Z;
      struct Orb { int n; int l; };
      const Orb order[] = {{1,0},{2,0},{2,1}};
      for (const auto& o : order) {
        if (rem <= 0) break;
        int cap = 2 * (2 * o.l + 1);
        int occ = std::min(rem, cap);
        cfg.shells.push_back({o.n, o.l, occ});
        rem -= occ;
      }
      auto t0 = Clock::now();
      auto res = tides::atomgen::AtomicLDA::Solve(cfg, 40.0, 6000, 0.4, 1e-10, 300);
      auto t1 = Clock::now();
      std::ostringstream p;
      p << "{\"Z\":" << Z << ",\"iters\":" << res.n_scf_iter
        << ",\"E\":" << std::scientific << std::setprecision(6)
        << res.total_energy << "}";
      Log("atomic_lda_scf", rep == 0 ? "cold" : "warm", rep, TimeMs(t0, t1),
          p.str());
    }
  }
}

void ProfileJacobiEig() {
  const std::string params = "{\"task\":\"T2.1-eig\",\"desc\":\"self-contained Jacobi eigensolver\"}";
  for (int n : {32, 64, 128, 256}) {
    for (int rep = 0; rep < 3; ++rep) {
      std::vector<double> a(static_cast<std::size_t>(n) * n);
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
          a[static_cast<std::size_t>(i) * n + j] =
              (i == j) ? 2.0 + 0.01 * i : (std::abs(i - j) == 1 ? 1.0 : 0.0);
      auto t0 = Clock::now();
      std::vector<double> w, v;
      tides::atomgen::SymmetricEigensolver::Solve(a, static_cast<std::size_t>(n), w, v);
      auto t1 = Clock::now();
      std::ostringstream p;
      p << "{\"n\":" << n << "}";
      Log("jacobi_eig", rep == 0 ? "cold" : "warm", rep, TimeMs(t0, t1), p.str());
    }
  }
}

void ProfileNaoGen() {
  const std::string params = "{\"task\":\"T2.2\",\"desc\":\"NAO generation\"}";
  for (int Z : {1, 2, 6}) {  // H, He, C
    for (int rep = 0; rep < 3; ++rep) {
      auto recipe = tides::basis::NaoGenerator::DzpRecipe(Z, "X");
      auto t0 = Clock::now();
      auto basis = tides::basis::NaoGenerator::Generate(recipe);
      auto t1 = Clock::now();
      std::ostringstream p;
      p << "{\"Z\":" << Z << ",\"n_functions\":" << basis.functions.size()
        << "}";
      Log("nao_generation", rep == 0 ? "cold" : "warm", rep, TimeMs(t0, t1),
          p.str());
    }
  }
}

void ProfileSpline() {
  const std::string params = "{\"task\":\"T2.4\",\"desc\":\"cubic spline build+eval\"}";
  for (int n_tab : {100, 500, 2000}) {
    for (int rep = 0; rep < 3; ++rep) {
      std::vector<double> R_tab, S_tab;
      for (int i = 0; i <= n_tab; ++i) {
        R_tab.push_back(8.0 * i / n_tab);
        S_tab.push_back(std::exp(-R_tab.back() * R_tab.back()));
      }
      auto t0 = Clock::now();
      tides::basis::CubicSpline spline(R_tab, S_tab);
      double dummy = 0;
      for (int i = 0; i < 10000; ++i) dummy += spline.Eval(4.0 * i / 10000);
      auto t1 = Clock::now();
      std::ostringstream p;
      p << "{\"n_tab\":" << n_tab << ",\"n_eval\":10000,\"dummy\":" << dummy
        << "}";
      Log("spline_build_eval", rep == 0 ? "cold" : "warm", rep, TimeMs(t0, t1),
          p.str());
    }
  }
}

}  // namespace

int main() {
  std::cerr << "# WP2 CPU profiling — JSON-lines to stdout\n";
  std::cerr << "# device: CPU (Intel MKL LAPACK for tridiagonal eig)\n";
  ProfileRadialSolve();
  ProfileAtomicLDA();
  ProfileJacobiEig();
  ProfileNaoGen();
  ProfileSpline();
  std::cerr << "# done\n";
  return 0;
}
