// T9.7: Campaign runner + reproducibility archiver tests.
//
// Validates:
//   - Campaign runner executes all cases and reports pass/fail
//   - Energy error is computed correctly
//   - Reproducibility hash is deterministic (same inputs -> same hash)
//   - Reproducibility verification detects changes
//   - JSON archive contains all required fields

#include "verification/campaign_runner.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using tides::verification::CampaignCase;
using tides::verification::CampaignResult;
using tides::verification::CaseResult;
using tides::verification::MachineInfo;
using tides::verification::RunCampaign;
using tides::verification::VerifyReproducibility;

int Fail(const std::string& msg) {
  std::cerr << "FAIL: " << msg << '\n';
  return 1;
}

// T9.7a: Campaign runner executes all cases.
int TestCampaignRunner() {
  std::cout << "\n=== T9.7a: Campaign runner ===\n";
  std::vector<CampaignCase> cases = {
    {"H2O_single", "H2O", "PBE_DZP", 3, -17.234, 1e-4, ""},
    {"He_atom", "He", "PBE_DZP", 1, -2.834, 1e-4, ""},
    {"H2_dimer", "H2", "PBE_DZP", 2, -1.174, 1e-4, ""},
  };

  // Runner: returns (energy, wall_time) for each case.
  auto runner = [](const CampaignCase& tc) -> std::pair<double, double> {
    // Simulate: return expected energy + small noise.
    double noise = 1e-6 * (tc.n_atoms % 3);
    return {tc.expected_energy + noise, 0.1 * tc.n_atoms};
  };

  auto result = RunCampaign(cases, runner, "abc123", {});

  std::cout << "  n_passed=" << result.n_passed
            << " n_failed=" << result.n_failed
            << " overall=" << (result.overall_pass ? "PASS" : "FAIL") << '\n';
  for (const auto& cr : result.cases)
    std::cout << "    " << cr.name << " E=" << cr.computed_energy
              << " err=" << cr.energy_error
              << " passed=" << cr.passed << '\n';

  if (result.n_passed != 3) return Fail("T9.7a: all 3 should pass");
  if (!result.overall_pass) return Fail("T9.7a: should pass overall");

  std::cout << "T9.7a: GREEN\n";
  return 0;
}

// T9.7b: Energy error and pass/fail logic.
int TestEnergyError() {
  std::cout << "\n=== T9.7b: Energy error ===\n";
  std::vector<CampaignCase> cases = {
    {"pass_case", "H2O", "PBE", 3, -17.0, 0.01, ""},
    {"fail_case", "H2O", "PBE", 3, -17.0, 0.001, ""},
  };

  auto runner = [](const CampaignCase& tc) -> std::pair<double, double> {
    if (tc.name == "pass_case")
      return {-17.005, 0.1};  // error = 0.005 <= 0.01 -> pass
    return {-17.005, 0.1};     // error = 0.005 > 0.001 -> fail
  };

  auto result = RunCampaign(cases, runner, "", {});

  if (result.cases[0].passed != true)
    return Fail("T9.7b: pass_case should pass");
  if (result.cases[1].passed != false)
    return Fail("T9.7b: fail_case should fail");
  if (result.n_passed != 1 || result.n_failed != 1)
    return Fail("T9.7b: wrong pass/fail counts");
  if (result.overall_pass)
    return Fail("T9.7b: should fail overall (1 failure)");

  std::cout << "  pass_case: err=" << result.cases[0].energy_error
            << " tol=" << 0.01 << " passed=" << result.cases[0].passed << '\n';
  std::cout << "  fail_case: err=" << result.cases[1].energy_error
            << " tol=" << 0.001 << " passed=" << result.cases[1].passed << '\n';
  std::cout << "T9.7b: GREEN\n";
  return 0;
}

// T9.7c: Reproducibility hash is deterministic.
int TestReproducibilityHash() {
  std::cout << "\n=== T9.7c: Reproducibility hash ===\n";
  std::vector<CampaignCase> cases = {
    {"H2O", "H2O", "PBE", 3, -17.0, 0.01, ""},
    {"He", "He", "PBE", 1, -2.8, 0.01, ""},
  };

  auto runner = [](const CampaignCase& tc) -> std::pair<double, double> {
    return {tc.expected_energy, 0.1};
  };

  auto r1 = RunCampaign(cases, runner, "abc", {});
  auto r2 = RunCampaign(cases, runner, "abc", {});

  std::string h1 = r1.reproducibility_hash();
  std::string h2 = r2.reproducibility_hash();
  std::cout << "  hash1=" << h1 << "\n  hash2=" << h2 << '\n';

  if (h1 != h2) return Fail("T9.7c: hashes should match for same inputs");
  if (h1.size() != 16) return Fail("T9.7c: hash should be 16 hex chars");
  if (!VerifyReproducibility(r1, r2))
    return Fail("T9.7c: VerifyReproducibility should return true");

  std::cout << "T9.7c: GREEN\n";
  return 0;
}

// T9.7d: Reproducibility detects changes.
int TestReproducibilityChange() {
  std::cout << "\n=== T9.7d: Reproducibility change detection ===\n";
  std::vector<CampaignCase> cases = {
    {"H2O", "H2O", "PBE", 3, -17.0, 0.01, ""},
  };

  auto runner1 = [](const CampaignCase& tc) -> std::pair<double, double> {
    return {tc.expected_energy, 0.1};
  };
  auto runner2 = [](const CampaignCase& tc) -> std::pair<double, double> {
    return {tc.expected_energy + 0.001, 0.1};  // different energy
  };

  auto r1 = RunCampaign(cases, runner1, "abc", {});
  auto r2 = RunCampaign(cases, runner2, "abc", {});

  if (VerifyReproducibility(r1, r2))
    return Fail("T9.7d: should detect different energies");

  std::cout << "  hash1=" << r1.reproducibility_hash()
            << "\n  hash2=" << r2.reproducibility_hash() << '\n';
  std::cout << "T9.7d: GREEN\n";
  return 0;
}

// T9.7e: JSON archive output.
int TestArchiveJson() {
  std::cout << "\n=== T9.7e: Archive JSON ===\n";
  std::vector<CampaignCase> cases = {
    {"H2O", "H2O", "PBE_DZP", 3, -17.0, 0.01, ""},
  };

  auto runner = [](const CampaignCase& tc) -> std::pair<double, double> {
    return {tc.expected_energy, 0.5};
  };

  MachineInfo machine;
  machine.hostname = "test-node";
  machine.cpu_model = "AMD EPYC";
  machine.n_cpu_cores = 64;

  auto result = RunCampaign(cases, runner, "deadbeef", machine);
  std::string json = result.toJson();

  std::cout << "  JSON length: " << json.size() << " chars\n";
  if (json.empty()) return Fail("T9.7e: JSON empty");
  if (json.find("\"campaign_name\"") == std::string::npos)
    return Fail("T9.7e: missing campaign_name");
  if (json.find("\"reproducibility_hash\"") == std::string::npos)
    return Fail("T9.7e: missing reproducibility_hash");
  if (json.find("\"machine\"") == std::string::npos)
    return Fail("T9.7e: missing machine info");
  if (json.find("\"hostname\"") == std::string::npos)
    return Fail("T9.7e: missing hostname");
  if (json.find("\"energy_error\"") == std::string::npos)
    return Fail("T9.7e: missing energy_error");

  std::cout << "T9.7e: GREEN\n";
  return 0;
}

}  // namespace

int main() {
  if (TestCampaignRunner()) return 1;
  if (TestEnergyError()) return 1;
  if (TestReproducibilityHash()) return 1;
  if (TestReproducibilityChange()) return 1;
  if (TestArchiveJson()) return 1;
  std::cout << "\ncampaign_runner_tests: ALL GREEN\n";
  return 0;
}
