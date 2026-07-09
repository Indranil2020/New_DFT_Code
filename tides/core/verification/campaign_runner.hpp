#pragma once

// T9.7: Campaign runner + reproducibility archiver.
//
// Runs a campaign of benchmark cases and archives results for reproducibility.
// A campaign is a collection of test cases with configurations that are run
// in sequence, with results stored in a structured archive.
//
// Components:
//   1. CampaignCase: a single benchmark case (system + config + expected values)
//   2. CampaignResult: results from running a campaign
//   3. Archive: stores campaign results with metadata for reproducibility
//
// The archive format is JSON-based, containing:
//   - Campaign metadata (name, timestamp, git hash, machine info)
//   - Per-case results (energy, forces, timing, pass/fail)
//   - Reproducibility hash (deterministic hash of inputs + results)
//
// Observable: campaign results are archived and retrievable; reproducibility
// hash verifies that re-running with the same inputs gives the same results.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <functional>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "common/status.hpp"

namespace tides::verification {

// Configuration for a single benchmark case.
struct CampaignCase {
  std::string name;
  std::string system;           // e.g., "H2O_64", "Si_216"
  std::string config_id;        // e.g., "PBE_DZP", "HSE06_TZP"
  int n_atoms = 0;
  double expected_energy = 0.0; // Ha (reference value)
  double energy_tolerance = 0.0; // Ha (allowed deviation)
  std::string notes;
};

// Result of a single campaign case.
struct CaseResult {
  std::string name;
  std::string system;
  std::string config_id;
  double computed_energy = 0.0;  // Ha
  double energy_error = 0.0;     // |computed - expected|
  double wall_time_s = 0.0;
  bool passed = false;
  std::string error_msg;
};

// Machine information for reproducibility.
struct MachineInfo {
  std::string hostname;
  std::string cpu_model;
  std::string gpu_model;
  int n_cpu_cores = 0;
  int n_gpus = 0;
  std::string os_version;
};

// Campaign result (full archive entry).
struct CampaignResult {
  std::string campaign_name;
  std::string timestamp;
  std::string git_hash;
  MachineInfo machine;
  std::vector<CaseResult> cases;
  int n_passed = 0;
  int n_failed = 0;
  bool overall_pass = false;

  // Reproducibility hash: FNV-1a hash of campaign name + all case results.
  [[nodiscard]] std::string reproducibility_hash() const {
    std::uint64_t hash = 14695981039346656037ULL;  // FNV offset basis
    auto fnv1a = [&](const std::string& s) {
      for (unsigned char c : s) {
        hash ^= c;
        hash *= 1099511628211ULL;  // FNV prime
      }
    };
    fnv1a(campaign_name);
    for (const auto& c : cases) {
      fnv1a(c.name);
      fnv1a(c.system);
      fnv1a(c.config_id);
      // Hash the energy as a string (deterministic).
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(15) << c.computed_energy;
      fnv1a(oss.str());
      oss.str("");
      oss << std::fixed << std::setprecision(6) << c.wall_time_s;
      fnv1a(oss.str());
    }
    std::ostringstream hex;
    hex << std::hex << std::setfill('0') << std::setw(16) << hash;
    return hex.str();
  }

  [[nodiscard]] std::string toJson() const {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(12);
    ss << "{\n";
    ss << "  \"campaign_name\": \"" << campaign_name << "\",\n";
    ss << "  \"timestamp\": \"" << timestamp << "\",\n";
    ss << "  \"git_hash\": \"" << git_hash << "\",\n";
    ss << "  \"reproducibility_hash\": \"" << reproducibility_hash() << "\",\n";
    ss << "  \"overall_pass\": " << (overall_pass ? "true" : "false") << ",\n";
    ss << "  \"summary\": {"
       << "\"passed\": " << n_passed
       << ", \"failed\": " << n_failed << "},\n";
    ss << "  \"machine\": {\n";
    ss << "    \"hostname\": \"" << machine.hostname << "\",\n";
    ss << "    \"cpu_model\": \"" << machine.cpu_model << "\",\n";
    ss << "    \"gpu_model\": \"" << machine.gpu_model << "\",\n";
    ss << "    \"n_cpu_cores\": " << machine.n_cpu_cores << ",\n";
    ss << "    \"n_gpus\": " << machine.n_gpus << "\n";
    ss << "  },\n";
    ss << "  \"cases\": [\n";
    for (std::size_t i = 0; i < cases.size(); ++i) {
      const auto& c = cases[i];
      ss << "    {\n";
      ss << "      \"name\": \"" << c.name << "\",\n";
      ss << "      \"system\": \"" << c.system << "\",\n";
      ss << "      \"config_id\": \"" << c.config_id << "\",\n";
      ss << "      \"computed_energy\": " << c.computed_energy << ",\n";
      ss << "      \"energy_error\": " << c.energy_error << ",\n";
      ss << "      \"wall_time_s\": " << c.wall_time_s << ",\n";
      ss << "      \"passed\": " << (c.passed ? "true" : "false") << "\n";
      ss << "    }" << (i + 1 < cases.size() ? "," : "") << "\n";
    }
    ss << "  ]\n";
    ss << "}\n";
    return ss.str();
  }
};

// Run a campaign of benchmark cases.
// Each case is evaluated by the provided runner function, which returns
// the computed energy and wall time.
//
// @param cases     List of campaign cases to run
// @param runner    Function that takes a CampaignCase and returns (energy, wall_time_s)
// @param git_hash  Current git hash for reproducibility
// @param machine   Machine info for the archive
// @return          Campaign result with all case results
[[nodiscard]] inline CampaignResult RunCampaign(
    const std::vector<CampaignCase>& cases,
    const std::function<std::pair<double, double>(const CampaignCase&)>& runner,
    const std::string& git_hash = "",
    const MachineInfo& machine = {}) {
  CampaignResult result;
  result.campaign_name = "nightly";
  result.git_hash = git_hash;
  result.machine = machine;

  std::time_t now = std::time(nullptr);
  char ts[32];
  std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
  result.timestamp = ts;

  for (const auto& tc : cases) {
    CaseResult cr;
    cr.name = tc.name;
    cr.system = tc.system;
    cr.config_id = tc.config_id;

    auto [energy, wall_time] = runner(tc);
    cr.computed_energy = energy;
    cr.wall_time_s = wall_time;
    cr.energy_error = std::fabs(energy - tc.expected_energy);
    cr.passed = (tc.energy_tolerance > 0)
        ? (cr.energy_error <= tc.energy_tolerance)
        : true;

    if (cr.passed)
      result.n_passed++;
    else
      result.n_failed++;

    result.cases.push_back(cr);
  }

  result.overall_pass = (result.n_failed == 0);
  return result;
}

// Verify reproducibility: re-run a campaign and check that the hash matches.
[[nodiscard]] inline bool VerifyReproducibility(
    const CampaignResult& ref,
    const CampaignResult& replay) {
  return ref.reproducibility_hash() == replay.reproducibility_hash();
}

}  // namespace tides::verification
