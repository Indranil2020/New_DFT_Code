// T9.5: Competitor farm containers + parsers tests.
//
// Validates:
//   - Container specs are generated for all 6 competitors
//   - Parsers correctly extract energy from sample output files
//   - Unit conversion (eV -> Ha) is correct
//   - Unknown code returns error

#include "verification/competitor_farm.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

using tides::verification::CompetitorResult;
using tides::verification::ContainerSpec;
using tides::verification::GetContainerSpecs;
using tides::verification::ParseCompetitorOutput;
using tides::verification::ParseCP2K;
using tides::verification::ParseFHIaims;
using tides::verification::ParsePySCF;
using tides::verification::ParseVASP;

int Fail(const std::string& msg) {
  std::cerr << "FAIL: " << msg << '\n';
  return 1;
}

// T9.5a: Container specs for all 6 competitors.
int TestContainerSpecs() {
  std::cout << "\n=== T9.5a: Container specs ===\n";
  auto specs = GetContainerSpecs();
  std::cout << "  " << specs.size() << " competitor specs:\n";
  for (const auto& s : specs) {
    std::cout << "    " << s.code_name
              << " (dockerfile=" << s.dockerfile.size() << " chars"
              << ", input=" << s.input_template.size() << " chars)\n";
  }
  if (specs.size() != 6) return Fail("T9.5a: expected 6 competitors");
  for (const auto& s : specs) {
    if (s.dockerfile.empty()) return Fail("T9.5a: empty dockerfile for " + s.code_name);
    if (s.input_template.empty()) return Fail("T9.5a: empty input for " + s.code_name);
    if (s.run_script.empty()) return Fail("T9.5a: empty run script for " + s.code_name);
  }
  std::cout << "T9.5a: GREEN\n";
  return 0;
}

// T9.5b: VASP parser extracts energy from sample vasprun.xml.
int TestVASPParser() {
  std::cout << "\n=== T9.5b: VASP parser ===\n";
  // Write a minimal sample vasprun.xml fragment.
  const char* sample =
      "<xml>\n"
      "<calculation>\n"
      "<energy>\n"
      "<i name=\"e_0_energy\"> -14.3386 </i>\n"
      "</energy>\n"
      "</calculation>\n</xml>\n";
  std::string path = "/tmp/tides_test_vasprun.xml";
  std::ofstream f(path);
  f << sample;
  f.close();

  auto res = ParseVASP(path);
  std::cout << "  energy=" << res.total_energy << " Ha"
            << " parsed_ok=" << res.parsed_ok << '\n';
  if (!res.parsed_ok) return Fail("T9.5b: parse failed: " + res.parse_error);
  // -14.3386 eV / 27.2114 = -0.5269... Ha
  double expected = -14.3386 / 27.2114;
  if (std::fabs(res.total_energy - expected) > 1e-6)
    return Fail("T9.5b: energy mismatch");
  std::remove(path.c_str());
  std::cout << "T9.5b: GREEN\n";
  return 0;
}

// T9.5c: CP2K parser extracts energy from sample log.
int TestCP2KParser() {
  std::cout << "\n=== T9.5c: CP2K parser ===\n";
  const char* sample =
      "  ENERGY| Total FORCE_EVAL ( QS ) energy (a.u.):  -17.234567\n"
      "  ENERGY| Total FORCE_EVAL ( QS ) energy (a.u.):  -17.234568\n";
  std::string path = "/tmp/tides_test_cp2k.out";
  std::ofstream f(path);
  f << sample;
  f.close();

  auto res = ParseCP2K(path);
  std::cout << "  energy=" << res.total_energy << " Ha"
            << " parsed_ok=" << res.parsed_ok << '\n';
  if (!res.parsed_ok) return Fail("T9.5c: parse failed: " + res.parse_error);
  if (std::fabs(res.total_energy - (-17.234568)) > 1e-10)
    return Fail("T9.5c: energy mismatch (should be last value)");
  std::remove(path.c_str());
  std::cout << "T9.5c: GREEN\n";
  return 0;
}

// T9.5d: PySCF parser extracts energy from sample output.
int TestPySCFParser() {
  std::cout << "\n=== T9.5d: PySCF parser ===\n";
  const char* sample =
      "converged SCF energy = -1.1173470350043018\n";
  std::string path = "/tmp/tides_test_pyscf.out";
  std::ofstream f(path);
  f << sample;
  f.close();

  auto res = ParsePySCF(path);
  std::cout << "  energy=" << res.total_energy << " Ha"
            << " parsed_ok=" << res.parsed_ok << '\n';
  if (!res.parsed_ok) return Fail("T9.5d: parse failed: " + res.parse_error);
  if (std::fabs(res.total_energy - (-1.1173470350043018)) > 1e-10)
    return Fail("T9.5d: energy mismatch");
  std::remove(path.c_str());
  std::cout << "T9.5d: GREEN\n";
  return 0;
}

// T9.5e: Unknown code returns error.
int TestUnknownCode() {
  std::cout << "\n=== T9.5e: Unknown code ===\n";
  auto res = ParseCompetitorOutput("UNKNOWN_CODE", "/dev/null");
  if (res.parsed_ok) return Fail("T9.5e: should fail for unknown code");
  if (res.parse_error != "unknown code")
    return Fail("T9.5e: wrong error message");
  std::cout << "  error: " << res.parse_error << '\n';
  std::cout << "T9.5e: GREEN\n";
  return 0;
}

}  // namespace

int main() {
  if (TestContainerSpecs()) return 1;
  if (TestVASPParser()) return 1;
  if (TestCP2KParser()) return 1;
  if (TestPySCFParser()) return 1;
  if (TestUnknownCode()) return 1;
  std::cout << "\ncompetitor_farm_tests: ALL GREEN\n";
  return 0;
}
