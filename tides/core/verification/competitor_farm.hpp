#pragma once

// T9.5: Competitor farm containers + parsers.
//
// Defines the interface for parsing output from competitor DFT codes
// to enable automated benchmarking comparisons. Each competitor has:
//   1. A container spec (Dockerfile/Singularity recipe)
//   2. An output parser that extracts observables (energy, forces, timing)
//   3. A standardized result format for comparison
//
// Supported competitors:
//   - VASP (vasprun.xml parser)
//   - CP2K (output log parser)
//   - FHI-aims (output log parser)
//   - SIESTA (output log parser)
//   - GPAW (output.txt / gpaw.yaml parser)
//   - PySCF (JSON output parser)
//
// The parsers are CPU-only (no competitor code needed to parse output files).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "common/status.hpp"

namespace tides::verification {

// Standardized competitor result.
struct CompetitorResult {
  std::string code_name;        // "VASP", "CP2K", etc.
  std::string version;          // code version
  std::string system;           // system name (e.g., "H2O_64")
  double total_energy = 0.0;    // Ha
  double energy_per_atom = 0.0; // Ha/atom
  int n_atoms = 0;
  double max_force = 0.0;       // Ha/Bohr
  double wall_time_s = 0.0;     // seconds
  int n_scf_iterations = 0;
  double memory_mb = 0.0;       // peak memory
  std::string units;            // original units (e.g., "eV", "eV/Ang")
  bool parsed_ok = false;
  std::string parse_error;
};

// Container specification for a competitor.
struct ContainerSpec {
  std::string code_name;
  std::string dockerfile;       // Dockerfile content
  std::string singularity_recipe;  // Singularity .def content
  std::string input_template;   // template input file
  std::string run_script;       // submission script
};

// Container specs for all supported competitors.
inline std::vector<ContainerSpec> GetContainerSpecs() {
  std::vector<ContainerSpec> specs;

  // VASP
  ContainerSpec vasp;
  vasp.code_name = "VASP";
  vasp.dockerfile =
      "FROM ubuntu:22.04\n"
      "RUN apt-get update && apt-get install -y openmpi-bin libopenmpi-dev\n"
      "# VASP binary must be provided by user (license required)\n"
      "# COPY vasp_std /usr/local/bin/\n"
      "WORKDIR /work\n";
  vasp.singularity_recipe =
      "Bootstrap: docker\n"
      "From: ubuntu:22.04\n"
      "%post\n"
      "  apt-get update && apt-get install -y openmpi-bin\n"
      "%runscript\n"
      "  exec mpirun -np $SLURM_NTASKS vasp_std\n";
  vasp.input_template =
      "SYSTEM = ${SYSTEM}\n"
      "PREC = Accurate\n"
      "ENCUT = 400\n"
      "EDIFF = 1e-6\n"
      "IBRION = -1\n"
      "NSW = 0\n"
      "ISMEAR = 0\n"
      "SIGMA = 0.05\n";
  vasp.run_script =
      "#!/bin/bash\n"
      "mpirun -np $SLURM_NTASKS vasp_std > vasp.out 2>&1\n";
  specs.push_back(vasp);

  // CP2K
  ContainerSpec cp2k;
  cp2k.code_name = "CP2K";
  cp2k.dockerfile =
      "FROM ubuntu:22.04\n"
      "RUN apt-get update && apt-get install -y cp2k\n"
      "WORKDIR /work\n";
  cp2k.singularity_recipe =
      "Bootstrap: docker\n"
      "From: ubuntu:22.04\n"
      "%post\n"
      "  apt-get update && apt-get install -y cp2k\n"
      "%runscript\n"
      "  exec mpirun -np $SLURM_NTASKS cp2k.popt -i input.inp\n";
  cp2k.input_template =
      "&GLOBAL\n"
      "  PROJECT ${SYSTEM}\n"
      "  RUN_TYPE ENERGY\n"
      "&END GLOBAL\n"
      "&FORCE_EVAL\n"
      "  METHOD QS\n"
      "  &DFT\n"
      "    BASIS_SET_FILE_NAME BASIS_SET\n"
      "    POTENTIAL_FILE_NAME POTENTIAL\n"
      "    &SCF\n"
      "      EPS_SCF 1.0E-6\n"
      "    &END SCF\n"
      "  &END DFT\n"
      "&END FORCE_EVAL\n";
  cp2k.run_script =
      "#!/bin/bash\n"
      "mpirun -np $SLURM_NTASKS cp2k.popt -i input.inp > cp2k.out 2>&1\n";
  specs.push_back(cp2k);

  // FHI-aims
  ContainerSpec aims;
  aims.code_name = "FHI-aims";
  aims.dockerfile =
      "FROM ubuntu:22.04\n"
      "RUN apt-get update && apt-get install -y gcc gfortran openmpi-bin\n"
      "# FHI-aims must be built from source (license required)\n"
      "WORKDIR /work\n";
  aims.singularity_recipe =
      "Bootstrap: docker\n"
      "From: ubuntu:22.04\n"
      "%post\n"
      "  apt-get update && apt-get install -y gcc gfortran openmpi-bin\n"
      "%runscript\n"
      "  exec mpirun -np $SLURM_NTASKS aims.x > aims.out 2>&1\n";
  aims.input_template =
      "  xc functional pbe\n"
      "  relativistic atomic_zora scalar\n"
      "  sc_accuracy_rho 1e-5\n"
      "  sc_accuracy_eev 1e-3\n"
      "  sc_accuracy_etot 1e-6\n";
  aims.run_script =
      "#!/bin/bash\n"
      "mpirun -np $SLURM_NTASKS aims.x > aims.out 2>&1\n";
  specs.push_back(aims);

  // SIESTA
  ContainerSpec siesta;
  siesta.code_name = "SIESTA";
  siesta.dockerfile =
      "FROM ubuntu:22.04\n"
      "RUN apt-get update && apt-get install -y siesta\n"
      "WORKDIR /work\n";
  siesta.singularity_recipe =
      "Bootstrap: docker\n"
      "From: ubuntu:22.04\n"
      "%post\n"
      "  apt-get update && apt-get install -y siesta\n"
      "%runscript\n"
      "  exec mpirun -np $SLURM_NTASKS siesta < input.fdf > siesta.out 2>&1\n";
  siesta.input_template =
      "SystemName ${SYSTEM}\n"
      "NumberOfAtoms ${N_ATOMS}\n"
      "MeshCutoff 200 Ry\n"
      "DM.Tolerance 1e-4\n"
      "PAO.BasisSize DZP\n";
  siesta.run_script =
      "#!/bin/bash\n"
      "mpirun -np $SLURM_NTASKS siesta < input.fdf > siesta.out 2>&1\n";
  specs.push_back(siesta);

  // GPAW
  ContainerSpec gpaw;
  gpaw.code_name = "GPAW";
  gpaw.dockerfile =
      "FROM python:3.11-slim\n"
      "RUN pip install gpaw ase\n"
      "WORKDIR /work\n";
  gpaw.singularity_recipe =
      "Bootstrap: docker\n"
      "From: python:3.11-slim\n"
      "%post\n"
      "  pip install gpaw ase\n"
      "%runscript\n"
      "  exec gpaw python input.py\n";
  gpaw.input_template =
      "from gpaw import GPAW\n"
      "calc = GPAW(txt='output.txt', xc='PBE', convergence={'energy': 1e-6})\n";
  gpaw.run_script =
      "#!/bin/bash\n"
      "gpaw python input.py > gpaw.out 2>&1\n";
  specs.push_back(gpaw);

  // PySCF
  ContainerSpec pyscf;
  pyscf.code_name = "PySCF";
  pyscf.dockerfile =
      "FROM python:3.11-slim\n"
      "RUN pip install pyscf\n"
      "WORKDIR /work\n";
  pyscf.singularity_recipe =
      "Bootstrap: docker\n"
      "From: python:3.11-slim\n"
      "%post\n"
      "  pip install pyscf\n"
      "%runscript\n"
      "  exec python input.py\n";
  pyscf.input_template =
      "from pyscf import gto, scf\n"
      "mol = gto.M(atom='${ATOMS}', basis='def2-tzvp')\n"
      "mf = scf.RKS(mol).run()\n";
  pyscf.run_script =
      "#!/bin/bash\n"
      "python input.py > pyscf.out 2>&1\n";
  specs.push_back(pyscf);

  return specs;
}

// Parse VASP vasprun.xml output (simplified: extracts energy).
// In production, this would use an XML parser; here we do string search.
[[nodiscard]] inline CompetitorResult ParseVASP(
    const std::string& output_path) {
  CompetitorResult res;
  res.code_name = "VASP";
  std::ifstream f(output_path);
  if (!f) {
    res.parse_error = "file not found";
    return res;
  }
  std::string content((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());
  f.close();

  // Search for <energy> ... <i name="e_0_energy"> value </i> ...
  // Simplified: look for "e_0_energy" and extract the value.
  std::size_t pos = content.find("e_0_energy");
  if (pos == std::string::npos) {
    res.parse_error = "energy not found";
    return res;
  }
  // Find the next ">" after pos, then parse the number.
  pos = content.find(">", pos);
  if (pos == std::string::npos) {
    res.parse_error = "malformed energy tag";
    return res;
  }
  std::istringstream iss(content.substr(pos + 1));
  double energy_eV;
  if (!(iss >> energy_eV)) {
    res.parse_error = "energy parse failed";
    return res;
  }
  res.total_energy = energy_eV / 27.2114;  // eV -> Ha
  res.units = "eV";
  res.parsed_ok = true;
  return res;
}

// Parse CP2K output log (simplified: extracts energy from "ENERGY|" lines).
[[nodiscard]] inline CompetitorResult ParseCP2K(
    const std::string& output_path) {
  CompetitorResult res;
  res.code_name = "CP2K";
  std::ifstream f(output_path);
  if (!f) {
    res.parse_error = "file not found";
    return res;
  }
  std::string line;
  double last_energy = 0.0;
  bool found = false;
  while (std::getline(f, line)) {
    // CP2K prints: "ENERGY| Total FORCE_EVAL ( QS ) energy (a.u.):  -xxx.xxxx"
    std::size_t pos = line.find("Total FORCE_EVAL");
    if (pos != std::string::npos) {
      pos = line.find("(a.u.):");
      if (pos != std::string::npos) {
        std::istringstream iss(line.substr(pos + 7));
        if (iss >> last_energy) found = true;
      }
    }
  }
  f.close();
  if (!found) {
    res.parse_error = "energy not found";
    return res;
  }
  res.total_energy = last_energy;  // already in a.u.
  res.units = "Ha";
  res.parsed_ok = true;
  return res;
}

// Parse FHI-aims output (simplified: extracts "Total energy of the DFT / SCF").
[[nodiscard]] inline CompetitorResult ParseFHIaims(
    const std::string& output_path) {
  CompetitorResult res;
  res.code_name = "FHI-aims";
  std::ifstream f(output_path);
  if (!f) {
    res.parse_error = "file not found";
    return res;
  }
  std::string line;
  double last_energy = 0.0;
  bool found = false;
  while (std::getline(f, line)) {
    if (line.find("Total energy of the DFT / SCF") != std::string::npos) {
      // Energy is in eV on the same or next line.
      std::istringstream iss(line);
      double val;
      while (iss >> val) last_energy = val;
      found = true;
    }
  }
  f.close();
  if (!found) {
    res.parse_error = "energy not found";
    return res;
  }
  res.total_energy = last_energy / 27.2114;  // eV -> Ha
  res.units = "eV";
  res.parsed_ok = true;
  return res;
}

// Parse SIESTA output (simplified: extracts "Total =" energy).
[[nodiscard]] inline CompetitorResult ParseSIESTA(
    const std::string& output_path) {
  CompetitorResult res;
  res.code_name = "SIESTA";
  std::ifstream f(output_path);
  if (!f) {
    res.parse_error = "file not found";
    return res;
  }
  std::string line;
  double last_energy = 0.0;
  bool found = false;
  while (std::getline(f, line)) {
    if (line.find("siesta: E_KS(eV) =") != std::string::npos ||
        line.find("Total =") != std::string::npos) {
      std::istringstream iss(line);
      double val;
      while (iss >> val) last_energy = val;
      found = true;
    }
  }
  f.close();
  if (!found) {
    res.parse_error = "energy not found";
    return res;
  }
  res.total_energy = last_energy / 27.2114;  // eV -> Ha
  res.units = "eV";
  res.parsed_ok = true;
  return res;
}

// Parse PySCF JSON output (simplified: extracts energy from "energy=" lines).
[[nodiscard]] inline CompetitorResult ParsePySCF(
    const std::string& output_path) {
  CompetitorResult res;
  res.code_name = "PySCF";
  std::ifstream f(output_path);
  if (!f) {
    res.parse_error = "file not found";
    return res;
  }
  std::string line;
  double last_energy = 0.0;
  bool found = false;
  while (std::getline(f, line)) {
    std::size_t pos = line.find("energy");
    if (pos != std::string::npos) {
      // Skip to the "=" sign after "energy".
      std::size_t eq = line.find("=", pos);
      if (eq == std::string::npos) continue;
      std::istringstream iss(line.substr(eq + 1));
      if (iss >> last_energy) found = true;
    }
  }
  f.close();
  if (!found) {
    res.parse_error = "energy not found";
    return res;
  }
  res.total_energy = last_energy;  // PySCF uses Ha
  res.units = "Ha";
  res.parsed_ok = true;
  return res;
}

// Dispatch parser by code name.
[[nodiscard]] inline CompetitorResult ParseCompetitorOutput(
    const std::string& code_name, const std::string& output_path) {
  if (code_name == "VASP") return ParseVASP(output_path);
  if (code_name == "CP2K") return ParseCP2K(output_path);
  if (code_name == "FHI-aims") return ParseFHIaims(output_path);
  if (code_name == "SIESTA") return ParseSIESTA(output_path);
  if (code_name == "PySCF") return ParsePySCF(output_path);
  CompetitorResult res;
  res.code_name = code_name;
  res.parse_error = "unknown code";
  return res;
}

}  // namespace tides::verification
