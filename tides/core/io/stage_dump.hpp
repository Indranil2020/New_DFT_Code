#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "common/status.hpp"

namespace tides::io {

// HDF5 stage-dump / restart (T8.4) — the bisect-the-physics enabler.
//
// Per 31-data-contracts: "Every stage dumpable AND injectable; bitwise
// round-trip test; version attribute mandatory."
// Stages: geometry | S | H0 | rho | vH | vxc | H | P | E_components | forces | stress.
//
// Since HDF5 C++ headers are not installed on this machine, we use a simple
// binary format (magic + version + named stages) that achieves the same
// contract: bitwise round-trip, versioned schema, injection (swap any stage
// for a reference dump and reproduce). A production HDF5 writer is a
// straightforward replacement (same API, HDF5 backend).
//
// Observable (T8.4): bitwise round-trip; schema versioned; injection demo.

struct StageData {
  std::string name;           // e.g. "rho", "vH", "H"
  std::vector<double> data;   // the stage's data (flattened)
  std::vector<std::size_t> shape;  // dimensions (e.g. {n, n} for matrices)
  std::string description;    // metadata
};

class StageDump {
 public:
  static constexpr std::uint32_t kMagic = 0x54494445;  // "TIDE"
  static constexpr std::uint32_t kVersion = 1;

  // Write all stages to a binary file.
  static tides::Status Write(const std::string& path,
                             const std::vector<StageData>& stages) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return tides::Status::IoError("cannot open " + path + " for writing");

    // Header: magic, version, n_stages.
    f.write(reinterpret_cast<const char*>(&kMagic), sizeof(kMagic));
    f.write(reinterpret_cast<const char*>(&kVersion), sizeof(kVersion));
    const std::uint32_t n = static_cast<std::uint32_t>(stages.size());
    f.write(reinterpret_cast<const char*>(&n), sizeof(n));

    for (const auto& s : stages) {
      // Stage name (length-prefixed).
      const std::uint32_t name_len = static_cast<std::uint32_t>(s.name.size());
      f.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
      f.write(s.name.data(), name_len);

      // Description.
      const std::uint32_t desc_len = static_cast<std::uint32_t>(s.description.size());
      f.write(reinterpret_cast<const char*>(&desc_len), sizeof(desc_len));
      f.write(s.description.data(), desc_len);

      // Shape.
      const std::uint32_t shape_len = static_cast<std::uint32_t>(s.shape.size());
      f.write(reinterpret_cast<const char*>(&shape_len), sizeof(shape_len));
      for (auto dim : s.shape) {
        const std::uint64_t d = static_cast<std::uint64_t>(dim);
        f.write(reinterpret_cast<const char*>(&d), sizeof(d));
      }

      // Data.
      const std::uint64_t data_len = static_cast<std::uint64_t>(s.data.size());
      f.write(reinterpret_cast<const char*>(&data_len), sizeof(data_len));
      f.write(reinterpret_cast<const char*>(s.data.data()),
              data_len * sizeof(double));
    }
    f.close();
    return tides::Status::Ok();
  }

  // Read all stages from a binary file.
  static tides::Result<std::vector<StageData>> Read(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return tides::Status::IoError("cannot open " + path + " for reading");

    std::uint32_t magic, version, n;
    f.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    f.read(reinterpret_cast<char*>(&version), sizeof(version));
    f.read(reinterpret_cast<char*>(&n), sizeof(n));
    if (magic != kMagic)
      return tides::Status::CorruptData("bad magic number");
    if (version != kVersion)
      return tides::Status::CorruptData("unsupported version " +
                                        std::to_string(version));

    std::vector<StageData> stages;
    stages.reserve(n);
    for (std::uint32_t i = 0; i < n; ++i) {
      StageData s;
      std::uint32_t name_len;
      f.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
      s.name.resize(name_len);
      f.read(s.name.data(), name_len);

      std::uint32_t desc_len;
      f.read(reinterpret_cast<char*>(&desc_len), sizeof(desc_len));
      s.description.resize(desc_len);
      f.read(s.description.data(), desc_len);

      std::uint32_t shape_len;
      f.read(reinterpret_cast<char*>(&shape_len), sizeof(shape_len));
      s.shape.resize(shape_len);
      for (std::uint32_t d = 0; d < shape_len; ++d) {
        std::uint64_t dim;
        f.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        s.shape[d] = static_cast<std::size_t>(dim);
      }

      std::uint64_t data_len;
      f.read(reinterpret_cast<char*>(&data_len), sizeof(data_len));
      s.data.resize(data_len);
      f.read(reinterpret_cast<char*>(s.data.data()), data_len * sizeof(double));

      stages.push_back(std::move(s));
    }
    f.close();
    return stages;
  }

  // Verify bitwise round-trip: write stages, read them back, compare.
  static bool VerifyRoundTrip(const std::vector<StageData>& stages) {
    const std::string path = "/tmp/tides_stage_dump_test.bin";
    auto s = Write(path, stages);
    if (!s.ok()) return false;
    auto result = Read(path);
    if (!result.ok()) return false;
    const auto& read_stages = result.value();
    if (read_stages.size() != stages.size()) return false;
    for (std::size_t i = 0; i < stages.size(); ++i) {
      if (read_stages[i].name != stages[i].name) return false;
      if (read_stages[i].data != stages[i].data) return false;
      if (read_stages[i].shape != stages[i].shape) return false;
    }
    return true;
  }

  // Injection: replace one stage with a reference and reproduce.
  // This is the "bisect-the-physics" enabler: swap our Poisson stage for a
  // reference dump and check the downstream energies match.
  static std::vector<StageData> InjectStage(
      std::vector<StageData> stages, const std::string& stage_name,
      const StageData& replacement) {
    for (auto& s : stages) {
      if (s.name == stage_name) {
        s = replacement;
        break;
      }
    }
    return stages;
  }
};

}  // namespace tides::io
