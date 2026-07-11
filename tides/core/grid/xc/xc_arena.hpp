#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "common/status.hpp"
#include "grid/xc/xc_engine.hpp"

namespace tides::grid::xc {

// Owns XC buffers for one SCF scope. Reserve() is a setup operation; a stable
// shape makes it a no-op, ensuring the SCF iteration path has no allocation.
class XcArena {
 public:
  XcArena();
  XcArena(const XcArena&) = delete;
  XcArena& operator=(const XcArena&) = delete;
  XcArena(XcArena&&) noexcept;
  XcArena& operator=(XcArena&&) noexcept;
  ~XcArena();

  [[nodiscard]] Status Reserve(std::size_t np, int nspin, bool need_grad,
                               bool need_tau, int nsys, cudaStream_t stream);
  [[nodiscard]] Status Release(cudaStream_t stream);

  [[nodiscard]] std::size_t capacity() const;
  [[nodiscard]] double* rho() const;
  [[nodiscard]] double* weights() const;
  [[nodiscard]] double* grad() const;
  [[nodiscard]] double* tau() const;
  [[nodiscard]] double* wv_rho() const;
  [[nodiscard]] double* wv_grad() const;
  [[nodiscard]] double* wv_tau() const;
  [[nodiscard]] double* exc_per_system() const;
  [[nodiscard]] std::int64_t* sys_offsets() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace tides::grid::xc
