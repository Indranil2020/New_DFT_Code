#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

namespace tides::atomgen {

// Logarithmic radial grid r = exp(x), x uniform on [x_min, x_max] with spacing
// h. Points cluster near the nucleus (r -> 0) where atomic wavefunctions vary
// rapidly, giving high accuracy at modest N (the standard choice for atomic
// structure). All quantities in Hartree atomic units (hbar = m_e = e = 1).
class LogGrid {
 public:
  LogGrid(double x_min, double x_max, std::size_t n)
      : x_min_(x_min), x_max_(x_max), n_(n) {
    h_ = (n_ > 1) ? (x_max_ - x_min_) / static_cast<double>(n_ - 1) : 0.0;
    r_.resize(n_);
    x_.resize(n_);
    for (std::size_t i = 0; i < n_; ++i) {
      x_[i] = x_min_ + h_ * static_cast<double>(i);
      r_[i] = std::exp(x_[i]);
    }
  }

  [[nodiscard]] std::size_t size() const { return n_; }
  [[nodiscard]] double h() const { return h_; }
  [[nodiscard]] double x_min() const { return x_min_; }
  [[nodiscard]] double x_max() const { return x_max_; }
  [[nodiscard]] const std::vector<double>& r() const { return r_; }
  [[nodiscard]] const std::vector<double>& x() const { return x_; }
  [[nodiscard]] double r_at(std::size_t i) const { return r_[i]; }
  [[nodiscard]] double x_at(std::size_t i) const { return x_[i]; }

 private:
  double x_min_ = 0.0;
  double x_max_ = 0.0;
  double h_ = 0.0;
  std::size_t n_ = 0;
  std::vector<double> r_;
  std::vector<double> x_;
};

}  // namespace tides::atomgen
