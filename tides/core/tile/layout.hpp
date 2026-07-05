#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <istream>
#include <limits>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "common/status.hpp"

namespace tides::tile {

enum class DType : std::uint32_t {
  kFloat64 = 1,
};

enum class ScaleMode : std::uint32_t {
  kNone = 0,
  kMaxAbs = 1,
};

enum class Symmetry : std::uint32_t {
  kGeneral = 0,
  kSymmetric = 1,
};

struct TileView {
  std::size_t ordinal = 0;
  std::size_t block_row = 0;
  std::size_t block_col = 0;
  std::size_t row_start = 0;
  std::size_t col_start = 0;
  std::size_t row_extent = 0;
  std::size_t col_extent = 0;
  std::uint32_t edge = 0;
  double scale = 0.0;
  const double* values = nullptr;

  [[nodiscard]] double value(std::size_t local_row,
                             std::size_t local_col) const {
    assert(local_row < edge);
    assert(local_col < edge);
    return values[local_row * edge + local_col];
  }
};

class TileMat {
 public:
  static constexpr std::uint32_t kSchemaVersion = 1;

  class ConstIterator {
   public:
    using difference_type = std::ptrdiff_t;
    using value_type = TileView;
    using iterator_category = std::forward_iterator_tag;

    ConstIterator(const TileMat* matrix, std::size_t ordinal)
        : matrix_(matrix), ordinal_(ordinal) {}

    [[nodiscard]] TileView operator*() const {
      assert(matrix_ != nullptr);
      return matrix_->tile(ordinal_);
    }

    ConstIterator& operator++() {
      ++ordinal_;
      return *this;
    }

    [[nodiscard]] bool operator==(const ConstIterator& other) const {
      return matrix_ == other.matrix_ && ordinal_ == other.ordinal_;
    }

    [[nodiscard]] bool operator!=(const ConstIterator& other) const {
      return !(*this == other);
    }

   private:
    const TileMat* matrix_ = nullptr;
    std::size_t ordinal_ = 0;
  };

  TileMat() = default;

  [[nodiscard]] static bool IsValidTileEdge(std::uint32_t edge) {
    return edge == 16 || edge == 32 || edge == 64;
  }

  [[nodiscard]] static Result<TileMat> FromDense(
      std::size_t rows, std::size_t cols, const std::vector<double>& dense,
      std::uint32_t tile_edge, Symmetry symmetry = Symmetry::kGeneral,
      ScaleMode scale_mode = ScaleMode::kMaxAbs) {
    if (!IsValidTileEdge(tile_edge)) {
      return Status::InvalidArgument("tile edge must be one of 16, 32, 64");
    }
    if (dense.size() != rows * cols) {
      return Status::InvalidArgument("dense payload size does not match shape");
    }
    if (symmetry == Symmetry::kSymmetric) {
      if (rows != cols) {
        return Status::InvalidArgument("symmetric TileMat must be square");
      }
      for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = i + 1; j < cols; ++j) {
          if (dense[i * cols + j] != dense[j * cols + i]) {
            return Status::InvalidArgument(
                "symmetric TileMat input is not exactly symmetric");
          }
        }
      }
    }

    TileMat out;
    out.rows_ = rows;
    out.cols_ = cols;
    out.tile_edge_ = tile_edge;
    out.block_rows_ = CeilDiv(rows, tile_edge);
    out.block_cols_ = CeilDiv(cols, tile_edge);
    out.symmetry_ = symmetry;
    out.scale_mode_ = scale_mode;
    out.row_ptr_.assign(out.block_rows_ + 1, 0);

    const std::size_t tile_cells =
        static_cast<std::size_t>(tile_edge) * tile_edge;
    for (std::size_t br = 0; br < out.block_rows_; ++br) {
      out.row_ptr_[br] = out.col_ind_.size();
      for (std::size_t bc = 0; bc < out.block_cols_; ++bc) {
        if (symmetry == Symmetry::kSymmetric && bc < br) {
          continue;
        }

        const std::size_t row0 = br * tile_edge;
        const std::size_t col0 = bc * tile_edge;
        const std::size_t row_extent = std::min<std::size_t>(tile_edge, rows - row0);
        const std::size_t col_extent = std::min<std::size_t>(tile_edge, cols - col0);

        bool has_nonzero = false;
        double max_abs = 0.0;
        for (std::size_t i = 0; i < row_extent; ++i) {
          for (std::size_t j = 0; j < col_extent; ++j) {
            const double v = dense[(row0 + i) * cols + (col0 + j)];
            has_nonzero = has_nonzero || v != 0.0;
            max_abs = std::max(max_abs, std::abs(v));
          }
        }
        if (!has_nonzero) {
          continue;
        }

        out.col_ind_.push_back(static_cast<std::uint32_t>(bc));
        out.scales_.push_back(scale_mode == ScaleMode::kMaxAbs ? max_abs : 1.0);
        const std::size_t offset = out.values_.size();
        out.values_.resize(offset + tile_cells, 0.0);
        for (std::size_t i = 0; i < row_extent; ++i) {
          for (std::size_t j = 0; j < col_extent; ++j) {
            out.values_[offset + i * tile_edge + j] =
                dense[(row0 + i) * cols + (col0 + j)];
          }
        }
      }
      out.row_ptr_[br + 1] = out.col_ind_.size();
    }

    return out;
  }

  [[nodiscard]] std::size_t rows() const { return rows_; }
  [[nodiscard]] std::size_t cols() const { return cols_; }
  [[nodiscard]] std::uint32_t tile_edge() const { return tile_edge_; }
  [[nodiscard]] std::size_t block_rows() const { return block_rows_; }
  [[nodiscard]] std::size_t block_cols() const { return block_cols_; }
  [[nodiscard]] std::size_t tile_count() const { return col_ind_.size(); }
  [[nodiscard]] DType dtype() const { return dtype_; }
  [[nodiscard]] ScaleMode scale_mode() const { return scale_mode_; }
  [[nodiscard]] Symmetry symmetry() const { return symmetry_; }
  [[nodiscard]] const std::vector<std::size_t>& row_ptr() const {
    return row_ptr_;
  }
  [[nodiscard]] const std::vector<std::uint32_t>& col_ind() const {
    return col_ind_;
  }
  [[nodiscard]] const std::vector<double>& scales() const { return scales_; }
  [[nodiscard]] const std::vector<double>& raw_values() const { return values_; }

  [[nodiscard]] ConstIterator begin() const { return {this, 0}; }
  [[nodiscard]] ConstIterator end() const { return {this, tile_count()}; }

  [[nodiscard]] TileView tile(std::size_t ordinal) const {
    assert(ordinal < tile_count());
    const auto row_it =
        std::upper_bound(row_ptr_.begin(), row_ptr_.end(), ordinal);
    assert(row_it != row_ptr_.begin());
    const std::size_t br =
        static_cast<std::size_t>(std::distance(row_ptr_.begin(), row_it) - 1);
    const std::size_t bc = col_ind_[ordinal];
    const std::size_t row0 = br * tile_edge_;
    const std::size_t col0 = bc * tile_edge_;
    return TileView{
        ordinal,
        br,
        bc,
        row0,
        col0,
        std::min<std::size_t>(tile_edge_, rows_ - row0),
        std::min<std::size_t>(tile_edge_, cols_ - col0),
        tile_edge_,
        scales_[ordinal],
        values_.data() + ordinal * static_cast<std::size_t>(tile_edge_) *
                             tile_edge_,
    };
  }

  [[nodiscard]] std::vector<double> ToDense() const {
    std::vector<double> dense(rows_ * cols_, 0.0);
    for (const TileView view : *this) {
      for (std::size_t i = 0; i < view.row_extent; ++i) {
        for (std::size_t j = 0; j < view.col_extent; ++j) {
          const double v = view.value(i, j);
          const std::size_t r = view.row_start + i;
          const std::size_t c = view.col_start + j;
          dense[r * cols_ + c] = v;
          if (symmetry_ == Symmetry::kSymmetric && r != c) {
            dense[c * cols_ + r] = v;
          }
        }
      }
    }
    return dense;
  }

  [[nodiscard]] double TraceFp64() const {
    const std::size_t n = std::min(rows_, cols_);
    double trace = 0.0;
    for (const TileView view : *this) {
      for (std::size_t i = 0; i < view.row_extent; ++i) {
        const std::size_t r = view.row_start + i;
        if (r >= n || r < view.col_start ||
            r >= view.col_start + view.col_extent) {
          continue;
        }
        trace += view.value(i, r - view.col_start);
      }
    }
    return trace;
  }

  [[nodiscard]] double FrobeniusNormFp64() const {
    long double sum = 0.0;
    for (const double v : ToDense()) {
      sum += static_cast<long double>(v) * static_cast<long double>(v);
    }
    return std::sqrt(static_cast<double>(sum));
  }

  [[nodiscard]] bool SameSchemaAndPayload(const TileMat& other) const {
    return rows_ == other.rows_ && cols_ == other.cols_ &&
           tile_edge_ == other.tile_edge_ && block_rows_ == other.block_rows_ &&
           block_cols_ == other.block_cols_ && dtype_ == other.dtype_ &&
           scale_mode_ == other.scale_mode_ && symmetry_ == other.symmetry_ &&
           row_ptr_ == other.row_ptr_ && col_ind_ == other.col_ind_ &&
           scales_ == other.scales_ && values_ == other.values_;
  }

  [[nodiscard]] Status SerializeBinary(std::ostream& os) const {
    constexpr std::array<char, 8> kMagic = {'T', 'I', 'D', 'E',
                                            'T', 'M', '1', '\0'};
    WriteBytes(os, kMagic.data(), kMagic.size());
    WriteU32(os, kSchemaVersion);
    WriteU64(os, rows_);
    WriteU64(os, cols_);
    WriteU32(os, tile_edge_);
    WriteU64(os, block_rows_);
    WriteU64(os, block_cols_);
    WriteU32(os, static_cast<std::uint32_t>(dtype_));
    WriteU32(os, static_cast<std::uint32_t>(scale_mode_));
    WriteU32(os, static_cast<std::uint32_t>(symmetry_));
    WriteU64(os, row_ptr_.size());
    WriteU64(os, col_ind_.size());
    WriteU64(os, scales_.size());
    WriteU64(os, values_.size());
    for (const std::size_t v : row_ptr_) {
      WriteU64(os, v);
    }
    for (const std::uint32_t v : col_ind_) {
      WriteU32(os, v);
    }
    for (const double v : scales_) {
      WriteF64(os, v);
    }
    for (const double v : values_) {
      WriteF64(os, v);
    }
    if (!os.good()) {
      return Status::IoError("failed while writing TileMat binary stream");
    }
    return Status::Ok();
  }

  [[nodiscard]] static Result<TileMat> DeserializeBinary(std::istream& is) {
    constexpr std::array<char, 8> kMagic = {'T', 'I', 'D', 'E',
                                            'T', 'M', '1', '\0'};
    std::array<char, 8> magic{};
    if (!ReadBytes(is, magic.data(), magic.size()) || magic != kMagic) {
      return Status::CorruptData("TileMat binary stream has bad magic");
    }

    TileMat out;
    std::uint32_t version = 0;
    std::uint32_t dtype = 0;
    std::uint32_t scale_mode = 0;
    std::uint32_t symmetry = 0;
    std::uint64_t row_ptr_size = 0;
    std::uint64_t col_ind_size = 0;
    std::uint64_t scales_size = 0;
    std::uint64_t values_size = 0;

    if (!ReadU32(is, &version) || version != kSchemaVersion ||
        !ReadU64(is, &out.rows_) || !ReadU64(is, &out.cols_) ||
        !ReadU32(is, &out.tile_edge_) || !ReadU64(is, &out.block_rows_) ||
        !ReadU64(is, &out.block_cols_) || !ReadU32(is, &dtype) ||
        !ReadU32(is, &scale_mode) || !ReadU32(is, &symmetry) ||
        !ReadU64(is, &row_ptr_size) || !ReadU64(is, &col_ind_size) ||
        !ReadU64(is, &scales_size) || !ReadU64(is, &values_size)) {
      return Status::CorruptData("TileMat binary header is truncated");
    }

    out.dtype_ = static_cast<DType>(dtype);
    out.scale_mode_ = static_cast<ScaleMode>(scale_mode);
    out.symmetry_ = static_cast<Symmetry>(symmetry);
    out.row_ptr_.resize(static_cast<std::size_t>(row_ptr_size));
    out.col_ind_.resize(static_cast<std::size_t>(col_ind_size));
    out.scales_.resize(static_cast<std::size_t>(scales_size));
    out.values_.resize(static_cast<std::size_t>(values_size));

    for (std::size_t& v : out.row_ptr_) {
      if (!ReadSize(is, &v)) {
        return Status::CorruptData("TileMat row_ptr payload is truncated");
      }
    }
    for (std::uint32_t& v : out.col_ind_) {
      if (!ReadU32(is, &v)) {
        return Status::CorruptData("TileMat col_ind payload is truncated");
      }
    }
    for (double& v : out.scales_) {
      if (!ReadF64(is, &v)) {
        return Status::CorruptData("TileMat scales payload is truncated");
      }
    }
    for (double& v : out.values_) {
      if (!ReadF64(is, &v)) {
        return Status::CorruptData("TileMat values payload is truncated");
      }
    }

    const Status invariant_status = out.ValidateInvariants();
    if (!invariant_status.ok()) {
      return invariant_status;
    }
    return out;
  }

  [[nodiscard]] Status SaveBinary(const std::string& path) const {
    std::ofstream os(path, std::ios::binary);
    if (!os.good()) {
      return Status::IoError("could not open TileMat binary file for writing");
    }
    return SerializeBinary(os);
  }

  [[nodiscard]] static Result<TileMat> LoadBinary(const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is.good()) {
      return Status::IoError("could not open TileMat binary file for reading");
    }
    return DeserializeBinary(is);
  }

  [[nodiscard]] Status ValidateInvariants() const {
    if (!IsValidTileEdge(tile_edge_)) {
      return Status::CorruptData("invalid tile edge");
    }
    if (block_rows_ != CeilDiv(rows_, tile_edge_) ||
        block_cols_ != CeilDiv(cols_, tile_edge_)) {
      return Status::CorruptData("block shape does not match matrix shape");
    }
    if (dtype_ != DType::kFloat64) {
      return Status::CorruptData("unsupported TileMat dtype");
    }
    if (scale_mode_ != ScaleMode::kNone && scale_mode_ != ScaleMode::kMaxAbs) {
      return Status::CorruptData("unsupported TileMat scale mode");
    }
    if (symmetry_ != Symmetry::kGeneral && symmetry_ != Symmetry::kSymmetric) {
      return Status::CorruptData("unsupported TileMat symmetry flag");
    }
    if (symmetry_ == Symmetry::kSymmetric && rows_ != cols_) {
      return Status::CorruptData("symmetric TileMat is not square");
    }
    if (row_ptr_.size() != block_rows_ + 1 || row_ptr_.empty()) {
      return Status::CorruptData("row_ptr has invalid length");
    }
    if (row_ptr_.front() != 0 || row_ptr_.back() != col_ind_.size()) {
      return Status::CorruptData("row_ptr boundary values are invalid");
    }
    for (std::size_t i = 1; i < row_ptr_.size(); ++i) {
      if (row_ptr_[i] < row_ptr_[i - 1]) {
        return Status::CorruptData("row_ptr is not monotone");
      }
    }
    if (scales_.size() != col_ind_.size()) {
      return Status::CorruptData("scale count does not match tile count");
    }
    const std::size_t tile_cells =
        static_cast<std::size_t>(tile_edge_) * tile_edge_;
    if (values_.size() != col_ind_.size() * tile_cells) {
      return Status::CorruptData("value payload size does not match tile count");
    }
    for (std::size_t br = 0; br < block_rows_; ++br) {
      std::uint32_t previous_col = 0;
      bool have_previous = false;
      for (std::size_t k = row_ptr_[br]; k < row_ptr_[br + 1]; ++k) {
        if (col_ind_[k] >= block_cols_) {
          return Status::CorruptData("tile column index is out of range");
        }
        if (symmetry_ == Symmetry::kSymmetric && col_ind_[k] < br) {
          return Status::CorruptData("symmetric TileMat stores lower triangle");
        }
        if (have_previous && col_ind_[k] <= previous_col) {
          return Status::CorruptData("tile columns are not strictly sorted");
        }
        previous_col = col_ind_[k];
        have_previous = true;
      }
    }
    return Status::Ok();
  }

 private:
  [[nodiscard]] static std::size_t CeilDiv(std::size_t value,
                                           std::size_t divisor) {
    return value == 0 ? 0 : 1 + (value - 1) / divisor;
  }

  static void WriteBytes(std::ostream& os, const char* data, std::size_t size) {
    os.write(data, static_cast<std::streamsize>(size));
  }

  static bool ReadBytes(std::istream& is, char* data, std::size_t size) {
    is.read(data, static_cast<std::streamsize>(size));
    return is.good();
  }

  static void WriteU32(std::ostream& os, std::uint32_t value) {
    os.write(reinterpret_cast<const char*>(&value), sizeof(value));
  }

  static void WriteU64(std::ostream& os, std::uint64_t value) {
    os.write(reinterpret_cast<const char*>(&value), sizeof(value));
  }

  static void WriteF64(std::ostream& os, double value) {
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    WriteU64(os, bits);
  }

  static bool ReadU32(std::istream& is, std::uint32_t* value) {
    is.read(reinterpret_cast<char*>(value), sizeof(*value));
    return is.good();
  }

  static bool ReadU64(std::istream& is, std::uint64_t* value) {
    is.read(reinterpret_cast<char*>(value), sizeof(*value));
    return is.good();
  }

  static bool ReadSize(std::istream& is, std::size_t* value) {
    std::uint64_t tmp = 0;
    if (!ReadU64(is, &tmp) ||
        tmp > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
      return false;
    }
    *value = static_cast<std::size_t>(tmp);
    return true;
  }

  static bool ReadF64(std::istream& is, double* value) {
    std::uint64_t bits = 0;
    if (!ReadU64(is, &bits)) {
      return false;
    }
    static_assert(sizeof(bits) == sizeof(*value));
    std::memcpy(value, &bits, sizeof(bits));
    return true;
  }

  std::size_t rows_ = 0;
  std::size_t cols_ = 0;
  std::uint32_t tile_edge_ = 16;
  std::size_t block_rows_ = 0;
  std::size_t block_cols_ = 0;
  DType dtype_ = DType::kFloat64;
  ScaleMode scale_mode_ = ScaleMode::kMaxAbs;
  Symmetry symmetry_ = Symmetry::kGeneral;
  std::vector<std::size_t> row_ptr_;
  std::vector<std::uint32_t> col_ind_;
  std::vector<double> scales_;
  std::vector<double> values_;
};

}  // namespace tides::tile
