#pragma once

#include <cassert>
#include <optional>
#include <string>
#include <utility>

namespace tides {

enum class StatusCode {
  kOk = 0,
  kInvalidArgument,
  kOutOfRange,
  kIoError,
  kCorruptData,
  kUnimplemented,
};

class Status {
 public:
  Status() = default;

  Status(StatusCode code, std::string message)
      : code_(code), message_(std::move(message)) {}

  [[nodiscard]] static Status Ok() { return {}; }

  [[nodiscard]] static Status InvalidArgument(std::string message) {
    return {StatusCode::kInvalidArgument, std::move(message)};
  }

  [[nodiscard]] static Status OutOfRange(std::string message) {
    return {StatusCode::kOutOfRange, std::move(message)};
  }

  [[nodiscard]] static Status IoError(std::string message) {
    return {StatusCode::kIoError, std::move(message)};
  }

  [[nodiscard]] static Status CorruptData(std::string message) {
    return {StatusCode::kCorruptData, std::move(message)};
  }

  [[nodiscard]] static Status Unimplemented(std::string message) {
    return {StatusCode::kUnimplemented, std::move(message)};
  }

  [[nodiscard]] bool ok() const { return code_ == StatusCode::kOk; }
  [[nodiscard]] StatusCode code() const { return code_; }
  [[nodiscard]] const std::string& message() const { return message_; }

 private:
  StatusCode code_ = StatusCode::kOk;
  std::string message_;
};

template <typename T>
class Result {
 public:
  Result(T value) : status_(Status::Ok()), value_(std::move(value)) {}

  Result(Status status) : status_(std::move(status)) {
    assert(!status_.ok());
  }

  [[nodiscard]] bool ok() const { return status_.ok(); }
  [[nodiscard]] const Status& status() const { return status_; }

  [[nodiscard]] const T& value() const {
    assert(ok());
    return *value_;
  }

  [[nodiscard]] T& value() {
    assert(ok());
    return *value_;
  }

  [[nodiscard]] T take_value() {
    assert(ok());
    return std::move(*value_);
  }

 private:
  Status status_;
  std::optional<T> value_;
};

}  // namespace tides
