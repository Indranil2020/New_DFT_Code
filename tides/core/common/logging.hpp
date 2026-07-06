#pragma once

// TIDES lightweight logging.
//
// Provides minimal logging macros with level control. No external dependencies.
// Levels: DEBUG < INFO < WARN < ERROR.
// Default level is INFO; can be changed at runtime via SetLogLevel().

#include <cstdio>
#include <string>
#include <string_view>

namespace tides::logging {

enum class LogLevel {
  kDebug = 0,
  kInfo = 1,
  kWarn = 2,
  kError = 3,
  kQuiet = 4,
};

inline LogLevel& CurrentLogLevel() {
  static LogLevel level = LogLevel::kInfo;
  return level;
}

inline void SetLogLevel(LogLevel level) { CurrentLogLevel() = level; }

inline const char* LevelLabel(LogLevel level) {
  switch (level) {
    case LogLevel::kDebug: return "DEBUG";
    case LogLevel::kInfo:  return "INFO";
    case LogLevel::kWarn:  return "WARN";
    case LogLevel::kError: return "ERROR";
    default:               return "QUIET";
  }
}

inline void Log(LogLevel level, std::string_view msg) {
  if (static_cast<int>(level) < static_cast<int>(CurrentLogLevel())) return;
  std::fprintf(stderr, "[TIDES %s] %.*s\n",
               LevelLabel(level),
               static_cast<int>(msg.size()), msg.data());
}

}  // namespace tides::logging

#define TIDES_LOG_DEBUG(msg) \
  ::tides::logging::Log(::tides::logging::LogLevel::kDebug, msg)
#define TIDES_LOG_INFO(msg)  \
  ::tides::logging::Log(::tides::logging::LogLevel::kInfo, msg)
#define TIDES_LOG_WARN(msg)  \
  ::tides::logging::Log(::tides::logging::LogLevel::kWarn, msg)
#define TIDES_LOG_ERROR(msg) \
  ::tides::logging::Log(::tides::logging::LogLevel::kError, msg)
