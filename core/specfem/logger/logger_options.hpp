#pragma once

#include "specfem/logger/logger.hpp"
#include <boost/program_options.hpp>
#include <optional>
#include <string>

namespace specfem {
namespace logger {

/**
 * @class LoggerOptions
 * @brief Command-line options for Logger configuration
 *
 * This class parses command-line arguments using boost::program_options
 * to allow runtime override of Logger settings.
 *
 * Supported options:
 * - --log-file=<filename>        : Set output log file
 * - --log-per-rank=<true|false>  : Enable per-rank log files and stdout
 * - --log-auto-flush=<true|false>: Enable auto-flush after each message
 * - --log-level=<level>          : Set minimum log level
 */
class LoggerOptions {
public:
  /**
   * @brief Create logger options from an existing variables_map
   *
   * @param vm Variables map containing parsed command-line options
   * @return LoggerOptions instance with extracted values
   */
  static LoggerOptions
  from_variables_map(const boost::program_options::variables_map &vm);

  // Optional values - only set if provided on command line
  std::optional<std::string> log_file; ///< Log file path
  std::optional<bool> per_rank;        ///< Per-rank file creation
  std::optional<bool> auto_flush;      ///< Auto-flush after each message
  std::optional<LogLevel> log_level;   ///< Minimum log level

private:
  LoggerOptions() = default;

  /**
   * @brief Convert string to LogLevel
   */
  static LogLevel string_to_log_level(const std::string &level_str);
};

} // namespace logger
} // namespace specfem
