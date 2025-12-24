#include "logger_options.hpp"
#include <algorithm>
#include <boost/program_options.hpp>
#include <cctype>
#include <sstream>

namespace specfem {
namespace logger {

LogLevel LoggerOptions::string_to_log_level(const std::string &level_str) {
  std::string upper_str = level_str;
  std::transform(upper_str.begin(), upper_str.end(), upper_str.begin(),
                 [](unsigned char c) { return std::toupper(c); });

  if (upper_str == "TRACE")
    return LogLevel::TRACE;
  if (upper_str == "DEBUG")
    return LogLevel::DEBUG;
  if (upper_str == "INFO")
    return LogLevel::INFO;
  if (upper_str == "WARNING" || upper_str == "WARN")
    return LogLevel::WARNING;
  if (upper_str == "ERROR")
    return LogLevel::ERROR;
  if (upper_str == "CRITICAL" || upper_str == "CRIT")
    return LogLevel::CRITICAL;

  throw std::invalid_argument(
      "Invalid log level: " + level_str +
      ". Valid values: TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL");
}

LoggerOptions LoggerOptions::from_variables_map(
    const boost::program_options::variables_map &vm) {
  LoggerOptions options;

  // Extract optional values from variables_map
  if (vm.count("log-file")) {
    options.log_file = vm["log-file"].as<std::string>();
  }

  if (vm.count("log-per-rank")) {
    options.per_rank = vm["log-per-rank"].as<bool>();
  }

  if (vm.count("log-auto-flush")) {
    options.auto_flush = vm["log-auto-flush"].as<bool>();
  }

  if (vm.count("log-level")) {
    std::string level_str = vm["log-level"].as<std::string>();
    options.log_level = string_to_log_level(level_str);
  }

  return options;
}

} // namespace logger
} // namespace specfem
