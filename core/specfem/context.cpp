#include "specfem/context.hpp"
#include "specfem/execute.hpp"
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace specfem {

Context::Context(int argc, char *argv[])
    : kokkos_guard_(argc, argv),
      mpi_(std::make_unique<specfem::MPI::MPI>(&argc, &argv)) {}

Context::~Context() = default;

template <specfem::dimension::type DimensionTag>
bool Context::execute(
    const YAML::Node &parameter_dict, const YAML::Node &default_dict,
    std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task> >
        &tasks) {
  try {
    // Call the appropriate templated execute function
    if constexpr (DimensionTag == specfem::dimension::type::dim2) {
      ::execute(parameter_dict, default_dict, tasks, mpi_.get());
    } else if constexpr (DimensionTag == specfem::dimension::type::dim3) {
      throw std::runtime_error("3D simulations are not yet enabled.");
    } else {
      std::cerr << "Unsupported dimension type" << std::endl;
      return false;
    }
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Error during execution: " << e.what() << std::endl;
    return false;
  }
}

bool Context::execute_with_dimension(
    const std::string &dimension, const YAML::Node &parameter_dict,
    const YAML::Node &default_dict,
    std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task> >
        &tasks) {
  try {
    // Use execution model enumeration for validation
    specfem::execution::model exec_model =
        specfem::execution::from_string(dimension);

    switch (exec_model) {
    case specfem::execution::model::dim2:
      return execute<specfem::dimension::type::dim2>(parameter_dict,
                                                     default_dict, tasks);
    case specfem::execution::model::dim3:
      return execute<specfem::dimension::type::dim3>(parameter_dict,
                                                     default_dict, tasks);
    default:
      std::cerr << "Unsupported execution model" << std::endl;
      return false;
    }
  } catch (const std::invalid_argument &e) {
    std::cerr << e.what() << std::endl;
    return false;
  }
}

specfem::MPI::MPI *Context::get_mpi() const { return mpi_.get(); }

// Explicit template instantiations
template bool Context::execute<specfem::dimension::type::dim2>(
    const YAML::Node &, const YAML::Node &,
    std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task> > &);

template bool Context::execute<specfem::dimension::type::dim3>(
    const YAML::Node &, const YAML::Node &,
    std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task> > &);

// ============================================================================
// ContextGuard Implementation
// ============================================================================

ContextGuard::ContextGuard(int argc, char *argv[])
    : context_(std::make_unique<Context>(argc, argv)) {}

ContextGuard::ContextGuard(const std::vector<std::string> &args) {
  int argc;
  char **argv;
  setup_argc_argv(args, argc, argv);

  try {
    context_ = std::make_unique<Context>(argc, argv);
  } catch (...) {
    cleanup_argc_argv(argc, argv);
    throw;
  }

  cleanup_argc_argv(argc, argv);
}

ContextGuard::~ContextGuard() = default;

Context &ContextGuard::get_context() { return *context_; }

specfem::MPI::MPI *ContextGuard::get_mpi() const { return context_->get_mpi(); }

void ContextGuard::setup_argc_argv(const std::vector<std::string> &args,
                                   int &argc, char **&argv) {
  argc = args.size();
  argv = new char *[argc + 1];

  for (size_t i = 0; i < args.size(); ++i) {
    const std::string &str = args[i];
    argv[i] = new char[str.length() + 1];
    std::strcpy(argv[i], str.c_str());
  }

  // Null-terminate argv following the specification
  argv[argc] = nullptr;
}

void ContextGuard::cleanup_argc_argv(int argc, char **argv) {
  if (argv) {
    for (int i = 0; i < argc; ++i) {
      delete[] argv[i];
    }
    delete[] argv;
  }
}

} // namespace specfem
