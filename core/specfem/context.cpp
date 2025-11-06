#include "specfem/context.hpp"
#include "specfem/execute.hpp"
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace specfem {

// Static member initialization
std::unique_ptr<Context> Context::instance_ = nullptr;

Context &Context::instance() {
  if (instance_ == nullptr) {
    // Use new instead of make_unique since constructor is private
    instance_.reset(new Context());
  }
  return *instance_;
}

Context::Context()
    : mpi_(nullptr), kokkos_initialized_(false), mpi_initialized_(false),
      context_initialized_(false) {}

// Note: Destructor will automatically clean up unique_ptr

Context::~Context() {
  if (context_initialized_) {
    finalize();
  }
}

bool Context::initialize(int argc, char *argv[]) {
  if (context_initialized_) {
    return false; // Already initialized
  }

  try {
    // Initialize MPI first
    mpi_ = std::make_unique<specfem::MPI::MPI>(&argc, &argv);
    mpi_initialized_ = true;

    // Initialize Kokkos
    Kokkos::initialize(argc, argv);
    kokkos_initialized_ = true;

    context_initialized_ = true;
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Error during initialization: " << e.what() << std::endl;
    // Cleanup on failure
    if (kokkos_initialized_) {
      Kokkos::finalize();
      kokkos_initialized_ = false;
    }
    if (mpi_initialized_) {
      mpi_.reset();
      mpi_initialized_ = false;
    }
    return false;
  }
}

bool Context::initialize_from_python(const std::vector<std::string> &py_argv) {
  if (context_initialized_) {
    return false; // Already initialized
  }

  int argc;
  char **argv;
  setup_argc_argv(py_argv, argc, argv);

  bool result = initialize(argc, argv);

  cleanup_argc_argv(argc, argv);
  return result;
}

bool Context::finalize() {
  if (!context_initialized_) {
    return false;
  }

  try {
    // Finalize Kokkos
    if (kokkos_initialized_) {
      Kokkos::finalize();
      kokkos_initialized_ = false;
    }

    // Finalize MPI (unique_ptr automatically handles deletion)
    if (mpi_initialized_) {
      mpi_.reset();
      mpi_initialized_ = false;
    }

    context_initialized_ = false;
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Error during finalization: " << e.what() << std::endl;
    return false;
  }
}

template <specfem::dimension::type DimensionTag>
bool Context::execute(
    const YAML::Node &parameter_dict, const YAML::Node &default_dict,
    std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task> >
        &tasks) {
  if (!context_initialized_) {
    std::cerr << "Context not initialized. Call initialize() first."
              << std::endl;
    return false;
  }

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

bool Context::is_initialized() const { return context_initialized_; }

bool Context::is_kokkos_initialized() const { return kokkos_initialized_; }

void Context::setup_argc_argv(const std::vector<std::string> &args, int &argc,
                              char **&argv) {
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

void Context::cleanup_argc_argv(int argc, char **argv) {
  if (argv) {
    for (int i = 0; i < argc; ++i) {
      delete[] argv[i];
    }
    delete[] argv;
  }
}

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
    : context_(Context::instance()), should_finalize_(false) {
  if (context_.is_initialized()) {
    throw std::runtime_error("ContextGuard: Context already initialized. "
                             "Cannot create multiple ContextGuard instances.");
  }

  if (!context_.initialize(argc, argv)) {
    throw std::runtime_error("ContextGuard: Failed to initialize Context");
  }

  should_finalize_ = true;
}

ContextGuard::ContextGuard(const std::vector<std::string> &args)
    : context_(Context::instance()), should_finalize_(false) {
  if (context_.is_initialized()) {
    throw std::runtime_error("ContextGuard: Context already initialized. "
                             "Cannot create multiple ContextGuard instances.");
  }

  if (!context_.initialize_from_python(args)) {
    throw std::runtime_error("ContextGuard: Failed to initialize Context");
  }

  should_finalize_ = true;
}

ContextGuard::~ContextGuard() {
  if (should_finalize_) {
    context_.finalize();
  }
}

Context &ContextGuard::get_context() { return context_; }

specfem::MPI::MPI *ContextGuard::get_mpi() const { return context_.get_mpi(); }

} // namespace specfem
