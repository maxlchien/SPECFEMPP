#include "enumerations/dimension.hpp"
#include "io/interface.hpp"
#include "parameter_parser/interface.hpp"
#include "specfem/periodic_tasks.hpp"
#include "specfem/receivers.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include <boost/program_options.hpp>
#include <chrono>
#include <iostream>
#include <yaml-cpp/yaml.h>

boost::program_options::options_description define_args() {
  namespace po = boost::program_options;

  po::options_description desc{ "======================================\n"
                                "------------SPECFEM Kokkos------------\n"
                                "======================================" };

  desc.add_options()("help,h", "Print this help message")(
      "parameters_file,p", po::value<std::string>(),
      "Location to parameters file")(
      "default_file,d",
      po::value<std::string>()->default_value(__default_file__),
      "Location of default parameters file.");

  return desc;
}

int parse_args(int argc, char **argv,
               boost::program_options::variables_map &vm) {

  const auto desc = define_args();
  boost::program_options::store(
      boost::program_options::parse_command_line(argc, argv, desc), vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }

  if (!vm.count("parameters_file")) {
    std::cout << desc << std::endl;
    return 0;
  }

  return 1;
}

void execute(const YAML::Node parameter_dict, const YAML::Node default_dict,
             std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task<
                 specfem::dimension::type::dim3> > >
                 tasks,
             specfem::MPI::MPI *mpi) {
  mpi->cout("=====================================================");
  mpi->cout("                     SPECFEM++                       ");
  mpi->cout("=====================================================\n\n");

  mpi->cout("=====================================================\n");

  // --------------------------------------------------------------
  //                    Read parameter file
  // --------------------------------------------------------------

  mpi->cout("Reading the parameter file:");
  mpi->cout("---------------------------");
  specfem::runtime_configuration::setup setup(parameter_dict, default_dict);
  const auto database_filename = setup.get_databases();

  // Get simulation parameters
  const specfem::simulation::type simulation_type = setup.get_simulation_type();
  const type_real dt = setup.get_dt();
  const int nsteps = setup.get_nsteps();

  mpi->cout("\n=====================================================\n");

  // --------------------------------------------------------------
  //                   Read mesh and materials
  // --------------------------------------------------------------

  // Read mesh from the mesh database file
  mpi->cout("Reading the mesh:");
  mpi->cout("-----------------");
  auto start_time = std::chrono::system_clock::now();
  const auto mesh =
      specfem::io::meshfem3d::read_3d_mesh(database_filename, mpi);
  std::chrono::duration<double> elapsed_seconds =
      std::chrono::system_clock::now() - start_time;
  mpi->cout("Time to read mesh: " + std::to_string(elapsed_seconds.count()) +
            " seconds");

  mpi->cout("=====================================================\n");

  // --------------------------------------------------------------
  //                   Get Quadrature
  // --------------------------------------------------------------
  const auto quadrature = setup.instantiate_quadrature();
  // --------------------------------------------------------------

  mpi->cout("=====================================================\n");

  // --------------------------------------------------------------
  //                   Get Sources
  // --------------------------------------------------------------
  auto [sources, t0] =
      specfem::io::read_3d_sources(setup.get_sources(), nsteps, setup.get_t0(),
                                   setup.get_dt(), simulation_type);
  setup.update_t0(t0); // Update t0 in case it was changed

  mpi->cout("Source Information:");
  mpi->cout("-------------------");
  if (mpi->main_proc()) {
    std::cout << "Number of sources : " << sources.size() << "\n" << std::endl;
  }

  for (auto &source : sources) {
    mpi->cout(source->print());
  }

  mpi->cout("=====================================================\n");

  // --------------------------------------------------------------
  //                   Get receivers
  // --------------------------------------------------------------
  // create single receiver receivers vector for now
  auto receivers = specfem::io::read_3d_receivers(setup.get_stations());

  mpi->cout("Receiver Information:");
  mpi->cout("---------------------");

  if (mpi->main_proc()) {
    std::cout << "Number of receivers : " << receivers.size() << "\n"
              << std::endl;
  }

  for (auto &receiver : receivers) {
    mpi->cout(receiver->print());
  }

  mpi->cout("=====================================================\n");

  // --------------------------------------------------------------
  //                   Generate Assembly
  // --------------------------------------------------------------
  const int nstep_between_samples = setup.get_nstep_between_samples();
  const int max_seismogram_time_step = setup.get_max_seismogram_step();

  mpi->cout("Generating Assembly:");
  mpi->cout("--------------------");
  start_time = std::chrono::system_clock::now();
  specfem::assembly::assembly<specfem::dimension::type::dim3> assembly(
      mesh, quadrature, sources, receivers, setup.get_seismogram_types(),
      setup.get_t0(), dt, nsteps, max_seismogram_time_step,
      nstep_between_samples, setup.get_simulation_type(),
      setup.allocate_boundary_values(), setup.instantiate_property_reader());
  elapsed_seconds = std::chrono::system_clock::now() - start_time;
  mpi->cout(assembly.print());
  mpi->cout("-------------------------------");
  mpi->cout("Time to generate assembly: " +
            std::to_string(elapsed_seconds.count()) + " seconds.");

  mpi->cout("=====================================================\n");

  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                   Instantiate Timescheme
  // --------------------------------------------------------------
  const auto time_scheme = setup.instantiate_timescheme(assembly.fields);

  if (mpi->main_proc())
    std::cout << *time_scheme << std::endl;

  // --------------------------------------------------------------

  mpi->cout("=====================================================\n");

  // --------------------------------------------------------------
  //                   Instantiate plotter
  // --------------------------------------------------------------
  const auto wavefield_plotter =
      setup.instantiate_wavefield_plotter(assembly, dt, mpi);
  if (wavefield_plotter) {
    tasks.push_back(wavefield_plotter);
  }
  // --------------------------------------------------------------

  mpi->cout("=====================================================\n");

  // --------------------------------------------------------------
  //                   Instantiate Solver
  // --------------------------------------------------------------
  std::shared_ptr<specfem::solver::solver> solver =
      setup.instantiate_solver<5>(dt, assembly, time_scheme, tasks);
  // --------------------------------------------------------------

  mpi->cout("=====================================================\n");

  // --------------------------------------------------------------
  //                   Execute Solver
  // --------------------------------------------------------------
  // Time the solver
  mpi->cout("Executing time loop:");
  mpi->cout("-------------------------------");

  const auto solver_start_time = std::chrono::system_clock::now();
  solver->run();
  const auto solver_end_time = std::chrono::system_clock::now();

  std::chrono::duration<double> solver_time =
      solver_end_time - solver_start_time;

  mpi->cout("Solver time: " + std::to_string(solver_time.count()) +
            " seconds.\n");
  // --------------------------------------------------------------

  mpi->cout("=====================================================\n");

  // --------------------------------------------------------------
  //                   Write Seismograms
  // --------------------------------------------------------------
  const auto seismogram_writer = setup.instantiate_seismogram_writer();
  if (seismogram_writer) {
    mpi->cout("Writing seismogram files.");
    seismogram_writer->write(assembly);
  }
  // --------------------------------------------------------------

  mpi->cout("=====================================================\n");
  mpi->cout("Done.\n");

  return;
}

int main(int argc, char **argv) {
  // Initialize MPI
  specfem::MPI::MPI *mpi = new specfem::MPI::MPI(&argc, &argv);
  // Initialize Kokkos
  Kokkos::initialize(argc, argv);
  {
    boost::program_options::variables_map vm;
    if (parse_args(argc, argv, vm)) {
      const std::string parameters_file =
          vm["parameters_file"].as<std::string>();
      const std::string default_file = vm["default_file"].as<std::string>();
      const YAML::Node parameter_dict = YAML::LoadFile(parameters_file);
      const YAML::Node default_dict = YAML::LoadFile(default_file);
      std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task<
          specfem::dimension::type::dim3> > >
          tasks;
      execute(parameter_dict, default_dict, tasks, mpi);
    }
  }
  // Finalize Kokkos
  Kokkos::finalize();
  // Finalize MPI
  delete mpi;
  return 0;
}
