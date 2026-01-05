#include "mpi.hpp"

namespace specfem {

int MPI::rank = -1;
int MPI::size = -1;

void MPI::initialize(int *argc, char ***argv) {
#ifdef MPI_PARALLEL
  int initialized;
  MPI_Initialized(&initialized);

  if (!initialized) {
    MPI_Init(argc, argv);
  }

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
  // Non-MPI build: single process
  rank = 0;
  size = 1;
#endif
}

void MPI::finalize() {
#ifdef MPI_PARALLEL
  int finalized;
  MPI_Finalized(&finalized);

  if (!finalized) {
    MPI_Finalize();
  }
#endif

  rank = -1;
  size = -1;
}

} // namespace specfem
