!=====================================================================
!
!                          S p e c f e m 3 D
!                          -----------------
!
!     Main historical authors: Dimitri Komatitsch and Jeroen Tromp
!                              CNRS, France
!                       and Princeton University, USA
!                 (there are currently many more authors!)
!                           (c) October 2017
!
! This program is free software; you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation; either version 3 of the License, or
! (at your option) any later version.
!
! This program is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.
!
! You should have received a copy of the GNU General Public License along
! with this program; if not, write to the Free Software Foundation, Inc.,
! 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
!
!=====================================================================


module constants

  include "constants.h"

  ! proc number for MPI process
  integer :: myrank

  ! a negative initial value is a convention that indicates that groups (i.e. sub-communicators, one per run) are off by default
  integer :: mygroup = -1

  ! MPI status size (will be set in init_mpi()
  integer :: my_status_size   = 1
  integer :: my_status_source = 1
  integer :: my_status_tag    = 1

  ! create a copy of the original output file path, to which we may add a "run0001/", "run0002/", "run0003/" prefix later
  ! if NUMBER_OF_SIMULTANEOUS_RUNS > 1
  character(len=MAX_STRING_LEN) :: OUTPUT_FILES = OUTPUT_FILES_BASE

  ! if doing simultaneous runs for the same mesh and model, see who should read the mesh and the model and broadcast it to others
  ! we put a default value here
  logical :: I_should_read_the_database = .true.

end module constants

!
!-------------------------------------------------------------------------------------------------
!

  module shared_input_parameters

! holds input parameters given in DATA/Par_file

  use constants, only: MAX_STRING_LEN

  implicit none

  ! parameters read from parameter file
  integer :: NPROC

  ! simulation parameters
  integer :: SIMULATION_TYPE
  integer :: NOISE_TOMOGRAPHY
  logical :: SAVE_FORWARD
  logical :: INVERSE_FWI_FULL_PROBLEM

  integer :: UTM_PROJECTION_ZONE
  logical :: SUPPRESS_UTM_PROJECTION

  logical :: UNDO_ATTENUATION_AND_OR_PML
  integer :: NT_DUMP_ATTENUATION

  ! number of time steps
  integer :: NSTEP
  double precision :: DT

  ! number of time step for external source time function
  integer :: NSTEP_STF

  ! Local Time Stepping (LTS)
  logical :: LTS_MODE

  ! partitioning scheme
  integer :: PARTITIONING_TYPE

  ! LDD Runge-Kutta time scheme
  logical :: USE_LDDRK
  logical :: INCREASE_CFL_FOR_LDDRK
  double precision :: RATIO_BY_WHICH_TO_INCREASE_IT

  ! mesh
  integer :: NGNOD

  character(len=MAX_STRING_LEN) :: MODEL
  character(len=MAX_STRING_LEN) :: COUPLED_MODEL_DIRECTORY
  character(len=MAX_STRING_LEN) :: SEP_MODEL_DIRECTORY

  ! physical parameters
  logical :: APPROXIMATE_OCEAN_LOAD,TOPOGRAPHY,ATTENUATION,ANISOTROPY
  logical :: GRAVITY

  character(len=MAX_STRING_LEN) :: TOMOGRAPHY_PATH

  character(len=MAX_STRING_LEN) :: Par_file
  character(len=MAX_STRING_LEN) :: MESH_PAR_FILE

  ! fault parameters
  logical :: HAS_FINITE_FAULT_SOURCE
  character(len=MAX_STRING_LEN) :: FAULT_PAR_FILE
  character(len=MAX_STRING_LEN) :: FAULT_STATIONS
  character(len=MAX_STRING_LEN) :: STRESS_FRICTION_FILE
  character(len=MAX_STRING_LEN) :: RSF_HETE_FILE

  ! attenuation
  ! reference frequency of seismic model
  double precision :: ATTENUATION_f0_REFERENCE
  ! Olsen attenuation (scaling from Vs)
  logical :: USE_OLSEN_ATTENUATION
  double precision :: OLSEN_ATTENUATION_RATIO
  ! automatic frequency band selection
  logical :: COMPUTE_FREQ_BAND_AUTOMATIC
  ! attenuation period range over which we try to mimic a constant Q factor
  double precision :: MIN_ATTENUATION_PERIOD,MAX_ATTENUATION_PERIOD
  ! logarithmic center frequency (center of attenuation band)
  double precision :: ATT_F_C_SOURCE

  ! absorbing boundaries
  logical :: PML_CONDITIONS,PML_INSTEAD_OF_FREE_SURFACE
  double precision :: f0_FOR_PML
  logical :: STACEY_ABSORBING_CONDITIONS,STACEY_INSTEAD_OF_FREE_SURFACE
  ! To use a bottom free surface instead of absorbing Stacey or PML condition
  logical :: BOTTOM_FREE_SURFACE

  ! sources and receivers Z coordinates given directly instead of with depth
  logical :: USE_SOURCES_RECEIVERS_Z

  ! for simultaneous runs from the same batch job
  integer :: NUMBER_OF_SIMULTANEOUS_RUNS
  logical :: BROADCAST_SAME_MESH_AND_MODEL

  ! movies
  logical :: CREATE_SHAKEMAP = .false.
  logical :: MOVIE_SURFACE = .false.
  logical :: MOVIE_VOLUME = .false.
  logical :: SAVE_DISPLACEMENT = .false.
  logical :: USE_HIGHRES_FOR_MOVIES = .false.
  logical :: MOVIE_VOLUME_STRESS = .false.
  integer :: MOVIE_TYPE = 1

  integer :: NTSTEP_BETWEEN_FRAMES
  double precision :: HDUR_MOVIE

  ! mesh
  logical :: SAVE_MESH_FILES
  character(len=MAX_STRING_LEN) :: LOCAL_PATH

  ! seismograms
  integer :: NTSTEP_BETWEEN_OUTPUT_INFO
  integer :: NTSTEP_BETWEEN_OUTPUT_SEISMOS,NTSTEP_BETWEEN_READ_ADJSRC
  integer :: NTSTEP_BETWEEN_OUTPUT_SAMPLE ! subsamp_seismos is deprecated and renamed to NTSTEP_BETWEEN_OUTPUT_SAMPLE
  logical :: SAVE_SEISMOGRAMS_DISPLACEMENT,SAVE_SEISMOGRAMS_VELOCITY,SAVE_SEISMOGRAMS_ACCELERATION,SAVE_SEISMOGRAMS_PRESSURE
  logical :: SAVE_SEISMOGRAMS_STRAIN
  logical :: SAVE_SEISMOGRAMS_IN_ADJOINT_RUN
  logical :: WRITE_SEISMOGRAMS_BY_MAIN,SAVE_ALL_SEISMOS_IN_ONE_FILE,USE_BINARY_FOR_SEISMOGRAMS,SU_FORMAT
  logical :: ASDF_FORMAT, READ_ADJSRC_ASDF

  ! sources
  logical :: USE_FORCE_POINT_SOURCE
  character(len=MAX_STRING_LEN) :: SOURCE_FILENAME
  logical :: USE_RICKER_TIME_FUNCTION,PRINT_SOURCE_TIME_FUNCTION

  ! external source time function
  logical :: USE_EXTERNAL_SOURCE_FILE

  logical :: USE_TRICK_FOR_BETTER_PRESSURE,USE_SOURCE_ENCODING,OUTPUT_ENERGY
  logical :: ANISOTROPIC_KL,SAVE_TRANSVERSE_KL,APPROXIMATE_HESS_KL,SAVE_MOHO_MESH
  logical :: ANISOTROPIC_VELOCITY_KL
  integer :: NTSTEP_BETWEEN_OUTPUT_ENERGY

  ! GPU simulations
  logical :: GPU_MODE

  ! adios file output
  logical :: ADIOS_ENABLED
  logical :: ADIOS_FOR_DATABASES, ADIOS_FOR_MESH, ADIOS_FOR_FORWARD_ARRAYS, ADIOS_FOR_KERNELS, ADIOS_FOR_UNDO_ATTENUATION

  ! HDF5 file i/o
  logical :: HDF5_ENABLED = .false.              ! for all databases i/o in hdf5
  logical :: HDF5_FOR_MOVIES = .false.           ! for movies (shakemap, surface movies, volume movies)

  ! HDF5 seismogram output
  logical :: HDF5_FORMAT  = .false.           ! for seismograms output in hdf5

  ! HDF5 IO server
  ! number of io dedicated nodes
  integer :: HDF5_IO_NODES = 0

  ! HDF5 IO writing mode (collective or independent)
  logical :: H5_COL = .true.

  ! flag for io-dedicated/compute node.
  logical :: IO_storage_task = .false.
  logical :: IO_compute_task = .true.

  ! external code coupling (DSM, AxiSEM)
  logical :: COUPLE_WITH_INJECTION_TECHNIQUE
  integer :: INJECTION_TECHNIQUE_TYPE
  character(len=MAX_STRING_LEN) :: TRACTION_PATH,TRACTION_PATH_new
  character(len=MAX_STRING_LEN) :: FKMODEL_FILE
  logical :: RECIPROCITY_AND_KH_INTEGRAL

  ! prescribed wavefield discontinuity on an interface
  logical :: IS_WAVEFIELD_DISCONTINUITY = .false. ! if .true. then wavefield discontinuity is turned on (default is false)

  end module shared_input_parameters

!
!-------------------------------------------------------------------------------------------------
!

  module shared_compute_parameters

  ! parameters to be computed based upon parameters above read from file
  use constants, only: CUSTOM_REAL

  implicit none

  ! number of sources given in CMTSOLUTION file
  integer :: NSOURCES

  ! anchor points
  integer :: NGNOD2D

  ! model
  integer :: IMODEL

  !! VM VM number of source for external source time function
  integer :: NSOURCES_STF

  ! simulation type
  logical :: ACOUSTIC_SIMULATION = .false.
  logical :: ELASTIC_SIMULATION = .false.
  logical :: POROELASTIC_SIMULATION = .false.

  ! fault rupture simulation
  logical :: FAULT_SIMULATION = .false.

  ! free surface
  ! for elevation search: x/y coordinates of free surface element midpoints
  real(kind=CUSTOM_REAL), dimension(:,:), allocatable :: free_surface_xy_midpoints
  real(kind=CUSTOM_REAL) :: free_surface_typical_size
  ! flag to calculate typical size only once
  logical :: free_surface_has_typical_size = .false.

  end module shared_compute_parameters

!
!-------------------------------------------------------------------------------------------------
!

  module shared_parameters

  use shared_input_parameters
  use shared_compute_parameters

  implicit none

  end module shared_parameters
