#include "plot_wavefield.hpp"
#include "enumerations/display.hpp"
#include "enumerations/interface.hpp"
#include "plotter.hpp"
#include "specfem/assembly.hpp"
#include "utilities/strings.hpp"

#ifdef NO_VTK

#include <sstream>

#else

#include <algorithm>
#include <boost/filesystem.hpp>
#include <cmath>
#include <fstream>
#include <vtkActor.h>
#include <vtkBiQuadraticQuad.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkDataSetMapper.h>
#include <vtkExtractEdges.h>
#include <vtkFloatArray.h>
#include <vtkGraphicsFactory.h>
#include <vtkJPEGWriter.h>
#include <vtkLookupTable.h>
#include <vtkNamedColors.h>
#include <vtkPNGWriter.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkQuad.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>
#include <vtkUnsignedCharArray.h>
#include <vtkUnstructuredGrid.h>
#include <vtkWindowToImageFilter.h>

#ifndef NO_HDF5
#include <hdf5.h>
#endif // NO_HDF5

#endif // NO_VTK

#ifdef NO_VTK

// Add this constructor implementation for NO_VTK builds
specfem::periodic_tasks::plot_wavefield::plot_wavefield(
    const specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly,
    const specfem::display::format &output_format,
    const specfem::wavefield::type &wavefield_type,
    const specfem::wavefield::simulation_field &wavefield, const type_real &dt,
    const int &time_interval, const boost::filesystem::path &output_folder,
    specfem::MPI::MPI *mpi)
    : assembly(assembly), wavefield(wavefield), wavefield_type(wavefield_type),
      plotter(time_interval), output_format(output_format),
      output_folder(output_folder), nspec(assembly.mesh.nspec), dt(dt),
      time_interval(time_interval), ngllx(assembly.mesh.element_grid.ngllx),
      ngllz(assembly.mesh.element_grid.ngllz), mpi(mpi) {
  std::ostringstream message;
  message
      << "Display section is not enabled, since SPECFEM++ was built without "
         "VTK\n"
      << "Please install VTK and rebuild SPECFEM++ with -DVTK_DIR=/path/to/vtk";
  throw std::runtime_error(message.str());
}

void specfem::periodic_tasks::plot_wavefield::run(
    specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly,
    const int istep) {
  std::ostringstream message;
  message
      << "Display section is not enabled, since SPECFEM++ was built without "
         "VTK\n"
      << "Please install VTK and rebuild SPECFEM++ with -DVTK_DIR=/path/to/vtk";
  throw std::runtime_error(message.str());
}

void specfem::periodic_tasks::plot_wavefield::initialize(
    specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly) {
  std::ostringstream message;
  message
      << "Display section is not enabled, since SPECFEM++ was built without "
         "VTK\n"
      << "Please install VTK and rebuild SPECFEM++ with -DVTK_DIR=/path/to/vtk";
  throw std::runtime_error(message.str());
}

void specfem::periodic_tasks::plot_wavefield::finalize(
    specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly) {
  std::ostringstream message;
  message
      << "Display section is not enabled, since SPECFEM++ was built without "
         "VTK\n"
      << "Please install VTK and rebuild SPECFEM++ with -DVTK_DIR=/path/to/vtk";
  throw std::runtime_error(message.str());
}

#else

// Constructor
specfem::periodic_tasks::plot_wavefield::plot_wavefield(
    const specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly,
    const specfem::display::format &output_format,
    const specfem::wavefield::type &wavefield_type,
    const specfem::wavefield::simulation_field &wavefield, const type_real &dt,
    const int &time_interval, const boost::filesystem::path &output_folder,
    specfem::MPI::MPI *mpi)
    : assembly(assembly), wavefield(wavefield), wavefield_type(wavefield_type),
      plotter(time_interval), output_format(output_format),
      output_folder(output_folder), nspec(assembly.mesh.nspec), dt(dt),
      time_interval(time_interval), ngllx(assembly.mesh.element_grid.ngllx),
      ngllz(assembly.mesh.element_grid.ngllz), mpi(mpi) {};

// Sigmoid function centered at 0.0
double specfem::periodic_tasks::plot_wavefield::sigmoid(double x) {
  return (1 / (1 + std::exp(-100 * x)) - 0.5) * 1.5;
}

// Get wavefield type to display
specfem::wavefield::type
specfem::periodic_tasks::plot_wavefield::get_wavefield_type() {
  if (wavefield_type == specfem::wavefield::type::displacement) {
    return specfem::wavefield::type::displacement;
  } else if (wavefield_type == specfem::wavefield::type::velocity) {
    return specfem::wavefield::type::velocity;
  } else if (wavefield_type == specfem::wavefield::type::acceleration) {
    return specfem::wavefield::type::acceleration;
  } else if (wavefield_type == specfem::wavefield::type::pressure) {
    return specfem::wavefield::type::pressure;
  } else if (wavefield_type == specfem::wavefield::type::rotation) {
    return specfem::wavefield::type::rotation;
  } else if (wavefield_type == specfem::wavefield::type::intrinsic_rotation) {
    return specfem::wavefield::type::intrinsic_rotation;
  } else if (wavefield_type == specfem::wavefield::type::curl) {
    return specfem::wavefield::type::curl;
  } else {
    std::ostringstream message;
    message << "Unsupported wavefield type for display. "
            << specfem::wavefield::to_string(wavefield_type)
            << " is not supported: " << __FILE__ << ":" << __LINE__;
    throw std::runtime_error(message.str());
  }
}

// Maps different materials to different colors
vtkSmartPointer<vtkDataSetMapper>
specfem::periodic_tasks::plot_wavefield::map_materials_with_color() {

  const auto &element_types = assembly.element_types;

  const std::unordered_map<specfem::element::medium_tag, std::array<int, 3> >
      material_colors = {
        { specfem::element::medium_tag::acoustic, // aqua color
          { 0, 255, 255 } },
        { specfem::element::medium_tag::elastic_psv, // sienna color
          { 160, 82, 45 } },
        { specfem::element::medium_tag::elastic_sh, // sienna color
          { 160, 82, 45 } },
        { specfem::element::medium_tag::elastic_psv_t, // sienna color
          { 160, 82, 45 } },
        { specfem::element::medium_tag::poroelastic, // off navy color
          { 40, 40, 128 } },
        { specfem::element::medium_tag::electromagnetic_te, // dark gray color
          { 169, 169, 169 } },
      };

  const auto &coordinates = assembly.mesh.h_coord;
  const int nspec = assembly.mesh.nspec;
  const int ngllx = assembly.mesh.element_grid.ngllx;
  const int ngllz = assembly.mesh.element_grid.ngllz;

  const int cell_points = 4;

  const std::array<int, cell_points> z_index = { 0, ngllz - 1, ngllz - 1, 0 };
  const std::array<int, cell_points> x_index = { 0, 0, ngllx - 1, ngllx - 1 };

  auto points = vtkSmartPointer<vtkPoints>::New();

  auto cells = vtkSmartPointer<vtkCellArray>::New();

  auto colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
  colors->SetNumberOfComponents(3);
  colors->SetName("Colors");

  for (int icell = 0; icell < nspec; ++icell) {
    for (int i = 0; i < cell_points; ++i) {
      points->InsertNextPoint(coordinates(0, icell, z_index[i], x_index[i]),
                              coordinates(1, icell, z_index[i], x_index[i]),
                              0.0);
    }
    auto quad = vtkSmartPointer<vtkQuad>::New();
    for (int i = 0; i < cell_points; ++i) {
      quad->GetPointIds()->SetId(i, icell * cell_points + i);
    }
    cells->InsertNextCell(quad);

    const auto material = element_types.get_medium_tag(icell);
    const auto color = material_colors.at(material);
    unsigned char color_uc[3] = { static_cast<unsigned char>(color[0]),
                                  static_cast<unsigned char>(color[1]),
                                  static_cast<unsigned char>(color[2]) };
    colors->InsertNextTypedTuple(color_uc);
  }

  auto unstructured_grid = vtkSmartPointer<vtkUnstructuredGrid>::New();
  unstructured_grid->SetPoints(points);
  unstructured_grid->SetCells(VTK_QUAD, cells);

  unstructured_grid->GetCellData()->SetScalars(colors);

  auto mapper = vtkSmartPointer<vtkDataSetMapper>::New();
  mapper->SetInputData(unstructured_grid);

  return mapper;
}

/**
 * @brief Get the wavefield on vtkUnstructured grid object as biquadratic quads

 * This function creates bilinear quadrilateral from the element corners,
 * midpoints and center points of element sides.
 *
 * Graphical Explanation looking at a single element (see below), create:
 *
 *
 *     3----•-----6-----•----2
 *     |    |     |     |    |
 *     •----•-----•-----•----•
 *     |    |     |     |    |
 *     7----•-----8-----•----5
 *     |    |     |     |    |
 *     •----•-----•-----•----•
 *     |    |     |     |    |
 *     0----•-----4-----•----1
 *
 * Where the above points (for GLL = 5) that are used to create the bilinear
 * quad are indicated by numbers 0-8 in the order of the points in the quad.
 * Each element has therefore 9 points, that are the used to return a
 * vtkUnstructuredGrid object containing vtkBiQuadraticQuad cells.
 *
 * @return vtkSmartPointer<vtkUnstructuredGrid>
 */
void specfem::periodic_tasks::plot_wavefield::create_biquad_grid() {
  const auto &coordinates = assembly.mesh.h_coord;

  const int ncells = nspec;
  const int cell_points = 9;

  const std::array<int, cell_points> z_index = { 0,
                                                 0,
                                                 ngllz - 1,
                                                 ngllz - 1,
                                                 0,
                                                 (ngllz - 1) / 2,
                                                 ngllz - 1,
                                                 (ngllz - 1) / 2,
                                                 (ngllz - 1) / 2 };
  const std::array<int, cell_points> x_index = { 0,
                                                 ngllx - 1,
                                                 ngllx - 1,
                                                 0,
                                                 (ngllx - 1) / 2,
                                                 ngllx - 1,
                                                 (ngllx - 1) / 2,
                                                 0,
                                                 (ngllx - 1) / 2 };

  auto points = vtkSmartPointer<vtkPoints>::New();
  auto cells = vtkSmartPointer<vtkCellArray>::New();

  for (int icell = 0; icell < ncells; ++icell) {
    for (int i = 0; i < cell_points; ++i) {
      points->InsertNextPoint(coordinates(0, icell, z_index[i], x_index[i]),
                              coordinates(1, icell, z_index[i], x_index[i]),
                              0.0);
    }
    auto quad = vtkSmartPointer<vtkBiQuadraticQuad>::New();
    for (int i = 0; i < cell_points; ++i) {
      quad->GetPointIds()->SetId(i, icell * cell_points + i);
    }
    cells->InsertNextCell(quad);
  }

  unstructured_grid = vtkSmartPointer<vtkUnstructuredGrid>::New();
  unstructured_grid->SetPoints(points);
  unstructured_grid->SetCells(VTK_BIQUADRATIC_QUAD, cells);
}

/**
 * @brief Get the wavefield on vtkUnstructured grid object
 *
 *
 * This function creates vertices for quadrilaterals from the coordinates x and
 * z. and the element based field. The field is of shape (nspec, ngll, ngll), so
 * are the coordinates. The functions creates quads of 4 GLL points. For ngll =
 * 5 this means that we have 16 quads per element, and a total of nspec * 16
 * quads.
 *
 * Graphical Explanation:
 *
 * Looking at a single element (see below), create quadrilateral for each
 * subrectangle of the element. Starting with the ix=0, iz=0 corner moving
 * counterclockwise for each subquad, indicated by the numbers coinciding with
 * the GLL points. Then we move in ix direction for each quad indicated by the
 * number on the face of each quad.
 *
 *     •----•-----•-----3----2
 *     | 12 |  13 |  14 | 15 |
 *     •----•-----•-----0----1
 *     |  8 |   9 |  10 | 11 |
 *     •----•-----•-----•----•
 *     |  4 |   5 |   6 |  7 |
 *     3----2-----•-----•----•
 *     |  0 |   1 |   2 |  3 |
 *     0----1-----•-----•----•
 *
 * So, for GLL = 5 each element each element has therefore 16 (as numbered 0-15)
 * quads. For the first and last quad we indicate the order of the gll points
 * used as vertices of the quad (0-3). Finally, the quads are the used to return
 * a vtkUnstructuredGrid object containing vtkQuad cells.
 *
 * The wavefield is assigned to the points accordingly.
 *
 */
void specfem::periodic_tasks::plot_wavefield::create_quad_grid() {
  const auto &coordinates = assembly.mesh.h_coord;

  // For ngll = 5, each spectral element has 16 cells
  const int n_cells_per_spec = (ngllx - 1) * (ngllz - 1);
  const int ncells = nspec * n_cells_per_spec;

  const int n_cell_points = 4;

  // The points of the cells are ordered as follows:
  // 3--2
  // |  |
  // 0--1
  const std::array<int, n_cell_points> z_index = { 0, 0, 1, 1 };
  const std::array<int, n_cell_points> x_index = { 0, 1, 1, 0 };

  auto points = vtkSmartPointer<vtkPoints>::New();
  auto cells = vtkSmartPointer<vtkCellArray>::New();

  int point_counter = 0; // Keep track of the global point index

  // Loop over the cells
  for (int ispec = 0; ispec < nspec; ++ispec) {
    for (int iz = 0; iz < ngllz - 1; ++iz) {
      for (int ix = 0; ix < ngllx - 1; ++ix) {
        auto quad = vtkSmartPointer<vtkQuad>::New();

        for (int ipoint = 0; ipoint < n_cell_points; ++ipoint) {
          int iz_pos = iz + z_index[ipoint];
          int ix_pos = ix + x_index[ipoint];

          // Insert the point
          points->InsertNextPoint(coordinates(0, ispec, iz_pos, ix_pos),
                                  coordinates(1, ispec, iz_pos, ix_pos), 0.0);

          // Set the point ID for this quad
          quad->GetPointIds()->SetId(ipoint, point_counter);
          point_counter++;
        }

        // Add the cell
        cells->InsertNextCell(quad);
      }
    }
  }

  // Create the unstructured grid
  unstructured_grid = vtkSmartPointer<vtkUnstructuredGrid>::New();
  unstructured_grid->SetPoints(points);
  unstructured_grid->SetCells(VTK_QUAD, cells);
}

// Compute wavefield scalar values for the grid points
vtkSmartPointer<vtkFloatArray>
specfem::periodic_tasks::plot_wavefield::compute_wavefield_scalars(
    specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly) {
  const auto wavefield_type = get_wavefield_type();
  const auto &wavefield_data =
      assembly.generate_wavefield_on_entire_grid(wavefield, wavefield_type);

  auto scalars = vtkSmartPointer<vtkFloatArray>::New();

  // For quad grid
  if (unstructured_grid->GetCellType(0) == VTK_QUAD) {
    const int n_cell_points = 4;
    const std::array<int, n_cell_points> z_index = { 0, 0, 1, 1 };
    const std::array<int, n_cell_points> x_index = { 0, 1, 1, 0 };

    // Loop over the cells
    for (int ispec = 0; ispec < nspec; ++ispec) {
      for (int iz = 0; iz < ngllz - 1; ++iz) {
        for (int ix = 0; ix < ngllx - 1; ++ix) {
          for (int ipoint = 0; ipoint < n_cell_points; ++ipoint) {
            int iz_pos = iz + z_index[ipoint];
            int ix_pos = ix + x_index[ipoint];

            // Insert scalar value
            if (wavefield_type == specfem::wavefield::type::pressure ||
                wavefield_type == specfem::wavefield::type::rotation ||
                wavefield_type ==
                    specfem::wavefield::type::intrinsic_rotation ||
                wavefield_type == specfem::wavefield::type::curl) {
              scalars->InsertNextValue(
                  std::abs(wavefield_data(ispec, iz_pos, ix_pos, 0)));
            } else {
              scalars->InsertNextValue(
                  std::sqrt((wavefield_data(ispec, iz_pos, ix_pos, 0) *
                             wavefield_data(ispec, iz_pos, ix_pos, 0)) +
                            (wavefield_data(ispec, iz_pos, ix_pos, 1) *
                             wavefield_data(ispec, iz_pos, ix_pos, 1))));
            }
          }
        }
      }
    }
  }
  // For biquadratic grid
  else if (unstructured_grid->GetCellType(0) == VTK_BIQUADRATIC_QUAD) {
    const int cell_points = 9;
    const std::array<int, cell_points> z_index = { 0,
                                                   0,
                                                   ngllz - 1,
                                                   ngllz - 1,
                                                   0,
                                                   (ngllz - 1) / 2,
                                                   ngllz - 1,
                                                   (ngllz - 1) / 2,
                                                   (ngllz - 1) / 2 };
    const std::array<int, cell_points> x_index = { 0,
                                                   ngllx - 1,
                                                   ngllx - 1,
                                                   0,
                                                   (ngllx - 1) / 2,
                                                   ngllx - 1,
                                                   (ngllx - 1) / 2,
                                                   0,
                                                   (ngllx - 1) / 2 };

    for (int icell = 0; icell < nspec; ++icell) {
      for (int i = 0; i < cell_points; ++i) {
        if (wavefield_type == specfem::wavefield::type::pressure ||
            wavefield_type == specfem::wavefield::type::rotation ||
            wavefield_type == specfem::wavefield::type::intrinsic_rotation ||
            wavefield_type == specfem::wavefield::type::curl) {
          scalars->InsertNextValue(
              std::abs(wavefield_data(icell, z_index[i], x_index[i], 0)));
        } else {
          scalars->InsertNextValue(
              std::sqrt((wavefield_data(icell, z_index[i], x_index[i], 0) *
                         wavefield_data(icell, z_index[i], x_index[i], 0)) +
                        (wavefield_data(icell, z_index[i], x_index[i], 1) *
                         wavefield_data(icell, z_index[i], x_index[i], 1))));
        }
      }
    }
  }

  return scalars;
}

template <specfem::display::format format>
void initialize(specfem::periodic_tasks::plot_wavefield &plotter,
                vtkSmartPointer<vtkFloatArray> &scalars);

template <>
void initialize<specfem::display::format::vtkhdf>(
    specfem::periodic_tasks::plot_wavefield &plotter,
    vtkSmartPointer<vtkFloatArray> &scalars) {

#ifndef NO_HDF5

  // Initialize VTK HDF5 file for time series output
  plotter.current_timestep = 0;
  plotter.numPoints = 0;
  plotter.numCells = 0;

  // Create HDF5 file
  plotter.hdf5_filename = (plotter.output_folder / "wavefield.vtkhdf").string();
  hid_t hdf5_file_id = H5Fcreate(plotter.hdf5_filename.c_str(), H5F_ACC_TRUNC,
                                 H5P_DEFAULT, H5P_DEFAULT);
  hid_t vtkhdf_group =
      H5Gcreate(hdf5_file_id, "/VTKHDF", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  // Set VTKHDF attributes
  {
    hid_t attr_space, attr;
    int version[2] = { 2, 0 };
    hsize_t dims[1] = { 2 };
    attr_space = H5Screate_simple(1, dims, NULL);
    attr = H5Acreate(vtkhdf_group, "Version", H5T_NATIVE_INT, attr_space,
                     H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_INT, version);
    H5Aclose(attr);
    H5Sclose(attr_space);

    // Set Type attribute
    hid_t str_type = H5Tcopy(H5T_C_S1);
    H5Tset_size(str_type, 16);
    attr_space = H5Screate(H5S_SCALAR);
    attr = H5Acreate(vtkhdf_group, "Type", str_type, attr_space, H5P_DEFAULT,
                     H5P_DEFAULT);
    const char *type_str = "UnstructuredGrid";
    H5Awrite(attr, str_type, type_str);
    H5Aclose(attr);
    H5Sclose(attr_space);
    H5Tclose(str_type);
  }

  // Write static geometry to HDF5 file
  plotter.numPoints = plotter.unstructured_grid->GetNumberOfPoints();
  plotter.numCells = plotter.unstructured_grid->GetNumberOfCells();

  // Extract connectivity
  vtkCellArray *cells_vtkh = plotter.unstructured_grid->GetCells();
  vtkIdType npts;
  const vtkIdType *pts;
  std::vector<long long> connectivity;
  std::vector<long long> offsets;
  std::vector<unsigned char> types;

  offsets.push_back(0);
  for (vtkIdType i = 0; i < plotter.numCells; i++) {
    cells_vtkh->GetCellAtId(i, npts, pts);
    for (vtkIdType j = 0; j < npts; j++) {
      connectivity.push_back(pts[j]);
    }
    offsets.push_back(connectivity.size());
    types.push_back(plotter.unstructured_grid->GetCellType(i));
  }

  // Store connectivity size for later use
  plotter.numConnectivityIds = connectivity.size();

  // Extract points as 2D array (numPoints, 3)
  std::vector<double> pointCoords(plotter.numPoints * 3);
  for (vtkIdType i = 0; i < plotter.numPoints; i++) {
    double pt[3];
    plotter.unstructured_grid->GetPoint(i, pt);
    pointCoords[i * 3 + 0] = pt[0];
    pointCoords[i * 3 + 1] = pt[1];
    pointCoords[i * 3 + 2] = pt[2];
  }

  // Write Points (static geometry) - 2D array (numPoints, 3)
  hsize_t point_dims[2] = { (hsize_t)plotter.numPoints, 3 };
  hid_t dataspace = H5Screate_simple(2, point_dims, NULL);
  hid_t dataset = H5Dcreate(vtkhdf_group, "Points", H5T_NATIVE_DOUBLE,
                            dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
           pointCoords.data());
  H5Dclose(dataset);
  H5Sclose(dataspace);

  // Write Connectivity (static)
  hsize_t dims[1];
  dims[0] = connectivity.size();
  dataspace = H5Screate_simple(1, dims, NULL);
  dataset = H5Dcreate(vtkhdf_group, "Connectivity", H5T_NATIVE_LLONG, dataspace,
                      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dataset, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT,
           connectivity.data());
  H5Dclose(dataset);
  H5Sclose(dataspace);

  // Write Offsets (static)
  dims[0] = offsets.size();
  dataspace = H5Screate_simple(1, dims, NULL);
  dataset = H5Dcreate(vtkhdf_group, "Offsets", H5T_NATIVE_LLONG, dataspace,
                      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dataset, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT,
           offsets.data());
  H5Dclose(dataset);
  H5Sclose(dataspace);

  // Write Types (static)
  dims[0] = types.size();
  dataspace = H5Screate_simple(1, dims, NULL);
  dataset = H5Dcreate(vtkhdf_group, "Types", H5T_NATIVE_UCHAR, dataspace,
                      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dataset, H5T_NATIVE_UCHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT,
           types.data());
  H5Dclose(dataset);
  H5Sclose(dataspace);

  // Extract and write material IDs as CellData
  const auto &element_types = plotter.assembly.element_types;
  std::vector<int> material_ids;

  // Map each VTK cell to its corresponding spectral element
  // For quad grid: each spectral element has (ngllx-1)*(ngllz-1) cells
  const int n_cells_per_spec = (plotter.ngllx - 1) * (plotter.ngllz - 1);

  for (int ispec = 0; ispec < plotter.nspec; ++ispec) {
    const auto material_tag = element_types.get_medium_tag(ispec);
    // Convert enum to integer for HDF5 storage
    const int material_id = static_cast<int>(material_tag);

    // Assign same material ID to all sub-cells within this spectral element
    for (int icell = 0; icell < n_cells_per_spec; ++icell) {
      material_ids.push_back(material_id);
    }
  }

  // Create CellData group and write material IDs
  hid_t cd_group = H5Gcreate(vtkhdf_group, "CellData", H5P_DEFAULT, H5P_DEFAULT,
                             H5P_DEFAULT);
  dims[0] = material_ids.size();
  dataspace = H5Screate_simple(1, dims, NULL);
  dataset = H5Dcreate(cd_group, "MaterialID", H5T_NATIVE_INT, dataspace,
                      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
           material_ids.data());
  H5Dclose(dataset);
  H5Sclose(dataspace);
  H5Gclose(cd_group);

  // Create PointData group and extensible dataset for wavefield scalars
  hid_t pd_group = H5Gcreate(vtkhdf_group, "PointData", H5P_DEFAULT,
                             H5P_DEFAULT, H5P_DEFAULT);

  // Create extensible dataset for wavefield scalars
  // Initial size: 0 (will grow as needed)
  hsize_t pd_initial_dims[1] = { 0 };
  hsize_t pd_max_dims[1] = { H5S_UNLIMITED };
  hid_t pd_dataspace = H5Screate_simple(1, pd_initial_dims, pd_max_dims);

  // Create dataset creation property list and set chunking
  hid_t pd_plist = H5Pcreate(H5P_DATASET_CREATE);
  hsize_t pd_chunk_dims[1] = {
    (hsize_t)plotter.numPoints
  }; // Chunk size = one timestep worth of data
  H5Pset_chunk(pd_plist, 1, pd_chunk_dims);

  hid_t pd_dataset =
      H5Dcreate(pd_group, "Wavefield", H5T_NATIVE_FLOAT, pd_dataspace,
                H5P_DEFAULT, pd_plist, H5P_DEFAULT);
  H5Dclose(pd_dataset);
  H5Pclose(pd_plist);
  H5Sclose(pd_dataspace);
  H5Gclose(pd_group);

  // Create extensible temporal metadata arrays instead of pre-allocated ones
  hsize_t temp_initial_dims[1] = { 0 };
  hsize_t temp_max_dims[1] = { H5S_UNLIMITED };
  hsize_t temp_chunk_dims[1] = { 1 }; // Chunk size = 1 timestep

  // Create dataset creation property list for chunking
  hid_t temp_plist = H5Pcreate(H5P_DATASET_CREATE);
  H5Pset_chunk(temp_plist, 1, temp_chunk_dims);

  hid_t temp_dataspace = H5Screate_simple(1, temp_initial_dims, temp_max_dims);
  dataset = H5Dcreate(vtkhdf_group, "NumberOfPoints", H5T_NATIVE_LLONG,
                      temp_dataspace, H5P_DEFAULT, temp_plist, H5P_DEFAULT);
  H5Dclose(dataset);
  H5Sclose(temp_dataspace);

  temp_dataspace = H5Screate_simple(1, temp_initial_dims, temp_max_dims);
  dataset = H5Dcreate(vtkhdf_group, "NumberOfCells", H5T_NATIVE_LLONG,
                      temp_dataspace, H5P_DEFAULT, temp_plist, H5P_DEFAULT);
  H5Dclose(dataset);
  H5Sclose(temp_dataspace);

  temp_dataspace = H5Screate_simple(1, temp_initial_dims, temp_max_dims);
  dataset = H5Dcreate(vtkhdf_group, "NumberOfConnectivityIds", H5T_NATIVE_LLONG,
                      temp_dataspace, H5P_DEFAULT, temp_plist, H5P_DEFAULT);
  H5Dclose(dataset);
  H5Sclose(temp_dataspace);

  H5Pclose(temp_plist);

  // Create Steps group and extensible metadata
  hid_t steps_group =
      H5Gcreate(vtkhdf_group, "Steps", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  // Create extensible datasets for time steps metadata
  // NSteps attribute - will be updated during run
  hid_t attr_space = H5Screate(H5S_SCALAR);
  int initial_nsteps = 0;
  hid_t attr = H5Acreate(steps_group, "NSteps", H5T_NATIVE_INT, attr_space,
                         H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite(attr, H5T_NATIVE_INT, &initial_nsteps);
  H5Aclose(attr);
  H5Sclose(attr_space);

  // Create extensible dataset for time values
  hsize_t steps_initial_dims[1] = { 0 };
  hsize_t steps_max_dims[1] = { H5S_UNLIMITED };
  hsize_t steps_chunk_dims[1] = { 1 };

  hid_t steps_plist = H5Pcreate(H5P_DATASET_CREATE);
  H5Pset_chunk(steps_plist, 1, steps_chunk_dims);

  hid_t steps_dataspace =
      H5Screate_simple(1, steps_initial_dims, steps_max_dims);
  dataset = H5Dcreate(steps_group, "Values", H5T_NATIVE_DOUBLE, steps_dataspace,
                      H5P_DEFAULT, steps_plist, H5P_DEFAULT);
  H5Dclose(dataset);
  H5Sclose(steps_dataspace);

  // Create extensible datasets for NumberOfParts and offset arrays
  steps_dataspace = H5Screate_simple(1, steps_initial_dims, steps_max_dims);
  dataset = H5Dcreate(steps_group, "NumberOfParts", H5T_NATIVE_LLONG,
                      steps_dataspace, H5P_DEFAULT, steps_plist, H5P_DEFAULT);
  H5Dclose(dataset);
  H5Sclose(steps_dataspace);

  steps_dataspace = H5Screate_simple(1, steps_initial_dims, steps_max_dims);
  dataset = H5Dcreate(steps_group, "PartOffsets", H5T_NATIVE_LLONG,
                      steps_dataspace, H5P_DEFAULT, steps_plist, H5P_DEFAULT);
  H5Dclose(dataset);
  H5Sclose(steps_dataspace);

  steps_dataspace = H5Screate_simple(1, steps_initial_dims, steps_max_dims);
  dataset = H5Dcreate(steps_group, "PointOffsets", H5T_NATIVE_LLONG,
                      steps_dataspace, H5P_DEFAULT, steps_plist, H5P_DEFAULT);
  H5Dclose(dataset);
  H5Sclose(steps_dataspace);

  steps_dataspace = H5Screate_simple(1, steps_initial_dims, steps_max_dims);
  dataset = H5Dcreate(steps_group, "CellOffsets", H5T_NATIVE_LLONG,
                      steps_dataspace, H5P_DEFAULT, steps_plist, H5P_DEFAULT);
  H5Dclose(dataset);
  H5Sclose(steps_dataspace);

  steps_dataspace = H5Screate_simple(1, steps_initial_dims, steps_max_dims);
  dataset = H5Dcreate(steps_group, "ConnectivityIdOffsets", H5T_NATIVE_LLONG,
                      steps_dataspace, H5P_DEFAULT, steps_plist, H5P_DEFAULT);
  H5Dclose(dataset);
  H5Sclose(steps_dataspace);

  // Create PointDataOffsets subgroup with extensible datasets
  hid_t pd_offsets_group = H5Gcreate(steps_group, "PointDataOffsets",
                                     H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  // Wavefield offsets (extensible)
  steps_dataspace = H5Screate_simple(1, steps_initial_dims, steps_max_dims);
  dataset = H5Dcreate(pd_offsets_group, "Wavefield", H5T_NATIVE_LLONG,
                      steps_dataspace, H5P_DEFAULT, steps_plist, H5P_DEFAULT);
  H5Dclose(dataset);
  H5Sclose(steps_dataspace);
  H5Gclose(pd_offsets_group);

  H5Pclose(steps_plist);
  H5Gclose(steps_group);

  // Close HDF5 file - will reopen for each timestep write
  H5Gclose(vtkhdf_group);
  H5Fclose(hdf5_file_id);

  plotter.mpi->cout("Initialized VTK HDF5 file for wavefield output: " +
                    plotter.hdf5_filename + " (extensible datasets)");

#else
  throw std::runtime_error(
      "VTK HDF5 output requested but HDF5 support not compiled.");
#endif
}

void initialize_display(specfem::periodic_tasks::plot_wavefield &plotter,
                        vtkSmartPointer<vtkFloatArray> &scalars) {

  // Create VTK objects that will persist between calls
  plotter.colors = vtkSmartPointer<vtkNamedColors>::New();

  // Create material mapper and actor
  plotter.material_mapper = plotter.map_materials_with_color();
  plotter.material_actor = vtkSmartPointer<vtkActor>::New();
  plotter.material_actor->SetMapper(plotter.material_mapper);

  // Create lookup table
  plotter.lut = vtkSmartPointer<vtkLookupTable>::New();
  plotter.lut->SetNumberOfTableValues(256);
  plotter.lut->Build();

  // Create a mapper for the wavefield
  plotter.wavefield_mapper = vtkSmartPointer<vtkDataSetMapper>::New();
  plotter.wavefield_mapper->SetInputData(plotter.unstructured_grid);
  plotter.wavefield_mapper->SetLookupTable(plotter.lut);
  plotter.wavefield_mapper->SetScalarModeToUsePointData();
  plotter.wavefield_mapper->SetColorModeToMapScalars();
  plotter.wavefield_mapper->SetScalarVisibility(1);

  // Set the range of the lookup table
  double range[2];
  scalars->GetRange(range);
  plotter.wavefield_mapper->SetScalarRange(range[0], range[1]);
  plotter.lut->SetRange(range[0], range[1]);

  // set color gradient from white to black
  for (int i = 0; i < 256; ++i) {
    double t = static_cast<double>(i) / 255.0;
    double transparency = plotter.sigmoid(t);
    plotter.lut->SetTableValue(i, 1.0 - t, 1.0 - t, 1.0 - t, transparency);
  }

  // Create an actor
  auto wavefield_actor = vtkSmartPointer<vtkActor>::New();
  wavefield_actor->SetMapper(plotter.wavefield_mapper);

  // Create renderer
  plotter.renderer = vtkSmartPointer<vtkRenderer>::New();
  plotter.renderer->AddActor(plotter.material_actor);
  plotter.renderer->AddActor(wavefield_actor);
  plotter.renderer->SetBackground(
      plotter.colors->GetColor3d("White").GetData());

  // Plot edges
  if (false) {
    // Create edges extractor and actors
    vtkSmartPointer<vtkExtractEdges> edges =
        vtkSmartPointer<vtkExtractEdges>::New();
    edges->SetInputData(plotter.unstructured_grid);
    edges->Update();

    vtkSmartPointer<vtkPolyDataMapper> outlineMapper =
        vtkSmartPointer<vtkPolyDataMapper>::New();
    outlineMapper->SetInputConnection(edges->GetOutputPort());
    outlineMapper->ScalarVisibilityOff();

    plotter.outlineActor = vtkSmartPointer<vtkActor>::New();
    plotter.outlineActor->SetMapper(outlineMapper);
    plotter.outlineActor->GetProperty()->SetColor(
        plotter.colors->GetColor3d("Black").GetData());
    plotter.outlineActor->GetProperty()->SetLineWidth(0.5);

    plotter.renderer->AddActor(plotter.outlineActor);
  }

  // Create render window
  plotter.render_window = vtkSmartPointer<vtkRenderWindow>::New();
  plotter.render_window->AddRenderer(plotter.renderer);
  plotter.render_window->SetSize(2560, 2560);
  plotter.render_window->SetWindowName("Wavefield");
}

template <>
void initialize<specfem::display::format::on_screen>(
    specfem::periodic_tasks::plot_wavefield &plotter,
    vtkSmartPointer<vtkFloatArray> &scalars) {

  ::initialize_display(plotter, scalars);

  // Create render window interactor
  plotter.render_window_interactor =
      vtkSmartPointer<vtkRenderWindowInteractor>::New();
  plotter.render_window_interactor->SetRenderWindow(plotter.render_window);
}

template <>
void initialize<specfem::display::format::PNG>(
    specfem::periodic_tasks::plot_wavefield &plotter,
    vtkSmartPointer<vtkFloatArray> &scalars) {

  ::initialize_display(plotter, scalars);

  // Set off screen rendering
  vtkSmartPointer<vtkGraphicsFactory> graphics_factory;
  graphics_factory->SetOffScreenOnlyMode(1);
  graphics_factory->SetUseMesaClasses(1);
  plotter.render_window->SetOffScreenRendering(1);
}

template <>
void initialize<specfem::display::format::JPG>(
    specfem::periodic_tasks::plot_wavefield &plotter,
    vtkSmartPointer<vtkFloatArray> &scalars) {

  ::initialize_display(plotter, scalars);

  // Set off screen rendering
  vtkSmartPointer<vtkGraphicsFactory> graphics_factory;
  graphics_factory->SetOffScreenOnlyMode(1);
  graphics_factory->SetUseMesaClasses(1);
  plotter.render_window->SetOffScreenRendering(1);
}

void specfem::periodic_tasks::plot_wavefield::initialize(
    specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly) {

  // Create the grid structure
  create_quad_grid(); // or create_biquad_grid() based on preference

  // Compute initial wavefield scalars and add to grid
  auto scalars = compute_wavefield_scalars(assembly);
  unstructured_grid->GetPointData()->SetScalars(scalars);

  switch (output_format) {
  case specfem::display::format::vtkhdf:
    ::initialize<specfem::display::format::vtkhdf>(*this, scalars);
    break;
  case specfem::display::format::on_screen:
    ::initialize<specfem::display::format::on_screen>(*this, scalars);
    break;
  case specfem::display::format::PNG:
    ::initialize<specfem::display::format::PNG>(*this, scalars);
    break;
  case specfem::display::format::JPG:
    ::initialize<specfem::display::format::JPG>(*this, scalars);
    break;
  default:
    throw std::runtime_error("Unsupported display format");
  }

  return;
}

void run_render(specfem::periodic_tasks::plot_wavefield &plotter,
                vtkSmartPointer<vtkFloatArray> &scalars) {
  // Get range of scalar values
  double range[2];
  scalars->GetRange(range);
  plotter.wavefield_mapper->SetScalarRange(range[0], range[1]);

  // Update lookup table range
  plotter.lut->SetRange(range[0], range[1]);
  plotter.lut->Build();

  // Render
  plotter.render_window->Render();
};

template <specfem::display::format format>
void run(specfem::periodic_tasks::plot_wavefield &plotter,
         vtkSmartPointer<vtkFloatArray> &scalars, const int istep);

template <>
void run<specfem::display::format::on_screen>(
    specfem::periodic_tasks::plot_wavefield &plotter,
    vtkSmartPointer<vtkFloatArray> &scalars, const int istep) {
  run_render(plotter, scalars);
}

template <>
void run<specfem::display::format::PNG>(
    specfem::periodic_tasks::plot_wavefield &plotter,
    vtkSmartPointer<vtkFloatArray> &scalars, const int istep) {

  // Render the field
  run_render(plotter, scalars);

  // create image filter
  auto image_filter = vtkSmartPointer<vtkWindowToImageFilter>::New();
  image_filter->SetInput(plotter.render_window);
  image_filter->Update();

  const auto filename =
      plotter.output_folder /
      ("wavefield" + specfem::utilities::to_zero_lead(istep, 6) + ".png");
  auto writer = vtkSmartPointer<vtkPNGWriter>::New();
  writer->SetFileName(filename.string().c_str());
  writer->SetInputConnection(image_filter->GetOutputPort());
  writer->Write();
  std::string message = "Wrote wavefield image to " + filename.string();
  plotter.mpi->cout(message);
}

template <>
void run<specfem::display::format::JPG>(
    specfem::periodic_tasks::plot_wavefield &plotter,
    vtkSmartPointer<vtkFloatArray> &scalars, const int istep) {

  auto image_filter = vtkSmartPointer<vtkWindowToImageFilter>::New();
  image_filter->SetInput(plotter.render_window);
  image_filter->Update();

  const auto filename =
      plotter.output_folder /
      ("wavefield" + specfem::utilities::to_zero_lead(istep, 6) + ".jpg");
  auto writer = vtkSmartPointer<vtkJPEGWriter>::New();
  writer->SetFileName(filename.string().c_str());
  writer->SetInputConnection(image_filter->GetOutputPort());
  writer->Write();
  std::string message = "Wrote wavefield image to " + filename.string();
  plotter.mpi->cout(message);
}

template <>
void run<specfem::display::format::vtkhdf>(
    specfem::periodic_tasks::plot_wavefield &plotter,
    vtkSmartPointer<vtkFloatArray> &scalars, const int istep) {

#ifndef NO_HDF5
  // Open HDF5 file for extending datasets
  hid_t hdf5_file_id =
      H5Fopen(plotter.hdf5_filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  hid_t vtkhdf_group = H5Gopen(hdf5_file_id, "/VTKHDF", H5P_DEFAULT);

  // Extend and write wavefield data
  hid_t pd_group = H5Gopen(vtkhdf_group, "PointData", H5P_DEFAULT);
  hid_t pd_dataset = H5Dopen(pd_group, "Wavefield", H5P_DEFAULT);

  // Extend the wavefield dataset to accommodate new timestep
  hsize_t new_size[1] = { (hsize_t)((plotter.current_timestep + 1) *
                                    plotter.numPoints) };
  H5Dset_extent(pd_dataset, new_size);

  // Write wavefield data for this timestep
  std::vector<float> scalar_data(plotter.numPoints);
  for (int i = 0; i < plotter.numPoints; i++) {
    scalar_data[i] = scalars->GetValue(i);
  }

  // Calculate offset and count for this timestep
  hsize_t offset = plotter.current_timestep * plotter.numPoints;
  hsize_t count = plotter.numPoints;

  // Select hyperslab in the file dataset
  hid_t filespace = H5Dget_space(pd_dataset);
  H5Sselect_hyperslab(filespace, H5S_SELECT_SET, &offset, NULL, &count, NULL);

  // Create memory dataspace
  hid_t memspace = H5Screate_simple(1, &count, NULL);

  // Write data to dataset
  H5Dwrite(pd_dataset, H5T_NATIVE_FLOAT, memspace, filespace, H5P_DEFAULT,
           scalar_data.data());

  H5Sclose(memspace);
  H5Sclose(filespace);
  H5Dclose(pd_dataset);
  H5Gclose(pd_group);

  // Update temporal metadata arrays - extend all arrays
  hsize_t new_timestep_count[1] = { (hsize_t)(plotter.current_timestep + 1) };

  // Extend and update NumberOfPoints array
  hid_t np_dataset = H5Dopen(vtkhdf_group, "NumberOfPoints", H5P_DEFAULT);
  H5Dset_extent(np_dataset, new_timestep_count);
  hsize_t ts_offset = plotter.current_timestep;
  hsize_t ts_count = 1;
  hid_t np_filespace = H5Dget_space(np_dataset);
  H5Sselect_hyperslab(np_filespace, H5S_SELECT_SET, &ts_offset, NULL, &ts_count,
                      NULL);
  hid_t np_memspace = H5Screate_simple(1, &ts_count, NULL);
  long long numPoints = plotter.numPoints;
  H5Dwrite(np_dataset, H5T_NATIVE_LLONG, np_memspace, np_filespace, H5P_DEFAULT,
           &numPoints);
  H5Sclose(np_memspace);
  H5Sclose(np_filespace);
  H5Dclose(np_dataset);

  // Extend and update NumberOfCells array
  hid_t nc_dataset = H5Dopen(vtkhdf_group, "NumberOfCells", H5P_DEFAULT);
  H5Dset_extent(nc_dataset, new_timestep_count);
  hid_t nc_filespace = H5Dget_space(nc_dataset);
  H5Sselect_hyperslab(nc_filespace, H5S_SELECT_SET, &ts_offset, NULL, &ts_count,
                      NULL);
  hid_t nc_memspace = H5Screate_simple(1, &ts_count, NULL);
  long long numCells = plotter.numCells;
  H5Dwrite(nc_dataset, H5T_NATIVE_LLONG, nc_memspace, nc_filespace, H5P_DEFAULT,
           &numCells);
  H5Sclose(nc_memspace);
  H5Sclose(nc_filespace);
  H5Dclose(nc_dataset);

  // Extend and update NumberOfConnectivityIds array
  hid_t nci_dataset =
      H5Dopen(vtkhdf_group, "NumberOfConnectivityIds", H5P_DEFAULT);
  H5Dset_extent(nci_dataset, new_timestep_count);
  hid_t nci_filespace = H5Dget_space(nci_dataset);
  H5Sselect_hyperslab(nci_filespace, H5S_SELECT_SET, &ts_offset, NULL,
                      &ts_count, NULL);
  hid_t nci_memspace = H5Screate_simple(1, &ts_count, NULL);
  long long numConnIds = plotter.numConnectivityIds;
  H5Dwrite(nci_dataset, H5T_NATIVE_LLONG, nci_memspace, nci_filespace,
           H5P_DEFAULT, &numConnIds);
  H5Sclose(nci_memspace);
  H5Sclose(nci_filespace);
  H5Dclose(nci_dataset);

  // Update Steps metadata
  hid_t steps_group = H5Gopen(vtkhdf_group, "Steps", H5P_DEFAULT);

  // Extend and write time value for this timestep
  double time_value = static_cast<double>(istep) * plotter.dt;
  hid_t values_dataset = H5Dopen(steps_group, "Values", H5P_DEFAULT);
  H5Dset_extent(values_dataset, new_timestep_count);
  hid_t values_filespace = H5Dget_space(values_dataset);
  H5Sselect_hyperslab(values_filespace, H5S_SELECT_SET, &ts_offset, NULL,
                      &ts_count, NULL);
  hid_t values_memspace = H5Screate_simple(1, &ts_count, NULL);
  H5Dwrite(values_dataset, H5T_NATIVE_DOUBLE, values_memspace, values_filespace,
           H5P_DEFAULT, &time_value);
  H5Sclose(values_memspace);
  H5Sclose(values_filespace);
  H5Dclose(values_dataset);

  // Extend and update NumberOfParts (always 1)
  hid_t nparts_dataset = H5Dopen(steps_group, "NumberOfParts", H5P_DEFAULT);
  H5Dset_extent(nparts_dataset, new_timestep_count);
  hid_t nparts_filespace = H5Dget_space(nparts_dataset);
  H5Sselect_hyperslab(nparts_filespace, H5S_SELECT_SET, &ts_offset, NULL,
                      &ts_count, NULL);
  hid_t nparts_memspace = H5Screate_simple(1, &ts_count, NULL);
  long long numParts = 1;
  H5Dwrite(nparts_dataset, H5T_NATIVE_LLONG, nparts_memspace, nparts_filespace,
           H5P_DEFAULT, &numParts);
  H5Sclose(nparts_memspace);
  H5Sclose(nparts_filespace);
  H5Dclose(nparts_dataset);

  // Extend and update offset arrays (all zeros for static geometry/single part)
  long long zeroOffset = 0;

  // PartOffsets
  hid_t part_offsets_dataset = H5Dopen(steps_group, "PartOffsets", H5P_DEFAULT);
  H5Dset_extent(part_offsets_dataset, new_timestep_count);
  hid_t part_offsets_filespace = H5Dget_space(part_offsets_dataset);
  H5Sselect_hyperslab(part_offsets_filespace, H5S_SELECT_SET, &ts_offset, NULL,
                      &ts_count, NULL);
  hid_t part_offsets_memspace = H5Screate_simple(1, &ts_count, NULL);
  H5Dwrite(part_offsets_dataset, H5T_NATIVE_LLONG, part_offsets_memspace,
           part_offsets_filespace, H5P_DEFAULT, &zeroOffset);
  H5Sclose(part_offsets_memspace);
  H5Sclose(part_offsets_filespace);
  H5Dclose(part_offsets_dataset);

  // PointOffsets
  hid_t point_offsets_dataset =
      H5Dopen(steps_group, "PointOffsets", H5P_DEFAULT);
  H5Dset_extent(point_offsets_dataset, new_timestep_count);
  hid_t point_offsets_filespace = H5Dget_space(point_offsets_dataset);
  H5Sselect_hyperslab(point_offsets_filespace, H5S_SELECT_SET, &ts_offset, NULL,
                      &ts_count, NULL);
  hid_t point_offsets_memspace = H5Screate_simple(1, &ts_count, NULL);
  H5Dwrite(point_offsets_dataset, H5T_NATIVE_LLONG, point_offsets_memspace,
           point_offsets_filespace, H5P_DEFAULT, &zeroOffset);
  H5Sclose(point_offsets_memspace);
  H5Sclose(point_offsets_filespace);
  H5Dclose(point_offsets_dataset);

  // CellOffsets
  hid_t cell_offsets_dataset = H5Dopen(steps_group, "CellOffsets", H5P_DEFAULT);
  H5Dset_extent(cell_offsets_dataset, new_timestep_count);
  hid_t cell_offsets_filespace = H5Dget_space(cell_offsets_dataset);
  H5Sselect_hyperslab(cell_offsets_filespace, H5S_SELECT_SET, &ts_offset, NULL,
                      &ts_count, NULL);
  hid_t cell_offsets_memspace = H5Screate_simple(1, &ts_count, NULL);
  H5Dwrite(cell_offsets_dataset, H5T_NATIVE_LLONG, cell_offsets_memspace,
           cell_offsets_filespace, H5P_DEFAULT, &zeroOffset);
  H5Sclose(cell_offsets_memspace);
  H5Sclose(cell_offsets_filespace);
  H5Dclose(cell_offsets_dataset);

  // ConnectivityIdOffsets
  hid_t conn_offsets_dataset =
      H5Dopen(steps_group, "ConnectivityIdOffsets", H5P_DEFAULT);
  H5Dset_extent(conn_offsets_dataset, new_timestep_count);
  hid_t conn_offsets_filespace = H5Dget_space(conn_offsets_dataset);
  H5Sselect_hyperslab(conn_offsets_filespace, H5S_SELECT_SET, &ts_offset, NULL,
                      &ts_count, NULL);
  hid_t conn_offsets_memspace = H5Screate_simple(1, &ts_count, NULL);
  H5Dwrite(conn_offsets_dataset, H5T_NATIVE_LLONG, conn_offsets_memspace,
           conn_offsets_filespace, H5P_DEFAULT, &zeroOffset);
  H5Sclose(conn_offsets_memspace);
  H5Sclose(conn_offsets_filespace);
  H5Dclose(conn_offsets_dataset);

  // Update PointDataOffsets/Wavefield
  hid_t pd_offsets_group =
      H5Gopen(steps_group, "PointDataOffsets", H5P_DEFAULT);
  hid_t wf_offsets_dataset =
      H5Dopen(pd_offsets_group, "Wavefield", H5P_DEFAULT);
  H5Dset_extent(wf_offsets_dataset, new_timestep_count);
  hid_t wf_offsets_filespace = H5Dget_space(wf_offsets_dataset);
  H5Sselect_hyperslab(wf_offsets_filespace, H5S_SELECT_SET, &ts_offset, NULL,
                      &ts_count, NULL);
  hid_t wf_offsets_memspace = H5Screate_simple(1, &ts_count, NULL);
  long long wavefieldOffset = plotter.current_timestep * plotter.numPoints;
  H5Dwrite(wf_offsets_dataset, H5T_NATIVE_LLONG, wf_offsets_memspace,
           wf_offsets_filespace, H5P_DEFAULT, &wavefieldOffset);
  H5Sclose(wf_offsets_memspace);
  H5Sclose(wf_offsets_filespace);
  H5Dclose(wf_offsets_dataset);
  H5Gclose(pd_offsets_group);

  // Update NSteps attribute
  int nsteps_written = plotter.current_timestep + 1;
  hid_t attr = H5Aopen(steps_group, "NSteps", H5P_DEFAULT);
  H5Awrite(attr, H5T_NATIVE_INT, &nsteps_written);
  H5Aclose(attr);

  H5Gclose(steps_group);

  // Close HDF5 resources
  H5Gclose(vtkhdf_group);
  H5Fclose(hdf5_file_id);

  plotter.current_timestep++;

  plotter.mpi->cout("Wrote wavefield data for timestep " +
                    std::to_string(istep) + " to HDF5 file (step " +
                    std::to_string(plotter.current_timestep) + ")");

#else
  throw std::runtime_error(
      "VTK HDF5 output requested but HDF5 support not compiled.");
#endif
}

void specfem::periodic_tasks::plot_wavefield::run(
    specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly,
    const int istep) {

  // Update the wavefield scalars only
  auto scalars = compute_wavefield_scalars(assembly);
  unstructured_grid->GetPointData()->SetScalars(scalars);

  switch (output_format) {
  case (display::format::vtkhdf):

    ::run<specfem::display::format::vtkhdf>(*this, scalars, istep);
    break;

  case (display::format::on_screen):
    ::run<specfem::display::format::on_screen>(*this, scalars, istep);
    break;

  case (display::format::PNG):

    ::run<specfem::display::format::PNG>(*this, scalars, istep);
    break;

  case (display::format::JPG):
    ::run<specfem::display::format::JPG>(*this, scalars, istep);
    break;

  default:
    throw std::runtime_error("Unsupported output format");
  }
}

void specfem::periodic_tasks::plot_wavefield::finalize(
    specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly) {

  // If interactive, start the event loop if it hasn't been started
  if (output_format == specfem::display::format::on_screen &&
      render_window_interactor) {
    render_window_interactor->Start();
  }

  // Clean up VTK objects
  renderer = nullptr;
  render_window = nullptr;
  render_window_interactor = nullptr;
  material_actor = nullptr;
  actor = nullptr;
  outlineActor = nullptr;
  material_mapper = nullptr;
  unstructured_grid = nullptr;
  lut = nullptr;
  colors = nullptr;
}

#endif // NO_VTK
