#include "io/mesh/impl/fortran/dim3/meshfem3d/read_materials.hpp"
#include "mesh/mesh.hpp"

std::tuple<Kokkos::View<int **, Kokkos::LayoutLeft, Kokkos::HostSpace>,
           specfem::mesh::meshfem3d::Materials<specfem::dimension::type::dim3> >
specfem::io::mesh::impl::fortran::dim3::meshfem3d::read_materials(
    std::ifstream &stream, int ngnod, const specfem::MPI::MPI *mpi) {

  using MaterialsType =
      specfem::mesh::meshfem3d::Materials<specfem::dimension::type::dim3>;

  MaterialsType materials;

  // TODO (Rohit : TOMOGRAPHIC_MATERIALS)
  // We are currently not reading undefined materials which use tomographic
  // models. Add support for reading these materials later.
  int num_materials, num_undefined_materials;

  specfem::io::read_fortran_line(stream, &num_materials,
                                 &num_undefined_materials);

  std::vector<typename MaterialsType::material_specification> mapping;

  for (int imat = 0; imat < num_materials; ++imat) {
    std::vector<type_real> material_properties(17, 0.0);
    specfem::io::read_fortran_line(stream, &material_properties);

    const int material_id = static_cast<int>(material_properties[6]);
    switch (material_id) {
    case 1: // Elastic or Acoustic material
    {
      const type_real rho = material_properties[0];
      const type_real vp = material_properties[1];
      const type_real vs = material_properties[2];
      const type_real Qkappa = material_properties[3];
      const type_real Qmu = material_properties[4];
      const int is_anisotropic = static_cast<int>(material_properties[5]);
      if (is_anisotropic <= 0) {
        if (specfem::utilities::is_close(vs, 0.0)) {
          // Acoustic material
          specfem::material::material<specfem::element::medium_tag::acoustic,
                                      specfem::element::property_tag::isotropic>
              material(rho, vp, Qkappa);
          const int index = materials.add_material(material, imat);
          mapping.push_back({ specfem::element::medium_tag::acoustic,
                              specfem::element::property_tag::isotropic, index,
                              imat });
        } else if (vs > 0.0) {
          // Isotropic elastic material
          specfem::material::material<specfem::element::medium_tag::elastic,
                                      specfem::element::property_tag::isotropic>
              material(rho, vp, vs, Qkappa, Qmu);
          const int index = materials.add_material(material, imat);
          mapping.push_back({ specfem::element::medium_tag::elastic,
                              specfem::element::property_tag::isotropic, index,
                              imat });

        } else {
          throw std::runtime_error(
              "Shear wave velocity (Vs) cannot be negative for elastic "
              "materials.");
        }
      } else {
        // Anisotropic elastic material
        // TODO (Rohit: ANISOTROPIC_MATERIALS): Add support for anisotropic
        // materials
        throw std::runtime_error("Anisotropic elastic materials are not "
                                 "supported yet for 3D simulations.");
      }
      break;
    }
    case 2: {
      // Poroelastic material
      // TODO (Rohit: POROELASTIC_MATERIALS): Add support for poroelastic
      // materials
      throw std::runtime_error(
          "Poroelastic materials are not supported yet for 3D simulations.");
      break;
    }
    default:
      throw std::runtime_error("Unknown material ID: " +
                               std::to_string(material_id));
    }
  }

  // TODO (Rohit: TOMOGRAPHIC_MATERIALS): Add support for reading tomographic
  // materials
  for (int imat = 0; imat < num_undefined_materials; ++imat) {
    std::vector<type_real> dummy(6);
    specfem::io::read_fortran_line(stream, &dummy);
  }

  int nspec;
  specfem::io::read_fortran_line(stream, &nspec);
  Kokkos::View<int **, Kokkos::LayoutLeft, Kokkos::HostSpace>
      control_node_index("specfem::mesh::control_node_index", nspec, ngnod);

  materials.material_index_mapping.resize(nspec);
  for (int ispec = 0; ispec < nspec; ++ispec) {
    int index;
    int database_index;
    int tomographic_model;
    std::vector<int> control_nodes(ngnod, 0);
    specfem::io::read_fortran_line(stream, &index, &database_index,
                                   &tomographic_model, &control_nodes);
    if (index < 1 || index > nspec) {
      throw std::runtime_error("Error reading material indices");
    }
    if (database_index < 1 || database_index > num_materials) {
      throw std::runtime_error("Error reading material indices");
    }
    if (tomographic_model == 1) {
      // Deprecated funcitionality within MESHFEM3D
      throw std::runtime_error(
          "Interfaces are deprecated within 3D simulations.");
    }
    if (tomographic_model == 2) {
      // TODO (Rohit: TOMOGRAPHIC_MATERIALS): Add support for reading
      // tomographic materials
      throw std::runtime_error(
          "Tomographic materials are not supported yet for 3D simulations.");
    }
    materials.material_index_mapping[index - 1] = mapping[database_index - 1];
    for (int inode = 0; inode < ngnod; ++inode) {
      control_node_index(index - 1, inode) = control_nodes[inode];
    }
  }

  return std::make_tuple(control_node_index, materials);
}
