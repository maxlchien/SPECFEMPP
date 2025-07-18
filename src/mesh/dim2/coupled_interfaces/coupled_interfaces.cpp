#include "mesh/dim2/coupled_interfaces/coupled_interfaces.hpp"
#include "mesh/dim2/coupled_interfaces/interface_container.hpp"
#include "mesh/dim2/coupled_interfaces/interface_container.tpp"
#include "mesh/mesh_base.hpp"

// specfem::mesh::coupled_interfaces::coupled_interfaces(
//     specfem::mesh::interface_container<specfem::element::medium_tag::elastic_psv,
//                                        specfem::element::medium_tag::acoustic>
//         elastic_acoustic,
//     specfem::mesh::interface_container<
//         specfem::element::medium_tag::acoustic,
//         specfem::element::medium_tag::poroelastic>
//         acoustic_poroelastic,
//     specfem::mesh::interface_container<
//         specfem::element::medium_tag::elastic_psv,
//         specfem::element::medium_tag::poroelastic>
//         elastic_poroelastic) {}

template <specfem::element::medium_tag Medium1,
          specfem::element::medium_tag Medium2>
std::variant<
    specfem::mesh::interface_container<
        specfem::dimension::type::dim2,
        specfem::element::medium_tag::elastic_psv,
        specfem::element::medium_tag::acoustic>,
    specfem::mesh::interface_container<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        specfem::element::medium_tag::poroelastic>,
    specfem::mesh::interface_container<
        specfem::dimension::type::dim2,
        specfem::element::medium_tag::elastic_psv,
        specfem::element::medium_tag::poroelastic> >
specfem::mesh::coupled_interfaces<
    specfem::dimension::type::dim2>::coupled_interfaces::get() const {
  if constexpr (Medium1 == specfem::element::medium_tag::elastic_psv &&
                Medium2 == specfem::element::medium_tag::acoustic) {
    return elastic_acoustic;
  } else if constexpr (Medium1 == specfem::element::medium_tag::acoustic &&
                       Medium2 == specfem::element::medium_tag::poroelastic) {
    return acoustic_poroelastic;
  } else if constexpr (Medium1 == specfem::element::medium_tag::elastic_psv &&
                       Medium2 == specfem::element::medium_tag::poroelastic) {
    return elastic_poroelastic;
  }
}

// Explicitly instantiate template member function
template int
specfem::mesh::interface_container<specfem::dimension::type::dim2,
                                   specfem::element::medium_tag::elastic_psv,
                                   specfem::element::medium_tag::acoustic>::
    get_spectral_elem_index<specfem::element::medium_tag::elastic_psv>(
        const int interface_index) const;

template int
specfem::mesh::interface_container<specfem::dimension::type::dim2,
                                   specfem::element::medium_tag::elastic_psv,
                                   specfem::element::medium_tag::acoustic>::
    get_spectral_elem_index<specfem::element::medium_tag::acoustic>(
        const int interface_index) const;

template int
specfem::mesh::interface_container<specfem::dimension::type::dim2,
                                   specfem::element::medium_tag::acoustic,
                                   specfem::element::medium_tag::poroelastic>::
    get_spectral_elem_index<specfem::element::medium_tag::acoustic>(
        const int interface_index) const;

template int
specfem::mesh::interface_container<specfem::dimension::type::dim2,
                                   specfem::element::medium_tag::acoustic,
                                   specfem::element::medium_tag::poroelastic>::
    get_spectral_elem_index<specfem::element::medium_tag::poroelastic>(
        const int interface_index) const;

template int
specfem::mesh::interface_container<specfem::dimension::type::dim2,
                                   specfem::element::medium_tag::elastic_psv,
                                   specfem::element::medium_tag::poroelastic>::
    get_spectral_elem_index<specfem::element::medium_tag::elastic_psv>(
        const int interface_index) const;

template int
specfem::mesh::interface_container<specfem::dimension::type::dim2,
                                   specfem::element::medium_tag::elastic_psv,
                                   specfem::element::medium_tag::poroelastic>::
    get_spectral_elem_index<specfem::element::medium_tag::poroelastic>(
        const int interface_index) const;

// Explicitly instantiate template member function
template std::variant<
    specfem::mesh::interface_container<
        specfem::dimension::type::dim2,
        specfem::element::medium_tag::elastic_psv,
        specfem::element::medium_tag::acoustic>,
    specfem::mesh::interface_container<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        specfem::element::medium_tag::poroelastic>,
    specfem::mesh::interface_container<
        specfem::dimension::type::dim2,
        specfem::element::medium_tag::elastic_psv,
        specfem::element::medium_tag::poroelastic> >
specfem::mesh::coupled_interfaces<specfem::dimension::type::dim2>::
    coupled_interfaces::get<specfem::element::medium_tag::elastic_psv,
                            specfem::element::medium_tag::acoustic>() const;

template std::variant<
    specfem::mesh::interface_container<
        specfem::dimension::type::dim2,
        specfem::element::medium_tag::elastic_psv,
        specfem::element::medium_tag::acoustic>,
    specfem::mesh::interface_container<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        specfem::element::medium_tag::poroelastic>,
    specfem::mesh::interface_container<
        specfem::dimension::type::dim2,
        specfem::element::medium_tag::elastic_psv,
        specfem::element::medium_tag::poroelastic> >
specfem::mesh::coupled_interfaces<specfem::dimension::type::dim2>::
    coupled_interfaces::get<specfem::element::medium_tag::acoustic,
                            specfem::element::medium_tag::poroelastic>() const;

template std::variant<
    specfem::mesh::interface_container<
        specfem::dimension::type::dim2,
        specfem::element::medium_tag::elastic_psv,
        specfem::element::medium_tag::acoustic>,
    specfem::mesh::interface_container<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        specfem::element::medium_tag::poroelastic>,
    specfem::mesh::interface_container<
        specfem::dimension::type::dim2,
        specfem::element::medium_tag::elastic_psv,
        specfem::element::medium_tag::poroelastic> >
specfem::mesh::coupled_interfaces<specfem::dimension::type::dim2>::
    coupled_interfaces::get<specfem::element::medium_tag::elastic_psv,
                            specfem::element::medium_tag::poroelastic>() const;
