import itertools
from typing import Literal

from _gmsh2meshfem.gmsh_dep import GmshContext
from _gmsh2meshfem.dim2.model import Model

from .layer import Layer, LayerBoundary


class LayeredBuilder:
    """Generates a layer topography domain in 2D, spanning from x=xlow to x=xhigh.
    Each layer `layers[i]` is bounded below by `boundaries[i]` and above by `boundaries[i+1]`.
    """

    xlow: float
    xhigh: float

    boundaries: list[LayerBoundary]
    layers: list[Layer]

    domain_boundary_type_top: Literal["neumann", "acoustic_free_surface", "absorbing"]
    domain_boundary_type_bottom: Literal[
        "neumann", "acoustic_free_surface", "absorbing"
    ]
    domain_boundary_type_left: Literal["neumann", "acoustic_free_surface", "absorbing"]
    domain_boundary_type_right: Literal["neumann", "acoustic_free_surface", "absorbing"]

    @property
    def width(self):
        return self.xhigh - self.xlow

    def __init__(
        self,
        xlow: float,
        xhigh: float,
        set_left_boundary: Literal[
            "neumann", "acoustic_free_surface", "absorbing"
        ] = "neumann",
        set_right_boundary: Literal[
            "neumann", "acoustic_free_surface", "absorbing"
        ] = "neumann",
        set_top_boundary: Literal[
            "neumann", "acoustic_free_surface", "absorbing"
        ] = "neumann",
        set_bottom_boundary: Literal[
            "neumann", "acoustic_free_surface", "absorbing"
        ] = "neumann",
    ):
        self.xlow = xlow
        self.xhigh = xhigh
        self.layers = []
        self.boundaries = []
        self.domain_boundary_type_top = set_top_boundary
        self.domain_boundary_type_bottom = set_bottom_boundary
        self.domain_boundary_type_left = set_left_boundary
        self.domain_boundary_type_right = set_right_boundary

    def create_model(self) -> Model:
        with GmshContext() as gmsh:
            built_layerbds = [
                bdlayer.build_layer(self.xlow, self.xhigh, gmsh=gmsh)
                for bdlayer in self.boundaries
            ]
            for ilayer, layerbd in enumerate(built_layerbds):
                layerbd.initialize_curve_copy(
                    None if ilayer == 0 else self.layers[ilayer - 1],
                    None if ilayer == len(self.layers) else self.layers[-1],
                    gmsh,
                )

            # store tags
            surfaces = []
            left_walls = []
            right_walls = []
            for i, (l0, l1) in enumerate(itertools.pairwise(built_layerbds)):
                layer_result = self.layers[i].generate_layer(l0, l1, gmsh)
                surfaces.append(layer_result.surface_index)
                left_walls.append(layer_result.left_wall_index)
                right_walls.append(layer_result.right_wall_index)

            # physical groups in model space. must sync with geo
            gmsh.model.geo.synchronize()

            left_tag = gmsh.model.add_physical_group(
                1, left_walls, name="left_boundary"
            )
            right_tag = gmsh.model.add_physical_group(
                1, right_walls, name="right_boundary"
            )
            bottom_tag = gmsh.model.add_physical_group(
                1, [built_layerbds[0].curve], name="bottom_boundary"
            )
            top_tag = gmsh.model.add_physical_group(
                1, [built_layerbds[-1].curve_copy], name="top_boundary"
            )

            bdry_neumann = []
            bdry_afs = []
            bdry_abs = []

            bdry_by_name = {
                "neumann": bdry_neumann,
                "acoustic_free_surface": bdry_afs,
                "absorbing": bdry_abs,
            }
            bdry_by_name[self.domain_boundary_type_bottom].extend(
                gmsh.model.get_entities_for_physical_group(1, bottom_tag)
            )
            bdry_by_name[self.domain_boundary_type_top].extend(
                gmsh.model.get_entities_for_physical_group(1, top_tag)
            )
            bdry_by_name[self.domain_boundary_type_left].extend(
                gmsh.model.get_entities_for_physical_group(1, left_tag)
            )
            bdry_by_name[self.domain_boundary_type_right].extend(
                gmsh.model.get_entities_for_physical_group(1, right_tag)
            )
            for name, bdry in bdry_by_name.items():
                if bdry:
                    gmsh.model.add_physical_group(1, bdry, name=name)

            # required for ngnod = 9
            gmsh.option.setNumber("Mesh.ElementOrder", 2)
            gmsh.model.mesh.generate()

            # === uncomment this to see GUI ===
            # gmsh.fltk.run()

            # =====================================================================
            #                      extract mesh model
            # =====================================================================
            return Model.from_meshed_surface(
                surface=surfaces,
                gmsh=gmsh,
                physical_group_captures=bdry_by_name.keys(),
            )
