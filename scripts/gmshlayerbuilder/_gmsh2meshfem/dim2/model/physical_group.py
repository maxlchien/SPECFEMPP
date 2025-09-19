from abc import ABC, abstractmethod
from typing import override

import numpy as np

from _gmsh2meshfem.gmsh_dep import GmshContext

from .boundary import BoundarySpec
from .index_mapping import IndexMapping


class PhysicalGroupImpl(ABC):
    dim: int
    name: str

    def __init__(
        self,
        gmsh: GmshContext,
        element_nodes: np.ndarray,
        node_mapping: IndexMapping,
        node_coords: np.ndarray,
        dim: int,
        name: str | None = None,
        tag: int | None = None,
    ):
        """Recovers the elements of a given dimension associated with a physical group tag or name.
        the gmsh context is only used in generation.
        `element_nodes` are the un-remapped node indices of each element,
        in the same order as `element_mapping.original_tag_list`.

        Args:
            gmsh (GmshContext): gmsh context
            element_nodes (np.ndarray): the node tags (shape = (N,9)) of all elements.
            node_mapping (IndexMapping): the index mapping of node tags.
            node_coords (np.ndarray): the coordinate array, re-indexed by node_mapping.
            dim (int): dimension of the element
            name (str | None, optional): identifier of the group. If multiple groups have the same
                name, then all of them with the correct dimension are taken. This behavior overrides
                the `tag` argument. If specified, `tag` is not needed.
            tag (int | None, optional): identifier of the group. If specified, `name` is not needed.
                If both `name` and `tag` are given, all tags with the given name are chosen, but if
                `tag` is not among them, an error is thrown.
        """
        self.dim = dim

        # a name may correspond with multiple tags
        tags = []

        if name is None:
            if tag is None:
                e = ValueError("`name` or `tag` must be specified!")
                raise e
            self.name = gmsh.model.get_physical_name(dim, tag)
        else:
            self.name = name
            for _dim, _tag in gmsh.model.get_physical_groups():
                if _dim == dim and gmsh.model.get_physical_name(_dim, _tag) == name:
                    # found a tag with this identity
                    tags.append(_tag)

        if tag is not None and tag not in tags:
            e = ValueError(
                f"`name` {name} was specified, but given `tag` {tag} (name = "
                f"{gmsh.model.get_physical_name(dim, tag)}) was not included!"
            )
            raise e

        # "tag" no longer needed.

        entities = []
        for tag in tags:
            entities.extend(gmsh.model.get_entities_for_physical_group(dim, tag))

        self._populate(gmsh, element_nodes, node_mapping, node_coords, dim, entities)

    @abstractmethod
    def _populate(
        self,
        gmsh: GmshContext,
        element_nodes: np.ndarray,
        node_mapping: IndexMapping,
        node_coords: np.ndarray,
        dim: int,
        entity_tags: list[int],
    ) -> None:
        raise NotImplementedError


class PhysicalGroup0D(PhysicalGroupImpl):
    nodes: np.ndarray

    @override
    def _populate(
        self,
        gmsh: GmshContext,
        element_nodes: np.ndarray,
        node_mapping: IndexMapping,
        node_coords: np.ndarray,
        dim: int,
        entity_tags: list[int],
    ):
        assert dim == 0
        nodelists = []
        for entity in entity_tags:
            gmsh.for_element_types_in_entity(
                dim,
                entity,
                mapping={
                    15: lambda elem_tags, node_tags: nodelists.append(
                        node_mapping.apply(node_tags)
                    )
                },
            )
        self.nodes = np.concatenate(nodelists)


class PhysicalGroup1D(PhysicalGroupImpl):
    edges: BoundarySpec

    @override
    def _populate(
        self,
        gmsh: GmshContext,
        element_nodes: np.ndarray,
        node_mapping: IndexMapping,
        node_coords: np.ndarray,
        dim: int,
        entity_tags: list[int],
    ):
        assert dim == 1
        self.edges = BoundarySpec.from_model_entity(
            gmsh, entity_tags, element_nodes, node_mapping, node_coords
        )
