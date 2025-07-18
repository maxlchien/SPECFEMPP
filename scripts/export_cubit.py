#!/usr/bin/env python
#
# Script to export a Cubit13+/Trelis 2D mesh in specfem2d format
# Initial Author unknown, comments and modifications by Alexis Bottero (alexis dot bottero At gmail dot com)
#
# this script supports both, QUAD4 for linear or QUAD9 for quadratic element types for surface elements,
# as well as BAR2 or BAR3 edge types for edges.

# Create your mesh in Cubit
# (or build the simpleAxisym2dMesh.py for a QUAD4 element example, MeshSansCPMLquad9.py for a QUAD9 element type example)
# and play this script within Cubit as a Python journal file.
#
# Instructions for mesh creation :
# The mesh by default is assumed to be in the XZ plane, but it will recognize also XY or YZ plane meshes!
#
# Define a block per material, for example like :
#      cubit.cmd('block 1 name "Acoustic channel" ') # acoustic material region
#      cubit.cmd('block 1 attribute count 6')        # number of attributes
#      cubit.cmd('block 1 attribute index 1 1')      # material index
#      cubit.cmd('block 1 attribute index 2 1500 ')  # vp
#      cubit.cmd('block 1 attribute index 3 0 ')     # vs
#      cubit.cmd('block 1 attribute index 4 1000 ')  # rho
#      cubit.cmd('block 1 attribute index 5 0 ')     # Q_flag
#      cubit.cmd('block 1 attribute index 6 0 ')     # anisotropy_flag
#      cubit.cmd('block 1 element type QUAD4')       # or 'QUAD9' for quadratic elements
#
# Define a block per absorbing border (abs_bottom, abs_right, abs_left, abs_top, topo, axis).
# For axisymmetric simulations, you don't need to create a block abs_left, but a block axis.
# Example:
#      cubit.cmd('block 3 edge in surf all with z_coord > -0.1') # topo
#      cubit.cmd('block 3 name "topo"')
#
# Define one block per pml layer of a given type (acoustic or elastic)
# pml_x_acoust,pml_z_acoust,pml_xz_acoust,pml_x_elast,pml_z_elast,pml_xz_elast
#      !! Warning !! pml blocks don't have faces in common
#      !! Warning !! you must create the corresponding absorbing surface blocks (abs_bottom, abs_right, abs_left, abs_top)!
#
# The names of the block and the entities types must match the ones given during the definition of the class mesh on this file :
# Below :
# class mesh(mesh_tools):
#     """ A class to store the mesh """
#     def __init__(self):
#
# !! Warning : a block in cubit != quad !! A block is a group of something (quads, edges, volumes, surfaces...)
# On this case the blocks are used to gather faces corresponding to different materials and edges corresponding to free surfaces,
# absorbing surfaces, topography or axis
from __future__ import print_function
import sys

try:
    import cubit
except:
    print("error importing cubit")
    sys.exit()

try:
    set
except NameError:
    from sets import Set as set


class mtools(object):
    """docstring for mtools"""

    def __init__(self, frequency, list_surf, list_vp):
        super(mtools, self).__init__()
        self.frequency = frequency
        self.list_surf = list_surf
        self.list_vp = list_vp
        self.ngll = 5
        self.percent_gll = 0.172
        self.point_wavelength = 5

    def __repr__(self):
        txt = "Meshing for frequency up to " + str(self.frequency) + "Hz\n"
        for surf, vp in zip(self.list_surf, self.list_vp):
            txt = (
                txt
                + "surface "
                + str(surf)
                + ", vp ="
                + str(vp)
                + "  -> size "
                + str(self.freq2meshsize(vp)[0])
                + " -> dt "
                + str(self.freq2meshsize(vp)[0])
                + "\n"
            )
        return txt

    def freq2meshsize(self, vp):
        velocity = vp * 0.5
        self.size = (
            (1 / 2.5)
            * velocity
            / self.frequency
            * (self.ngll - 1)
            / self.point_wavelength
        )
        self.dt = 0.4 * self.size / vp * self.percent_gll
        return self.size, self.dt

    def mesh_it(self):
        for surf, vp in zip(self.list_surf, self.list_vp):
            command = "surface " + str(surf) + " size " + str(self.freq2meshsize(vp)[0])
            cubit.cmd(command)
            command = "surface " + str(surf) + "scheme pave"
            cubit.cmd(command)
            command = "mesh surf " + str(surf)
            cubit.cmd(command)


class block_tools(object):
    def __int__(self):
        pass

    def create_blocks(
        self,
        mesh_entity,
        list_entity=None,
    ):
        if mesh_entity == "surface":
            txt = " face in surface "
        elif mesh_entity == "curve":
            txt = " edge in curve "
        elif mesh_entity == "group":
            txt = " face in group "
        if list_entity:
            if not isinstance(list_entity, list):
                list_entity = [list_entity]
        for entity in list_entity:
            iblock = cubit.get_next_block_id()
            command = "block " + str(iblock) + txt + str(entity)
            cubit.cmd(command)

    def material_file(self, filename):
        matfile = open(filename, "w")
        material = []
        for record in matfile:
            mat_name, vp_str = record.split()
            vp = float(vp_str)
            material.append([mat_name, vp])
        self.material = dict(material)

    def assign_block_material(self, id_block, mat_name, vp=None):
        try:
            material = self.material
        except:
            material = None
        cubit.cmd("block " + str(id_block) + " attribute count 2")
        cubit.cmd("block " + str(id_block) + "  attribute index 1 " + str(id_block))
        if material:
            if material.has_key(mat_name):
                cubit.cmd(
                    "block "
                    + str(id_block)
                    + "  attribute index 2 "
                    + str(material[mat_name])
                )
                print(
                    "block "
                    + str(id_block)
                    + " - material "
                    + mat_name
                    + " - vp "
                    + str(material[mat_name])
                    + " from database"
                )
        elif vp:
            cubit.cmd("block " + str(id_block) + "  attribute index 2 " + str(vp))
            print(
                "block "
                + str(id_block)
                + " - material "
                + mat_name
                + " - vp "
                + str(vp)
            )
        else:
            print(
                "assignment impossible: check if "
                + mat_name
                + " is in the database or specify vp"
            )


class mesh_tools(block_tools):
    """Tools for the mesh
    #########
    dt,edge_dt,freq,edge_freq = seismic_resolution(edges,velocity,bins_d = None,bins_u = None,sidelist = None,ngll = 5,np = 8)
        Given the velocity of a list of edges, seismic_resolution provides the minimum Dt required for the stability condition (and the corrisponding edge).
        Furthermore, given the number of gll point in the element (ngll) and the number of GLL point for wavelength, it provide the maximum resolved frequency.
    #########
    length = edge_length(edge)
        return the length of a edge
    #########
    edge_min,length = edge_min_length(surface)
        given the cubit id of a surface, it return the edge with minimun length
    #########
    """

    def __int__(self):
        pass

    def seismic_resolution(
        self, edges, velocity, bins_d=None, bins_u=None, sidelist=None
    ):
        """
        dt,edge_dt,freq,edge_freq = seismic_resolution(edges,velocity,bins_d = None,bins_u = None,sidelist = None,ngll = 5,np = 8)
            Given the velocity of a list of edges, seismic_resolution provides the minimum Dt required for the stability condition (and the corrisponding edge).
            Furthermore, given the number of gll point in the element (ngll) and the number of GLL point for wavelength, it provide the maximum resolved frequency.
        """
        ratiostore = 1e10
        dtstore = 1e10
        edgedtstore = -1
        edgeratiostore = -1
        for edge in edges:
            d = self.edge_length(edge)
            ratio = (1 / 2.5) * velocity / d * (self.ngll - 1) / self.point_wavelength
            dt = 0.4 * d / velocity * self.percent_gll
            if dt < dtstore:
                dtstore = dt
                edgedtstore = edge
            if ratio < ratiostore:
                ratiostore = ratio
                edgeratiostore = edge
            try:
                for bin_d, bin_u, side in zip(bins_d, bins_u, sidelist):
                    if ratio >= bin_d and ratio < bin_u:
                        command = "sideset " + str(side) + " edge " + str(edge)
                        cubit.cmd(command)
                        # print(command)
                        break
            except:
                pass
        return dtstore, edgedtstore, ratiostore, edgeratiostore

    def edge_length(self, edge):
        """
        length = edge_length(edge)
            return the length of a edge
        """
        from math import sqrt

        nodes = cubit.get_connectivity("edge", edge)
        x0, y0, z0 = cubit.get_nodal_coordinates(nodes[0])
        x1, y1, z1 = cubit.get_nodal_coordinates(nodes[1])
        d = sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)
        return d

    def edge_min_length(self, surface):
        """
        edge_min,length = edge_min_length(surface)
            given the cubit id of a surface, it return the edge with minimun length
        """
        self.dmin = 99999
        edge_store = 0
        command = "group 'list_edge' add edge in surf " + str(surface)
        command = command.replace("[", " ").replace("]", " ")
        # print(command)
        cubit.cmd(command)
        group = cubit.get_id_from_name("list_edge")
        edges = cubit.get_group_edges(group)
        command = "delete group " + str(group)
        cubit.cmd(command)
        for edge in edges:
            d = self.edge_length(edge)
            if d < dmin:
                self.dmin = d
                edge_store = edge
        self.edgemin = edge_store
        return self.edgemin, self.dmin

    def jac_check(self, nodes, plane_id, type=""):
        x0 = cubit.get_nodal_coordinates(nodes[0])
        x1 = cubit.get_nodal_coordinates(nodes[1])
        x2 = cubit.get_nodal_coordinates(nodes[2])
        # plane identifier: 1 == XZ-plane, 2 == XY-plane, 3 == YZ-plane
        if plane_id == 1:
            # XZ-plane
            xv1 = x1[0] - x0[0]
            xv2 = x2[0] - x1[0]
            zv1 = x1[2] - x0[2]
            zv2 = x2[2] - x1[2]
            jac = -xv2 * zv1 + xv1 * zv2
        elif plane_id == 2:
            # XY-plane
            xv1 = x1[0] - x0[0]
            xv2 = x2[0] - x1[0]
            yv1 = x1[1] - x0[1]
            yv2 = x2[1] - x1[1]
            jac = -xv2 * yv1 + xv1 * yv2
        else:
            # YZ-plane
            yv1 = x1[1] - x0[1]
            yv2 = x2[1] - x1[1]
            zv1 = x1[2] - x0[2]
            zv2 = x2[2] - x1[2]
            jac = -yv2 * zv1 + yv1 * zv2
        # checks jacobian
        if jac > 0:
            return nodes
        elif jac < 0:
            # change the ordre for the local coordinate system
            if type == "QUAD9":
                # for 9 node finite elements Page.11 in Specfem2d-manual.pdf
                return (
                    nodes[0],
                    nodes[3],
                    nodes[2],
                    nodes[1],
                    nodes[7],
                    nodes[6],
                    nodes[5],
                    nodes[4],
                    nodes[8],
                )
            else:
                # QUAD4 type
                return nodes[0], nodes[3], nodes[2], nodes[1]
        else:
            print("error, jacobian = 0", jac, nodes, "x0/x1/x2:", x0, x1, x2)

    def mesh_analysis(self, frequency):
        cubit.cmd("set info on")  # Turn off return messages from Cubit commands
        cubit.cmd("set echo off")  # Turn off echo of Cubit commands
        cubit.cmd("set journal off")  # Do not save journal file
        bins_d = [0.0001] + range(0, int(frequency) + 1) + [1000]
        bins_u = bins_d[1:]
        dt = []
        ed_dt = []
        r = []
        ed_r = []
        nstart = cubit.get_next_sideset_id()
        command = "del sideset all"
        cubit.cmd(command)
        for bin_d, bin_u in zip(bins_d, bins_u):
            nsideset = cubit.get_next_sideset_id()
            command = "create sideset " + str(nsideset)
            cubit.cmd(command)
            command = (
                "sideset "
                + str(nsideset)
                + " name "
                + "'ratio-["
                + str(bin_d)
                + "_"
                + str(bin_u)
                + "['"
            )
            cubit.cmd(command)
        nend = cubit.get_next_sideset_id()
        sidelist = range(nstart, nend)
        for block in self.block_mat:
            name = cubit.get_exodus_entity_name("block", block)
            velocity = self.material[name][1]
            if velocity > 0:
                faces = cubit.get_block_faces(block)
                edges = []
                for face in faces:
                    es = cubit.get_sub_elements("face", face, 1)
                    edges = edges + list(es)
                dtstore, edgedtstore, ratiostore, edgeratiostore = (
                    self.seismic_resolution(edges, velocity, bins_d, bins_u, sidelist)
                )
                dt.append(dtstore)
                ed_dt.append(edgedtstore)
                r.append(ratiostore)
                ed_r.append(edgeratiostore)
        self.ddt = zip(ed_dt, dt)
        self.dr = zip(ed_r, r)

        def sorter(x, y):
            return cmp(x[1], y[1])

        self.ddt.sort(sorter)
        self.dr.sort(sorter)
        print(self.ddt, self.dr)
        print(
            "Deltat minimum => edge:"
            + str(self.ddt[0][0])
            + " dt: "
            + str(self.ddt[0][1])
        )
        print(
            "Minimum frequency resolved => edge:"
            + str(self.dr[0][0])
            + " frequency: "
            + str(self.dr[0][1])
        )
        cubit.cmd("set info on")
        cubit.cmd("set echo on")
        return self.ddt[0], self.dr[0]


class mesh(mesh_tools):
    """A class to store the mesh"""

    def __init__(self):
        super(mesh, self).__init__()
        self.mesh_name = "mesh_file"
        self.axisymmetric_mesh = (
            False  # Will be set to true if a group self.pml_boun_name is found
        )
        self.topo_mesh = False  # Will be set to true if a group self.topo is found
        self.forcing_mesh = (
            False  # Will be set to true if a group self.forcing_boun_name is found
        )
        self.abs_mesh = False  # Will be set to true if a group self.pml_boun_name or self.abs_boun_name is found
        self.pml_layers = (
            False  # Will be set to true if a group self.pml_boun_name is found
        )
        self.write_nummaterial_velocity_file = (
            False  # Will be set to True if 2d blocks have 6 attributes
        )
        self.nodecoord_name = (
            "nodes_coords_file"  # Name of nodes coordinates file to create
        )
        self.material_name = "materials_file"  # Name of material file to create
        self.nummaterial_name = "nummaterial_velocity_file"
        self.absname = (
            "absorbing_surface_file"  # Name of absorbing surface file to create
        )
        self.forcname = "forcing_surface_file"  # Name of forcing surface file to create
        self.freename = "free_surface_file"  # Name of free surface file to create
        self.pmlname = "elements_cpml_list"  # Name of cpml file to create
        self.axisname = "elements_axis"  # Name of axial elements file to create and name of the block containing axial edges
        self.recname = "STATIONS"
        self.face = ["QUAD4", "QUAD9"]  # Faces' type
        self.edge = ["BAR2", "BAR3"]  # Edges' type
        self.plane_id = (
            1  # plane identifier: 1 == XZ-plane, 2 == XY-plane, 3 == YZ-plane
        )
        self.topo = "topo"  # Name of the block containing topography edges
        self.pml_boun_name = [
            "pml_x_acoust",
            "pml_z_acoust",
            "pml_xz_acoust",
            "pml_x_elast",
            "pml_z_elast",
            "pml_xz_elast",
        ]  # Name of the block containing pml layers elements
        self.abs_boun_name = [
            "abs_bottom",
            "abs_right",
            "abs_top",
            "abs_left",
        ]  # Name of the block containing absorbing layer edges
        self.forcing_boun_name = [
            "forcing_bottom",
            "forcing_right",
            "forcing_top",
            "forcing_left",
        ]  # Name of the block containing forcing layer edges
        self.abs_boun = []  # block numbers for abs boundaries
        self.pml_boun = []  # block numbers for pml boundaries
        self.nforc = 4  # Maximum number of forcing surfaces (4)
        self.nabs = 4  # Maximum number of absorbing surfaces (4)
        self.rec = "receivers"
        self.block_definition()  # Import blocks features from Cubit
        self.ngll = 5
        self.percent_gll = 0.172
        self.point_wavelength = 5
        cubit.cmd("compress")  # Fill the gaps in the numbering of the entities
        # get 2d plane of coordinates
        # determined based on bounding box dimensions
        list_vol = cubit.parse_cubit_list("volume", "all")
        # print("# volume list length = ",len(list_vol))
        xmin_box = cubit.get_total_bounding_box("volume", list_vol)[0]
        xmax_box = cubit.get_total_bounding_box("volume", list_vol)[1]
        ymin_box = cubit.get_total_bounding_box("volume", list_vol)[3]
        ymax_box = cubit.get_total_bounding_box("volume", list_vol)[4]
        zmin_box = cubit.get_total_bounding_box("volume", list_vol)[
            6
        ]  # it is the z_min of the box ... box= xmin,xmax,d,ymin,ymax,d,zmin...
        zmax_box = cubit.get_total_bounding_box("volume", list_vol)[7]
        # print("# bounding box xmin/xmax = ",xmin_box,xmax_box)
        # print("# bounding box ymin/ymax = ",ymin_box,ymax_box)
        # print("# bounding box zmin/zmax = ",zmin_box,zmax_box)
        # print("")
        # plane identifier: 1 == XZ-plane, 2 == XY-plane, 3 == YZ-plane
        if abs(ymax_box - ymin_box) < 0.001:
            print("# cubit2specfem2d: mesh in XZ plane")
            self.plane_id = 1
        elif abs(zmax_box - zmin_box) < 0.001:
            print("# cubit2specfem2d: mesh in XY plane")
            self.plane_id = 2
        elif abs(xmax_box - xmin_box) < 0.001:
            print("# cubit2specfem2d: mesh in YZ plane")
            self.plane_id = 3
        else:
            print(
                "# WARNING: cubit2specfem2d: mesh is not 2D, will ignored Y-dimension"
            )
            self.plane_id = 1
        # print("# plane identifier: ",plane_id)
        # print("#")

    def __repr__(self):
        pass

    def block_definition(self):
        """Import blocks features from Cubit"""
        block_flag = []  # Will contain material id (1 if fluid 2 if solid)
        block_mat = []  # Will contain face block ids
        block_bc = []  # Will contain edge block ids
        block_bc_flag = []  # Will contain edge id -> 2
        abs_boun = (
            [-1] * self.nabs
        )  # total 4 sides of absorbing boundaries (index 0 : bottom, index 1 : right, index 2 : top, index 3 : left)
        forcing_boun = (
            [-1] * self.nforc
        )  # 4 possible forcing boundaries (index 0 : bottom, index 1 : right, index 2 : top, index 3 : left)
        # pml_boun = [-1] * 6 # To store pml layers id (for each pml layer : x_acoust, z_acoust, xz_acoust, x_elast, z_elast, xz_elast)
        pml_boun = [
            [] for _ in range(6)
        ]  # To store the block id corresponding to pml layers id (arbitrary number of blocks for each pml layer : x_acoust, z_acoust, xz_acoust, x_elast, z_elast, xz_elast)
        material = {}  # Will contain each material name and their properties
        bc = {}  # Will contains each boundary name and their connectivity -> 2
        blocks = cubit.get_block_id_list()  # Load the blocks list
        for block in blocks:  # Loop on the blocks
            name = cubit.get_exodus_entity_name(
                "block", block
            )  # Contains the name of the blocks
            type = cubit.get_block_element_type(
                block
            )  # Contains the block element type (QUAD4...)
            if type in self.face:  # If we are dealing with a block containing faces
                print("block: ", name, " contains faces ", type)
                nAttributes = cubit.get_block_attribute_count(block)
                if nAttributes != 1 and nAttributes != 6:
                    print(
                        "Blocks not properly defined, 2d blocks must have one attribute (material id) or 6 attributes"
                    )
                    return None, None, None, None, None, None, None, None
                flag = int(
                    cubit.get_block_attribute_value(block, 0)
                )  # Fetch the first attribute value (containing material id)
                print("  nAttributes : ", nAttributes)
                if nAttributes == 6:
                    self.write_nummaterial_velocity_file = True
                    velP = cubit.get_block_attribute_value(
                        block, 1
                    )  # Fetch the first attribute value (containing P wave velocity)
                    velS = cubit.get_block_attribute_value(
                        block, 2
                    )  # Fetch the second attribute value (containing S wave velocity)
                    rho = cubit.get_block_attribute_value(
                        block, 3
                    )  # Fetch the third attribute value (containing material density)
                    qFlag = cubit.get_block_attribute_value(
                        block, 4
                    )  # Fetch the first attribute value (containing Qflag)
                    anisotropy_flag = cubit.get_block_attribute_value(
                        block, 5
                    )  # Fetch the first attribute value (containing anisotropy_flag)
                    anisotropy_flag = int(anisotropy_flag)
                    # Store (material_id,rho,velP,velS,Qflag,anisotropy_flag) in par :
                    par = tuple([flag, rho, velP, velS, qFlag, anisotropy_flag])
                    material[name] = (
                        par  # associate the name of the block to its id and properties
                    )
                block_flag.append(int(flag))  # Append material id to block_flag
                block_mat.append(block)  # Append block id to block_mat
                for pml_idx, pml_name in enumerate(self.pml_boun_name):
                    # block considered refered to one of the pml layers
                    if pml_name in name:
                        pml_boun[pml_idx].append(block)
                        self.abs_mesh = True
                        self.pml_layers = True
                    # -> Put it at the correct position in pml_boun
                    # (index 0 : pml_x_acoust, index 1 : pml_z_acoust, index 2 : pml_xz_acoust,
                    #  index 3 : pml_x_elast, index 4 : pml_z_elast, index 5 : pml_xz_elast)
                # if name in self.pml_boun_name : # If the block considered refered to one of the pml layer
                #    self.abs_mesh = True
                #    self.pml_layers = True
                #    pml_boun[self.pml_boun_name.index(name)] = block
                #    # -> Put it at the correct position in pml_boun
                #    # (index 0 : pml_x_acoust, index 1 : pml_z_acoust, index 2 : pml_xz_acoust,
                #    #  index 3 : pml_x_elast, index 4 : pml_z_elast, index 5 : pml_xz_elast)
            elif type in self.edge:  # If we are dealing with a block containing edges
                print("block: ", name, " contains edges ", type)
                block_bc_flag.append(2)  # Append "2" to block_bc_flag
                block_bc.append(block)  # Append block id to block_bc
                bc[name] = (
                    2  # Associate the name of the block with its connectivity : an edge has connectivity = 2
                )
                if name == self.topo:
                    self.topo_mesh = True
                    print("  topo_mesh: ", self.topo_mesh)
                    topography = block  # If the block considered refered to topography store its id in "topography"
                if name in self.forcing_boun_name:
                    self.forcing_mesh = True
                    print("  forcing_mesh: ", self.forcing_mesh)
                    forcing_boun[self.forcing_boun_name.index(name)] = block
                    # -> Put it at the correct position in abs_boun (index 0 : bottom, index 1 : right, index 2 : top, index 3 : left)
                if name == self.axisname:
                    self.axisymmetric_mesh = True
                    print("  axisymmetric_mesh: ", self.axisymmetric_mesh)
                    axisId = block  # AXISYM If the block considered refered to the axis store its id in "axisId"
                if (
                    name in self.abs_boun_name
                ):  # If the block considered refered to one of the boundaries
                    self.abs_mesh = True
                    print("  abs_mesh: ", self.abs_mesh)
                    abs_boun[self.abs_boun_name.index(name)] = block
                    # -> Put it at the correct position in abs_boun (index 0 : bottom, index 1 : right, index 2 : top, index 3 : left)
            else:
                print("Blocks not properly defined", type)
                return None, None, None, None, None, None, None, None
        nsets = cubit.get_nodeset_id_list()  # Get the list of all nodeset
        if len(nsets) == 0:
            self.receivers = None  # If this list is empty : put None in self.receivers
        for nset in nsets:
            name = cubit.get_exodus_entity_name(
                "nodeset", nset
            )  # Contains the name of the nodeset
            if name == self.rec:  # If the name considered match self.rec (receivers)
                self.receivers = nset  # Store the id of the nodeset in self.receivers
            else:
                print("nodeset " + name + " not defined")
                self.receivers = None
        # Store everything in the object :
        try:
            self.block_mat = block_mat
            self.block_flag = block_flag
            self.block_bc = block_bc
            self.block_bc_flag = block_bc_flag
            self.bc = bc
            if self.write_nummaterial_velocity_file:
                self.material = material
            if self.abs_mesh:
                self.abs_boun = abs_boun
            if self.topo_mesh:
                self.topography = topography
            if self.forcing_mesh:
                self.forcing_boun = forcing_boun
            if self.axisymmetric_mesh:
                self.axisId = axisId
            if self.pml_layers:
                self.pml_boun = pml_boun
        except:
            print("Blocks not properly defined")

    #    def tomo(self,flag,vel):
    #        vp = vel/1000
    #        rho = (1.6612*vp-0.472*vp**2+0.0671*vp**3-0.0043*vp**4+0.000106*vp**4)*1000
    #        txt = '%3i %1i %20f %20f %20f %1i %1i\n' % (flag,1,rho,vel,vel/(3**.5),0,0)
    #        return txt
    def mat_parameter(self, properties):
        # print properties
        # format nummaterials file:
        # #material_domain_id #material_id #rho #vp #vs #Q_kappa #Q_mu #anisotropy_flag
        # get material properties
        mat_id = properties[0]
        # print('number of material:', mat_id)
        # material id flag must be strictly positive or negative, but not equal to 0
        if mat_id > 0:
            # material defined
            rho = properties[1]
            vp = properties[2]
            vs = properties[3]
            Q_kappa = 9999.0  # not defined yet
            Q_mu = properties[4]
            aniso_flag = properties[5]
            # determine acoustic==1 or elastic==2 domain
            domain_id = 1 if vs == 0.0 else 2
            # format:
            # (1)material_domain_id #(2)material_id  #(3)rho  #(4)vp   #(5)vs   #(6)Q_kappa   #(7)Q_mu  #(8)anisotropy_flag
            # with material_domain_id ==1 for acoustic or ==2 for elastic materials
            txt = (
                f"{domain_id} {mat_id} {rho} {vp} {vs} {Q_kappa} {Q_mu} {aniso_flag}\n"
            )
        elif mat_id < 0:
            # material undefined, for tomography file
            if properties[2] == "tomography":
                # format:
                # (1)domain_id #(2)material_id  tomography elastic  #(3)filename #(4)positive
                txt = "%1i %3i %s %s 1\n" % (
                    2,
                    mat_id,
                    "tomography elastic",
                    properties[3],
                )
        else:
            raise RuntimeError(
                "Error: material id must be strictly positive or negative, but not equal to 0"
            )
        # info output
        # print("material: ",txt)
        return txt

    def nummaterial_write(self, nummaterial_name, placeholder=True):
        """Write material features on file : nummaterial_name"""
        print("Writing " + nummaterial_name + ".....")
        nummaterial = open(
            nummaterial_name, "w"
        )  # Create the file "nummaterial_name" and open it
        if placeholder:
            txt = """# nummaterial_velocity_file - created by script cubit2specfem2d.py
# format:
#(1)domain_id #(2)material_id #(3)rho #(4)vp #(5)vs #(6)Q_k #(7)Q_mu #(8)ani
#
#  where
#     domain_id          : 1=acoustic / 2=elastic / 3=poroelastic
#     material_id        : POSITIVE integer identifier of material block
#     rho                : density
#     vp                 : P-velocity
#     vs                 : S-velocity
#     Q_k                : 9999 = no Q_kappa attenuation
#     Q_mu               : 9999 = no Q_mu attenuation
#     ani                : 0=no anisotropy/ 1,2,.. check with aniso_model.f90
#
# example:
# 2   1 2300 2800 1500 9999.0 9999.0 0
#
# or
#
#(1)domain_id #(2)material_id  tomography elastic  #(3)filename #(4)positive
#
#  where
#     domain_id : 1=acoustic / 2=elastic / 3=poroelastic
#     material_id        : NEGATIVE integer identifier of material block
#     filename           : filename of the tomography file
#     positive           : a positive unique identifier
#
# example:
# 2  -1 tomography elastic tomo.xyz 1
#
# materials
"""
            nummaterial.write(txt)
        # writes block materials
        for block in self.block_mat:  # For each 2D block
            name = cubit.get_exodus_entity_name(
                "block", block
            )  # Extract the name of the block
            nummaterial.write(str(self.mat_parameter(self.material[name])))
        nummaterial.close()
        print("Ok")

    def mesh_write(self, mesh_name):
        """Write mesh (quads ids with their corresponding nodes ids) on file : mesh_name"""
        meshfile = open(mesh_name, "w")
        print("Writing " + mesh_name + ".....")
        cubit.cmd("set info off")  # Turn off return messages from Cubit commands
        cubit.cmd("set echo off")  # Turn off echo of Cubit commands
        num_elems = cubit.get_quad_count()  # Store the number of elements
        toWritetoFile = [""] * (num_elems + 1)
        toWritetoFile[0] = str(num_elems) + "\n"
        # meshfile.write(str(num_elems)+'\n') # Write it on first line
        num_write = 0
        for block, flag in zip(self.block_mat, self.block_flag):  # for each 2D block
            quads = cubit.get_block_faces(block)  # Import quads ids
            type = cubit.get_block_element_type(
                block
            )  # Contains the block element type (QUAD4...)
            for inum, quad in enumerate(quads):  # For each of these quads
                if type == "QUAD9":
                    # QUAD9
                    nodes = cubit.get_expanded_connectivity(
                        "face", quad
                    )  # Get all the nodes in quad including interior points
                else:
                    # QUAD4
                    nodes = cubit.get_connectivity("face", quad)  # Get the nodes
                nodes = self.jac_check(nodes, self.plane_id, type)  # Check the jacobian
                if type == "QUAD9":
                    # QUAD9: quadratic element: 4 corners + 4 edge mid-points + 1 center nodal point
                    # format: #elem1 #elem2 #elem3 #elem4 #elem5 #elem6 #elem7 #elem8 #elem9
                    txt = ("%10i %10i %10i %10i %10i %10i %10i %10i %10i\n") % nodes
                else:
                    # QUAD4: linear element: 4 corners
                    # format: #elem1 #elem2 #elem3 #elem4
                    txt = ("%10i %10i %10i %10i\n") % nodes
                toWritetoFile[quad] = txt
                # meshfile.write(txt) # Write a line to mesh file
            num_write = num_write + inum + 1
            print("block", block, "number of type ", type, " : ", inum + 1)
        meshfile.writelines(toWritetoFile)
        meshfile.close()
        print("Ok num elements/write =", str(num_elems), str(num_write))
        cubit.cmd("set info on")
        cubit.cmd("set echo on")

    def material_write(self, mat_name):
        """Write quads material on file : mat_name"""
        mat = open(mat_name, "w")
        print("Writing " + mat_name + ".....")
        cubit.cmd("set info off")  # Turn off return messages from Cubit commands
        cubit.cmd("set echo off")  # Turn off echo of Cubit commands
        num_elems = cubit.get_quad_count()  # Store the number of elements
        toWritetoFile = [""] * num_elems
        # print('block_mat:',self.block_mat)
        # print('block_flag:',self.block_flag)
        for block, flag in zip(self.block_mat, self.block_flag):  # for each 2D block
            print("mat: ", block, " flag: ", flag)
            quads = cubit.get_block_faces(block)  # Import quads id
            for quad in quads:  # For each quad
                toWritetoFile[quad - 1] = ("%10i\n") % flag
                # mat.write(('%10i\n') % flag) # Write its id in the file
        mat.writelines(toWritetoFile)
        mat.close()
        print("Ok")
        cubit.cmd("set info on")
        cubit.cmd("set echo on")

    def pmls_write(self, pml_name):
        """Write pml elements on file : mat_name"""
        pml_file = open(pml_name, "w")
        print("Writing " + pml_name + ".....")
        cubit.cmd("set info off")  # Turn off return messages from Cubit commands
        cubit.cmd("set echo off")  # Turn off echo of Cubit commands
        npml_elements = 0
        # id_element = 0 # Global id
        indexFile = 1
        faces_all = [[] for _ in range(6)]
        for block, flag in zip(self.block_mat, self.block_flag):  # For each 2D block
            for ipml in range(
                6
            ):  # ipml = 0,1,2,3,4,5 : for each pml layer (x_acoust, z_acoust, xz_acoust,x_elast, z_elast, xz_elast)
                if (
                    block in self.pml_boun[ipml]
                ):  # If the block considered correspond to the pml
                    faces_all[ipml] = faces_all[ipml] + list(
                        cubit.get_block_faces(block)
                    )  # Concatenation
        npml_elements = sum(map(len, faces_all))
        toWritetoFile = [""] * (npml_elements + 1)
        toWritetoFile[0] = (
            "%10i\n" % npml_elements
        )  # Print the number of faces on the pmls
        # pml_file.write('%10i\n' % npml_elements) # Print the number of faces on the pmls
        print("Number of elements in all PMLs :", npml_elements)
        for block, flag in zip(self.block_mat, self.block_flag):  # For each 2D block
            quads = cubit.get_block_faces(block)  # Import quads id
            type = cubit.get_block_element_type(
                block
            )  # Contains the block element type (QUAD4...)
            for quad in quads:  # For each quad
                # id_element = id_element+1 # global id of this quad
                for ipml in range(
                    0, 6
                ):  # iabs = 0,1,2,3,4,5 : for each pml layer (x_acoust, z_acoust, xz_acoust,x_elast, z_elast, xz_elast)
                    if (
                        faces_all[ipml] != []
                    ):  # type(faces_all[ipml]) is not int: # ~ if there are elements in that pml
                        if (
                            quad in faces_all[ipml]
                        ):  # If this quad is belong to that pml
                            #  nodes = cubit.get_connectivity('face',quad) # Import the nodes describing the quad
                            #  nodes = self.jac_check(list(nodes),self.plane_id) # Check the jacobian of the quad
                            toWritetoFile[indexFile] = ("%10i %10i\n") % (
                                quad,
                                ipml % 3 + 1,
                            )
                            indexFile = indexFile + 1
                            # pml_file.write(('%10i %10i\n') % (id_element,ipml%3+1)) # Write its id in the file next to its type
        # ipml%3+1 = 1 -> element belongs to a X CPML layer only (either in Xmin or in Xmax)
        # ipml%3+1 = 2 -> element belongs to a Z CPML layer only (either in Zmin or in Zmax)
        # ipml%3+1 = 3 -> element belongs to both a X and a Y CPML layer (i.e., to a CPML corner)
        pml_file.writelines(toWritetoFile)
        pml_file.close()
        print("Ok")
        cubit.cmd("set info on")  # Turn on return messages from Cubit commands
        cubit.cmd("set echo on")  # Turn on echo of Cubit commands

    def nodescoord_write(self, nodecoord_name):
        """Write nodes coordinates on file : nodecoord_name"""
        nodecoord = open(nodecoord_name, "w")
        print("Writing " + nodecoord_name + ".....")
        cubit.cmd("set info off")  # Turn off return messages from Cubit commands
        cubit.cmd("set echo off")  # Turn off echo of Cubit commands
        node_list = cubit.parse_cubit_list(
            "node", "all"
        )  # Import all the nodes of the model
        num_nodes = len(node_list)  # Total number of nodes
        nodecoord.write(
            "%10i\n" % num_nodes
        )  # Write the number of nodes on the first line
        for node in node_list:  # For all nodes
            x, y, z = cubit.get_nodal_coordinates(
                node
            )  # Import its coordinates (3 coordinates even for a 2D model in cubit)
            # plane identifier: 1 == XZ-plane, 2 == XY-plane, 3 == YZ-plane
            if self.plane_id == 1:
                txt = ("%20f %20f\n") % (x, z)
            elif self.plane_id == 2:
                txt = ("%20f %20f\n") % (x, y)
            else:
                txt = ("%20f %20f\n") % (y, z)
            nodecoord.write(txt)  # Write 2d coordinates on the file
        nodecoord.close()
        print("Ok")
        cubit.cmd("set info on")  # Turn on return messages from Cubit commands
        cubit.cmd("set echo on")  # Turn on echo of Cubit commands

    def free_write(self, freename):  # freename = None):
        """Write free surface on file : freename"""
        # if not freename: freename = self.freename
        freeedge = open(freename, "w")
        print("Writing " + freename + ".....")
        cubit.cmd("set info off")  # Turn off return messages from Cubit commands
        cubit.cmd("set echo off")  # Turn off echo of Cubit commands
        cubit.cmd("set journal off")  # Do not save journal file
        if self.topo_mesh:
            for block, flag in zip(
                self.block_bc, self.block_bc_flag
            ):  # For each 1D block
                if block == self.topography:  # If the block correspond to topography
                    edges_all = set(
                        cubit.get_block_edges(block)
                    )  # Import all topo edges id as a Set
            toWritetoFile = []  # [""]*(len(edges_all)+1)
            toWritetoFile.append(
                "%10i\n" % len(edges_all)
            )  # Print the number of edges on the free surface
            for block, flag in zip(
                self.block_mat, self.block_flag
            ):  # For each 2D block
                # print('free surface: ',block,flag)
                quads = cubit.get_block_faces(block)  # Import quads id
                type = cubit.get_block_element_type(
                    block
                )  # Contains the block element type (QUAD4...)
                for quad in quads:  # For each quad
                    edges = set(
                        cubit.get_sub_elements("face", quad, 1)
                    )  # Get the lower dimension entities associated with a higher dimension entities.
                    # Here it gets the 1D edges associates with the face of id "quad". Store it as a Set
                    intersection = (
                        edges & edges_all
                    )  # Contains the edges of the considered quad that is on the free surface
                    if len(intersection) != 0:  # If this quad touch the free surface
                        # print("  ",quad," -> this quad touch the free surface!")
                        if type == "QUAD9":
                            # QUAD9
                            nodes = cubit.get_expanded_connectivity(
                                "face", quad
                            )  # Get all the nodes in quad including interior points
                        else:
                            # QUAD4
                            nodes = cubit.get_connectivity(
                                "face", quad
                            )  # Import the nodes describing the quad
                        # print("    it is described by nodes:",nodes," and edges :",edges)
                        # print("      edges:",intersection," is/are on the free surface")
                        nodes = self.jac_check(
                            list(nodes), self.plane_id, type
                        )  # Check the jacobian of the quad
                        for e in intersection:  # For each edge on the free surface
                            node_edge = cubit.get_connectivity(
                                "edge", e
                            )  # Import the nodes describing the edge
                            # print("      edge",e,"is composed of nodes",node_edge)
                            nodes_ok = []
                            for i in nodes:  # Loop on the nodes of the quad
                                if (
                                    i in node_edge
                                ):  # If this node is belonging to the free surface
                                    nodes_ok.append(i)  # Put it in nodes_ok
                            # print("    nodes:",nodes_ok,"belong to free surface")
                            # free surface contains 1/ element number, 2/ number of nodes that form the free surface,
                            # 3/ first node on the free surface, 4/ second node on the free surface, if relevant (if 2/ is equal to 2)
                            # format: #elemid #num_nodes==2 #node1 #node2
                            txt = "%10i %10i %10i %10i\n" % (
                                quad,
                                2,
                                nodes_ok[0],
                                nodes_ok[1],
                            )
                            toWritetoFile.append(txt)
                            # Write the id of the quad, 2 (number of nodes describing a free surface elements), and the nodes
            freeedge.writelines(toWritetoFile)
        else:
            freeedge.write(
                "0"
            )  # Even without any free surface specfem2d need a file with a 0 in first line
        freeedge.close()
        print("Ok")
        cubit.cmd("set info on")  # Turn on return messages from Cubit commands
        cubit.cmd("set echo on")  # Turn on echo of Cubit commands

    def forcing_write(self, forcname):
        """Write forcing surfaces on file : forcname"""
        forceedge = open(forcname, "w")
        print("Writing " + forcname + ".....")
        cubit.cmd("set info off")  # Turn off return messages from Cubit commands
        cubit.cmd("set echo off")  # Turn off echo of Cubit commands
        cubit.cmd("set journal off")  # Do not save journal file
        edges_forc = (
            [set()] * self.nforc
        )  # edges_forc[0] will be a Set containing the nodes describing the forcing boundary
        # (index 0 : bottom, index 1 : right, index 2 : top, index 3 : left)
        nedges_all = 0  # To count the total number of forcing edges
        for block, flag in zip(self.block_bc, self.block_bc_flag):  # For each 1D block
            for iforc in range(
                0, self.nforc
            ):  # iforc = 0,1,2,3 : for each forcing boundaries
                if (
                    block == self.forcing_boun[iforc]
                ):  # If the block considered correspond to the boundary
                    edges_forc[iforc] = set(
                        cubit.get_block_edges(block)
                    )  # Store each edge on edges_forc
                    nedges_all = nedges_all + len(
                        edges_forc[iforc]
                    )  # add the number of edges to nedges_all
        toWritetoFile = [""] * (nedges_all + 1)
        toWritetoFile[0] = (
            "%10i\n" % nedges_all
        )  # Write the total number of forcing edges to the first line of file
        # forceedge.write('%10i\n' % nedges_all) # Write the total number of forcing edges to the first line of file
        print("Number of edges", nedges_all)
        # id_element = 0
        indexFile = 1
        for block, flag in zip(self.block_mat, self.block_flag):  # For each 2D block
            quads = cubit.get_block_faces(block)  # Import quads id
            type = cubit.get_block_element_type(
                block
            )  # Contains the block element type (QUAD4...)
            for quad in quads:  # For each quad
                # id_element = id_element+1 # id of this quad
                edges = set(
                    cubit.get_sub_elements("face", quad, 1)
                )  # Get the lower dimension entities associated with a higher dimension entities.
                # Here it gets the 1D edges associates with the face of id "quad". Store it as a Set
                for iforc in range(
                    0, self.nforc
                ):  # iforc = 0,1,2,3 : for each forcing boundaries
                    intersection = (
                        edges & edges_forc[iforc]
                    )  # Contains the edges of the considered quad that is on the forcing boundary considered
                    if (
                        len(intersection) != 0
                    ):  # If this quad touch the forcing boundary considered
                        if type == "QUAD9":
                            # QUAD9
                            nodes = cubit.get_expanded_connectivity(
                                "face", quad
                            )  # Get all the nodes in quad including interior points
                        else:
                            # QUAD4
                            nodes = cubit.get_connectivity(
                                "face", quad
                            )  # Import the nodes describing the quad
                        nodes = self.jac_check(
                            list(nodes), self.plane_id, type
                        )  # Check the jacobian of the quad
                        for e in (
                            intersection
                        ):  # For each edge on the forcing boundary considered
                            node_edge = cubit.get_connectivity(
                                "edge", e
                            )  # Import the nodes describing the edge
                            nodes_ok = []
                            for i in nodes:  # Loop on the nodes of the quad
                                if (
                                    i in node_edge
                                ):  # If this node is belonging to forcing surface
                                    nodes_ok.append(i)  # add it to nodes_ok
                            # forcname contains 1/ element number, 2/ number of nodes that form the acoustic forcing edge
                            # (which currently must always be equal to two, see comment below),
                            # 3/ first node on the acforcing surface, 4/ second node on the acforcing surface
                            # 5/ 1 = IBOTTOME, 2 = IRIGHT, 3 = ITOP, 4 = ILEFT
                            # txt = '%10i %10i %10i %10i %10i\n' % (id_element,2,nodes_ok[0],nodes_ok[1],iforc+1)
                            txt = "%10i %10i %10i %10i %10i\n" % (
                                quad,
                                2,
                                nodes_ok[0],
                                nodes_ok[1],
                                iforc + 1,
                            )
                            # Write the id of the quad, 2 (number of nodes describing a free surface elements), the nodes and the type of boundary
                            # print(indexFile)
                            toWritetoFile[indexFile] = txt
                            indexFile = indexFile + 1
                            # forceedge.write(txt)
        forceedge.writelines(toWritetoFile)
        forceedge.close()
        print("Ok")
        cubit.cmd("set info on")  # Turn on return messages from Cubit commands
        cubit.cmd("set echo on")  # Turn on echo of Cubit commands

    def abs_write(self, absname):
        """Write absorbing surfaces on file : absname"""
        # if not absname: absname = self.absname
        absedge = open(absname, "w")
        print("Writing " + absname + ".....")
        cubit.cmd("set info off")  # Turn off return messages from Cubit commands
        cubit.cmd("set echo off")  # Turn off echo of Cubit commands
        cubit.cmd("set journal off")  # Do not save journal file.
        edges_abs = (
            [set()] * self.nabs
        )  # edges_abs[0] will be a Set containing the nodes describing bottom adsorbing boundary
        # (index 0 : bottom, index 1 : right, index 2 : top, index 3 : left)
        nedges_all = 0  # To count the total number of absorbing edges
        for block, flag in zip(self.block_bc, self.block_bc_flag):  # For each 1D block
            for iabs in range(
                0, self.nabs
            ):  # iabs = 0,1,2,3 : for each absorbing boundaries
                if (
                    block == self.abs_boun[iabs]
                ):  # If the block considered correspond to the boundary
                    edges_abs[iabs] = set(
                        cubit.get_block_edges(block)
                    )  # Store each edge on edges_abs
                    nedges_all = nedges_all + len(
                        edges_abs[iabs]
                    )  # add the number of edges to nedges_all
        toWritetoFile = [""] * (nedges_all + 1)
        toWritetoFile[0] = (
            "%10i\n" % nedges_all
        )  # Write the total number of absorbing edges to the first line of file
        # absedge.write('%10i\n' % nedges_all) # Write the total number of absorbing edges to the first line of file
        print("Number of edges", nedges_all)
        # id_element = 0
        indexFile = 1
        for block, flag in zip(self.block_mat, self.block_flag):  # For each 2D block
            quads = cubit.get_block_faces(block)  # Import quads id
            type = cubit.get_block_element_type(
                block
            )  # Contains the block element type (QUAD4...)
            for quad in quads:  # For each quad
                # id_element = id_element+1 # id of this quad
                edges = set(
                    cubit.get_sub_elements("face", quad, 1)
                )  # Get the lower dimension entities associated with a higher dimension entities.
                # Here it gets the 1D edges associates with the face of id "quad". Store it as a Set
                for iabs in range(
                    0, self.nabs
                ):  # iabs = 0,1,2,3 : for each absorbing boundaries
                    intersection = (
                        edges & edges_abs[iabs]
                    )  # Contains the edges of the considered quad that is on the absorbing boundary considered
                    if (
                        len(intersection) != 0
                    ):  # If this quad touch the absorbing boundary considered
                        if type == "QUAD9":
                            # QUAD9
                            nodes = cubit.get_expanded_connectivity(
                                "face", quad
                            )  # Get all the nodes in quad including interior points
                        else:
                            # QUAD4
                            nodes = cubit.get_connectivity(
                                "face", quad
                            )  # Import the nodes describing the quad
                        nodes = self.jac_check(
                            list(nodes), self.plane_id, type
                        )  # Check the jacobian of the quad
                        for e in (
                            intersection
                        ):  # For each edge on the absorbing boundary considered
                            node_edge = cubit.get_connectivity(
                                "edge", e
                            )  # Import the nodes describing the edge
                            nodes_ok = []
                            for i in nodes:  # Loop on the nodes of the quad
                                if (
                                    i in node_edge
                                ):  # If this node is belonging to absorbing surface
                                    nodes_ok.append(i)  # Add it to nodes_ok
                            # 'abs_surface' contains 1/ element number, 2/ number of nodes that form the absorbing edge
                            # (which currently must always be equal to 2),
                            # 3/ first node on the abs surface, 4/ second node on the abs surface
                            # 5/ 1=IBOTTOM, 2=IRIGHT, 3=ITOP, 4=ILEFT
                            # txt = '%10i %10i %10i %10i %10i\n' % (id_element,2,nodes_ok[0],nodes_ok[1],iabs+1)
                            txt = "%10i %10i %10i %10i %10i\n" % (
                                quad,
                                2,
                                nodes_ok[0],
                                nodes_ok[1],
                                iabs + 1,
                            )
                            # Write the id of the quad, 2 (number of nodes describing a free surface elements), the nodes and the type of boundary
                            toWritetoFile[indexFile] = txt
                            indexFile = indexFile + 1
                            # absedge.write(txt)
        absedge.writelines(toWritetoFile)
        absedge.close()
        print("Ok")
        cubit.cmd("set info on")  # Turn on return messages from Cubit commands
        cubit.cmd("set echo on")  # Turn on echo of Cubit commands

    def axis_write(self, axis_name):
        """Write axis on file"""
        axisedge = open(axis_name, "w")
        print("Writing " + axis_name + ".....")
        cubit.cmd("set info off")  # Turn off return messages from Cubit commands
        cubit.cmd("set echo off")  # Turn off echo of Cubit commands
        cubit.cmd("set journal off")  # Do not save journal file
        for block, flag in zip(self.block_bc, self.block_bc_flag):  # For each 1D block
            if block == self.axisId:  # If the block correspond to the axis
                edges_all = set(
                    cubit.get_block_edges(block)
                )  # Import all axis edges id as a Set
        toWritetoFile = [""] * (len(edges_all) + 1)
        toWritetoFile[0] = "%10i\n" % len(
            edges_all
        )  # Write the number of edges on the axis
        # axisedge.write('%10i\n' % len(edges_all)) # Write the number of edges on the axis
        print("Number of edges on the axis :", len(edges_all))
        # id_element = 0
        indexFile = 1
        for block, flag in zip(self.block_mat, self.block_flag):  # For each 2D block
            quads = cubit.get_block_faces(block)  # Import quads id
            type = cubit.get_block_element_type(
                block
            )  # Contains the block element type (QUAD4...)
            for quad in quads:  # For each quad
                # id_element = id_element+1 # id of this quad
                edges = set(
                    cubit.get_sub_elements("face", quad, 1)
                )  # Get the lower dimension entities associated with a higher dimension entities.
                # Here it gets the 1D edges associates with the face of id "quad". Store it as a Set
                intersection = (
                    edges & edges_all
                )  # Contains the edges of the considered quad that are on the axis
                if len(intersection) != 0:  # If this quad touch the axis
                    if type == "QUAD9":
                        # QUAD9
                        nodes = cubit.get_expanded_connectivity(
                            "face", quad
                        )  # Get all the nodes in quad including interior points
                    else:
                        # QUAD4
                        nodes = cubit.get_connectivity(
                            "face", quad
                        )  # Import the nodes describing the quad
                    nodes = self.jac_check(
                        list(nodes), self.plane_id, type
                    )  # Check the jacobian of the quad
                    for e in intersection:  # For each edge on the axis
                        node_edge = cubit.get_connectivity(
                            "edge", e
                        )  # Import the nodes describing the edge
                        nodes_ok = []
                        for i in nodes:  # Loop on the nodes of the quad
                            if i in node_edge:  # If this node is belonging to the axis
                                nodes_ok.append(i)  # Add it to nodes_ok
                        # format: #ispec_id #dump==2 #inode1 #inode2
                        txt = "%10i %10i %10i %10i\n" % (
                            quad,
                            2,
                            nodes_ok[0],
                            nodes_ok[1],
                        )
                        # txt = '%10i %10i %10i %10i\n' % (id_element,2,nodes_ok[0],nodes_ok[1])
                        # txt = '%10i %10i %10i %10i %10i\n' % (id_element,2,nodes_ok[0],nodes_ok[1],4)
                        # Write the id of the quad, 2 (number of nodes describing a free surface elements), the nodes
                        toWritetoFile[indexFile] = txt
                        indexFile = indexFile + 1
                        # axisedge.write(txt)
        axisedge.writelines(toWritetoFile)
        axisedge.close()
        print("Ok")
        cubit.cmd("set info on")  # Turn on return messages from Cubit commands
        cubit.cmd("set echo on")  # Turn on echo of Cubit commands

    def rec_write(self, recname):
        """Write receivers coordinates on file recname"""
        print("Writing " + self.recname + ".....")
        recfile = open(self.recname, "w")
        nodes = cubit.get_nodeset_nodes(
            self.receivers
        )  # Import nodes in nodeset containing receiver positions
        for i, n in enumerate(nodes):  # For each receiver
            x, y, z = cubit.get_nodal_coordinates(
                n
            )  # Import its coordinates (3 coordinates even for a 2D model in cubit)
            # plane identifier: 1 == XZ-plane, 2 == XY-plane, 3 == YZ-plane
            if self.plane_id == 1:
                txt = ("ST%i XX %20f %20f 0.0 0.0 \n") % (i, x, z)
            elif self.plane_id == 2:
                txt = ("ST%i XX %20f %20f 0.0 0.0 \n") % (i, x, y)
            else:
                txt = ("ST%i XX %20f %20f 0.0 0.0 \n") % (i, y, z)
            recfile.write(txt)  # Write 2d coordinates on the file
        recfile.close()
        print("Ok")

    def write(self, path=""):
        """Write mesh in specfem2d format"""
        print("Writing " + path + ".....")
        import os

        # cubit.cmd('set info off') # Turn off return messages from Cubit commands
        # cubit.cmd('set echo off') # Turn off echo of Cubit commands
        # cubit.cmd('set journal off') # Do not save journal file
        if len(path) != 0:  # If a path is supplied add a / at the end if needed
            if path[-1] != "/":
                path = path + "/"
        else:
            path = os.getcwd() + "/"
        self.mesh_write(path + self.mesh_name)  # Write mesh file
        self.material_write(path + self.material_name)  # Write material file
        self.nodescoord_write(path + self.nodecoord_name)  # Write nodes coord file
        self.free_write(
            path + self.freename
        )  # Write free surface file (specfem2d needs it even if there is no free surface)
        if self.abs_mesh:
            self.abs_write(path + self.absname)  # Write absorbing surface file
        if self.forcing_mesh:
            self.forcing_write(path + self.forcname)  # Write forcing surface file
        if self.axisymmetric_mesh:
            self.axis_write(path + self.axisname)  # Write axis on file
        if self.pml_layers:
            self.pmls_write(path + self.pmlname)  # Write axis on file
        if self.write_nummaterial_velocity_file:
            self.nummaterial_write(
                path + self.nummaterial_name
            )  # Write nummaterial file
        if self.receivers:
            self.rec_write(
                path + self.recname
            )  # If receivers has been set (as nodeset) write receiver file as well
        print("Mesh files has been writen in " + path)
        # cubit.cmd('set info on') # Turn on return messages from Cubit commands
        # cubit.cmd('set echo on') # Turn on echo of Cubit commands


def export2SPECFEM2D(path_exporting_mesh_SPECFEM2D="."):
    # reads in mesh from cubit
    profile = mesh()
    # writes out
    profile.write(path=path_exporting_mesh_SPECFEM2D)
    print("# END SPECFEM2D exporting process......")
