# -*- coding: utf-8 -*-
# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from copy import deepcopy
import warnings
import numpy as np
import matplotlib.pyplot as plt  # TODO: Somehow this produces issues on my Mac, therefore I use the 3 lines below

__all__ = ['Lattice', 'SquareLattice']


class Lattice():
    """Basic Lattice object

       This class describes the primitive vectors and positions of sublattice sites. All of this structural
       information can be used to build up a larger system by translation.

       Parameters
       ----------
       basis_vectors : array_like
            A list of the basis_vectors connecting different unit cells. Elements of this list should be array_like
            as well.
       layout : array_like
            A list of the number of times each basis vector is repeated along a given dimension. If None then
            the layout (1,1,..1) is inferred.
       bc : str
            A string specifying the boundary conditions. Can be one of ['closed', 'periodic']
       store_sites : bool
            Parameter to indicate whether to store the sites in a list within the Lattice object or not.
       """
    allowed_bc = ['periodic', 'closed', 'semi-periodic', 'open']

    def __init__(self, basis_vectors: list, layout: list = None, bc: str = 'closed',
                 store_sites=True, periodic_dims=None):
        """Builds a Lattice object"""
        # 1. Set basis vectors
        self._vectors = np.asarray(basis_vectors)

        # 2. Set layout
        if layout is not None:
            assert len(layout) == len(basis_vectors), 'Please specify one layout number for every dimension'
            self._layout = np.asarray(layout, dtype=np.uint64)
        else:
            self._layout = np.fromiter((1 for v in basis_vectors), dtype=np.uint8)

        # 3. Set boundary condition and periodic dimensions
        assert bc in Lattice.allowed_bc, 'Boundary condition %s is currently ' \
                                         'not supported. Try one of %s' % (bc, str(Lattice.allowed_bc))
        self._bc = bc
        if bc == 'semi-periodic':
            assert len(periodic_dims) == len(self._layout), 'Please provide periodic dimensions as a list, ' \
                                                            'e.g. [True, False, True] for a 3D lattice with' \
                                                            'periodicity along directions 0 and 2.'
            self._periodic_dims = np.asarray(periodic_dims)
        elif bc == 'closed' or bc == 'open':
            self._periodic_dims = np.fromiter((False for i in range(len(self._layout))), dtype=bool)
        elif bc == 'periodic':
            self._periodic_dims = np.fromiter((True for i in range(len(self._layout))), dtype=bool)

        # 4. Generate the sites and edges if store_sites is True.
        self._sites = []
        self._edges = []
        if store_sites:
            self.generate_sites()
            self.generate_edges()

    # BASIC LATTICE PROPERTIES
    @property
    def vectors(self) -> np.ndarray:
        """Returns the list of basis vectors that make up the unit cell of the lattice"""
        return self._vectors

    @property
    def layout(self) -> tuple:
        """Returns the number of times the unit cell is repeated along each direction"""
        return self._layout

    @property
    def boundary_cond(self) -> str:
        """Returns the lattice boundary condition"""
        return self._bc

    @property
    def periodic_dims(self) -> np.ndarray:
        """Returns a list indicating along which dimensions the lattice is periodic"""
        return self._periodic_dims

    @property
    def ndim(self) -> int:
        """Returns the dimensionality of the lattice, i.e. the number of basis vectors.

        Returns
        -------
        int
            Number of dimensions of the lattice. (as number of basis vectors)
        """
        return len(self._vectors)

    @property
    def nsites(self) -> int:
        """Returns the number of sites of the unit cell

        Returns
        -------
        int
            Number of sites in a unit cell
        """
        return int(np.prod(self.layout))

    @property
    def nedges(self) -> int:
        """Returns the number of edges in the unit cell

        Returns
        -------
        int
            Number of edges in a unit cell
        """
        # Case 1: Edges not saved. In this case a calculation needs to be performed
        if self._edges == []:
            n_dangling_edges = np.zeros(self.ndim, dtype=np.uint64)  # array of dangling edges for each dimension
            for i in range(self.ndim):
                if not self.periodic_dims[i]:
                    all_except_dim_i = np.arange(self.ndim) != i
                    n_dangling_edges[i] = np.prod(self.layout[all_except_dim_i])
            return np.uint64(self.nsites * self.ndim - sum(n_dangling_edges))

        # Case 2: Edges are saved. In this case simply return the length of the edge list.
        else:
            return len(self._edges)

    @property
    def sites(self):
        """Returns a list of the sites in the unit cell"""
        if self._sites == []:
            message = '\n Sites not saved. If you want to save all sites to the Lattice object call the ' \
                      'generate_sites class method.'
            warnings.warn(message)
        return self._sites

    @property
    def edges(self):
        """Returns a list of the edges in the unit cell in the format [site_index, direction]

        Examples
        --------
            [3, 1] denotes the edge going from site with index 3 into direction 1.
        """
        if self._edges == []:
            message = '\n Edges not saved. If you want to save all edges to the Lattice object call the ' \
                      'generate_edges class method.'
            warnings.warn(message)
        return self._edges

    # HELPER FUNCTIONS:
    def _site_counter(self, direction: int) -> int:
        """Returns the number by which the site index increases when moving
        along specified direction in the lattice"""
        assert isinstance(direction, (int, np.integer)), 'Direction must be an integer-type'
        assert 0 <= direction < self.ndim, 'Direction %d out of bounds for lattice of ndim %d' % (direction, self.ndim)

        count = 1
        for i in range(direction):
            count *= self.layout[i]
        return count

    def _nsites_perp_to_direction(self, direction: int) -> int:
        """Returns the number of sites in a hyperplane perpendicular to direction i"""
        all_except_dim_i = np.arange(self.ndim) != direction
        return int(np.prod(self.layout[all_except_dim_i]))

    # COPY METHOD
    def copy(self):
        """Return a copy of self"""
        return deepcopy(self)

    # CONSOLE OUTPUT
    def __str__(self):
        """Returns the basic facts about the given lattice in string format"""
        full_str = 'Lattice object \n'
        full_str += 'Dim:'.ljust(14) + '%d \n' % self.ndim
        full_str += 'Layout:'.ljust(14) + '%s \n' % str(self.layout)
        full_str += 'Nsites:'.ljust(14) + '%d \n' % self.nsites
        full_str += 'Nedges:'.ljust(14) + '%d \n' % self.nedges
        full_str += 'BC:'.ljust(14) + '%s \n' % self.boundary_cond
        if self.boundary_cond == 'semi-periodic':
            full_str += 'Periodic dim:'.ljust(14) + str(self.periodic_dims) + '\n'
        full_str += 'Basis: %s' % str(self.vectors)
        return full_str

    def __repr__(self):
        """Lattice object summary to appear directly as console output"""
        return self.__str__()

    # SITE FUNCTIONALITY
    def in_sites(self, site: np.ndarray):
        """Throws assertion if 'site' is not a valid unit cell lattice site

        Parameters
        ----------
        site : array-like
            Index vector of the site
        """
        assert len(site) == self.ndim, 'Array of length %d cannot be used as site index for lattice of dimension %d' % \
                                       (len(site), self.ndim)

        for i, xi, ni in zip(range(self.ndim), site, self.layout):
            assert isinstance(xi, (int, np.integer)), 'Lattice vectors must have int ' \
                                                      'components.'
            assert 0 <= xi < ni, 'Site %s is not in unit cell. ' \
                                 'Index %d out of range for dimension %d with %d sites. ' % (str(site), xi, i, ni)

    def project(self, site: np.ndarray) -> np.ndarray:
        """Projects a site to the unit cell. Used iff periodic boundary conditions are chosen.

        Parameters
        ----------
        site : array-like
            Index vector of the site to project

        Returns
        -------
        projected_site : array-like
            The projection of 'site' to the unit cell is returned in index vector format.
        """
        assert len(site) == self.ndim, 'Array of length %d cannot be used as index for lattice of dimension %d' % \
                                       (len(site), self.ndim)

        projected_site = np.zeros(self.ndim, dtype=np.int32)

        for i, xi, ni in zip(range(self.ndim), site, self.layout):
            assert isinstance(xi, (int, np.integer)), 'Lattice vectors must have int ' \
                                                      'components.'
            projected_site[i] = xi % ni

        return projected_site

    def generate_sites(self):
        """Generate all lattice sites in the unit cell in the format of index vectors and saves them to
        the Lattice object.

        Examples
        --------
        [1,2,0] indicates the site at 1 * self._vectors[0] + 2 * self._vectors[1] + 3 * self._vectors[2] in a 3d
        lattice.
        """
        # add first site manually
        self._sites = []

        for index in range(self.nsites):
            v = np.zeros(self.ndim, dtype=np.int32)
            temp = index

            for direction in range(self.ndim - 1, -1, -1):  # iterate downwards
                v[direction] = int(temp / self._site_counter(direction))
                temp = temp % self._site_counter(direction)
            self._sites.append(v)
        self._sites = np.asarray(self._sites)

    def site_index(self, site: np.ndarray) -> int:
        """Returns the index of 'site' in the list of lattice sites. This index is used e.g. for the
        Jordan-Wigner transform.
        Reverse of self.site_vector

        Parameters
        ----------
        site : array-like
            Index vector of the site

        Returns
        -------
        lattice_index : int
            Site index indicating the position of the site in self._sites
        """
        self.in_sites(site)
        lattice_index = 0

        jump_factors = [self._site_counter(i) for i in range(self.ndim)]

        for jump, xi in zip(jump_factors, site):
            lattice_index += int(jump * xi)
        return lattice_index

    def site_vector(self, index: int) -> np.ndarray:
        """Returns the coordinate vector of the lattice at site with the given index.
        Reverse of self.site_index

        Parameters
        ----------
        index : int
            Site index indicating the position of the site in self._sites

        Returns
        -------
        list
            A list with the vector components of the site indexed by 'index'
        """
        assert 0 <= index < self.nsites, 'Index %d not contained in unit cell. Max index: %d' % (index, self.nsites - 1)
        v = np.zeros(self.ndim, dtype=int)
        temp = index

        for direction in range(self.ndim - 1, -1, -1):  # iterate downwards
            v[direction] = int(temp / self._site_counter(direction))
            temp = temp % self._site_counter(direction)
        return v

    def neighbors(self, site: np.ndarray) -> list:
        """Returns the neighbors of the specified site

        Parameters
        ----------
        site : array-like
            Index vector of the site

        Returns
        -------
        neighbor_list : array-like
            List of index vectors of the neighboring sites of 'site', i.e. of all sites that are connected to
            'site' with exactly 1 edge.
        """
        self.in_sites(site)

        neighbor_list = []

        for i in range(self.ndim):
            # if the site has a `left`s-neighbor along dimension i
            if site[i] > 0 or self.periodic_dims[i]:
                prev_site = deepcopy(site)  # instantiate a list for the previous site along dimension i
                prev_site[i] = (prev_site[i] - 1) % self.layout[i]  # lower the ith component by 1
                neighbor_list.append(prev_site)  # append the site to the list of neighbors

            # if the site has a `right`-neighbor along dim i
            if site[i] < self.layout[i] - 1 or self.periodic_dims[i]:
                next_site = deepcopy(site)
                next_site[i] = (next_site[i] + 1) % self.layout[i]
                neighbor_list.append(next_site)

        return neighbor_list

    def is_boundary_site(self, site: np.ndarray) -> bool:
        """Returns True iff `site` is a boundary site.
        (boundary sites occur only in the case of `open` or `closed` boundary conditions)"""
        self.in_sites(site)

        for dim in range(self.ndim):
            if site[dim] == 0 or site[dim] == self.layout[dim] - 1:
                if not self.periodic_dims[dim]:
                    return True
        return False

    def is_boundary_along(self, site: np.ndarray, dim: int, direction='positive') -> bool:
        """Returns True iff `site` is a boundary site along dimension `dim`.
        (boundary sites occur only in the case of `open` or `closed` boundary conditions)"""
        self.in_sites(site)

        # 1. filter out the periodic cases (no sites at boundary)
        if self.periodic_dims[dim]:
            return False

        # 2. Treat the cases where sites at the boundary are possible:
        #    If we are at the last site along `dim` then walking in positive direction we hit a boundary:
        is_at_positive_boundary = (site[dim] == self.layout[dim]-1)
        #    If we are at the first site along `dim` then walking in negative direction we hit a boundary:
        is_at_negative_boundary = (site[dim] == 0)

        # 3. Return the corresponding truth value
        if direction == 'positive':
            return is_at_positive_boundary
        elif direction == 'negative':
            return is_at_negative_boundary
        elif direction == 'both':
            return is_at_negative_boundary or is_at_positive_boundary
        else:
            # Catch any disallowed direction statements
            raise UserWarning("'direction' must be one of ['positive', 'negative', 'both']")

    # EDGE FUNCTIONALITY
    def in_edges(self, edge: np.ndarray):
        """Throws an assertion if edge is not in lattice.

        Parameters
        ----------
        edge : list
            Edge in the format [site_index, direction]
            E.g. [0, 1] means the edge going from the site indexed by 0 (self._sites[0]) in direction 1.
        """
        # Check if the start_site is in the lattice
        start_site = self.site_vector(edge[0])
        self.in_sites(start_site)

        # Check whether the direction of the edge is valid
        # Case 1: Open boundary conditions
        if self.boundary_cond == 'open':
            # Check if direction is valid (incoming edges have negative directions)
            assert -self.ndim <= edge[1] < self.ndim, 'Lattice of dim {0}' \
                                                      ' has no dimension {1}'.format(self.ndim, edge[1])

            # In the case of an incoming edge (from outside), check that the start_site is at the lattice boundary
            if edge[1] < 0:
                dim_i = -edge[1] - 1
                assert start_site[dim_i] == 0, 'Site is not at incoming lattice boundary along ' \
                                               'dim {0}'.format(dim_i)

        # Case 2: Other boundary conditions
        else:
            # Check if direction is valid
            direction = edge[1]
            assert 0 <= direction < self.ndim, 'Lattice of dim {0} has no dimension {1}'.format(self.ndim, direction)
            # If we are at an edge at the boundary, check whether the given direction is periodic
            if start_site[direction] == self.layout[direction] - 1:
                assert self.periodic_dims[direction], 'No periodicity along direction {0}. Therefore edge {1}' \
                                                      'is not in Lattice of size {2}.'.format(direction, edge,
                                                                                              self.layout)

    def is_boundary_edge(self, edge: np.ndarray) -> bool:
        """Returns True iff `edge` is a boundary edge
        (boundary edges occur only in the case of `open` boundary conditions)"""
        self.in_edges(edge)
        if not self.boundary_cond == 'open':
            return False  # in this case no edge connects to outside
        else:
            site = self.site_vector(edge[0])
            direction = edge[1]
            if direction < 0:
                return True  # in this case we have an incoming edge
            elif site[direction] == self.layout[direction] - 1:
                return True  # in this case we have outgoing edge at the lattice boundarys
            else:
                return False  # edge in the bulk

    def generate_edges(self):
        """Generate the edges of the lattice sites in the format: [site_index, direction]

        Examples
        --------
        [0, 1] means the edge going from the site indexed by 0 (self._sites[0]) in direction 1.
        """
        outgoing_edges = [[] for i in range(self.ndim)]
        # For open boundary conditions also include incoming edges
        if self.boundary_cond == 'open':
            incoming_edges = [[] for i in range(self.ndim)]

        for dim_i in range(self.ndim):
            # Go through all sites and find add the connected edges per site along dim_i
            for site_index, site in enumerate(self._sites):

                # For open boundary conditions  include an incoming edge coming from no site for
                # the first site along a given dimension
                if self.boundary_cond == 'open' and site[dim_i] == 0:
                    incoming_edges[dim_i].append([site_index, -dim_i - 1])  # noted down with negative direction

                # If site is not the last along dim_i add an outgoing edge in this direction
                if self.periodic_dims[dim_i] or self.boundary_cond == 'open' or site[dim_i] < self.layout[dim_i] - 1:
                    outgoing_edges[dim_i].append([site_index, dim_i])

        outgoing_edges = np.concatenate(outgoing_edges)
        self._edges = outgoing_edges

        # For open boundary conditions, prepend the incoming edges along dim_i to the list
        if self.boundary_cond == 'open':
            incoming_edges = np.concatenate(incoming_edges)
            self._edges = np.vstack((incoming_edges, outgoing_edges))

        # # OLD
        # self._edges = []
        # for site in self._sites:
        #    for dim_i in range(self.ndim):
        #        # if site is not the last along a dimension i add an edge along this direction of
        #        if self.periodic_dims[dim_i] or site[dim_i] < self.layout[dim_i] - 1:
        #            self._edges.append([site, dim_i])

    def edge_vector(self, index: int) -> np.ndarray:
        """Returns the edge label in the format [site_index, direction] of the edge stored in self._edges at the given
        'index' value.
        Reverse of self.edge_index.

        Parameters
        ----------
        index : int
            Index at which the 'edge' is stored

        Returns
        -------
        np.ndarray
            Edge in the format [site, direction]
            E.g. [0, 1] means the edge going from the site indexed by 0 (self._sites[0]) in direction 1.
        """
        return self._edges[index]  # Todo: make more efficient than list lookup

    def edge_index(self, edge: np.ndarray) -> int:
        """Returns the index of 'edge' in the list of edge sites.
        Reverse of self.edge_vector.

        Parameters
        ----------
        edge : list,
            Edge in the format [site_index, direction]
            E.g. [0, 1] means the edge going from the site indexed by 0 (self._sites[0]) in direction 1.

        Returns
        -------
        int
            Index value of 'edge' in the list self._edges stored in the Lattice object.
        """
        # Make sure to get the right format
        edge = np.asarray(edge)
        # Check that edge is contained in lattice
        self.in_edges(edge)

        # TODO come up with a fast, direct method
        for i, other_edge in enumerate(self.edges):
            if np.array_equal(edge, other_edge):
                return i

    def edge_endsites(self, edge: np.ndarray, project=False):
        """Returns the two sites connected by 'edge'. Assumes edge format [site_index, direction]
        Does not check whether the returned edge_sites are in the lattice.

        Parameters
        ----------
        edge : numpy.ndarray
            Edge in the format [site, direction].
            E.g. [0, 1] means the edge going from the site indexed by 0 (self._sites[0]) in direction 1.
        project : bool, optional
            If True then the returned edge sites are projected to within the unit cell if they lie outside
            the unit cell.

        Returns
        -------
        start_site : numpy.ndarray
            The site on the LHS of the edge.
        end_site : numpy.ndarray
            The site on the RHS of the edge.
        """
        edge = np.asarray(edge)
        start_site = self.site_vector(edge[0])
        direction = edge[1]

        # Case 1: Special case of incoming edges in the case of open boundary conditions
        if self.boundary_cond == 'open' and edge[1] < 0:
            warnings.warn('Incoming edge on a `Lattice` with `open` boundary conditions has no two endsites')
            end_site = deepcopy(start_site)
            end_site[-direction - 1] -= 1  # Note: In this case end_site is not within the `Lattice`.
            return start_site, end_site

        # Case 2: General case of outgoing edges
        assert 0 <= edge[1] < self.ndim, 'Direction %d out of bounds for lattice of dimension %d' % (edge[1], self.ndim)
        end_site = deepcopy(start_site)
        end_site[direction] += 1

        # Project to unit cell if `project` flag is True.
        if project:
            return self.project(start_site), self.project(end_site)

        return start_site, end_site

    def edge_from_endsites(self, site1: np.ndarray, site2: np.ndarray) -> np.ndarray:
        """Returns the directed 'edge' from site1 to site2. site1 must be a valid lattice site, site2
        may be outside the lattice unit cell in the case of `open` boundary conditions.

        Parameters
        ----------
        site1 : array-like
            Index vector of site 1, must be in the lattice unit cell
        site2 : array-like
            Index vector of site 2

        Returns
        -------
        numpy array
            The directed edge connecting site1 and site2 in format [start_site, direction].
        """
        site1 = np.asarray(site1)
        site2 = np.asarray(site2)

        diff = site2 - site1
        direction = np.flatnonzero(diff)
        assert len(direction) == 1, 'Invalid edge direction.'

        site_index = self.site_index(site1)

        # Cases:
        # diff positive --> move in forward direction --> diff must be 1 for a valid edge in the lattice
        if diff[direction] > 0:
            assert diff[direction] == 1, 'The given `site2` is more than one lattice spacing away form `site1`.' \
                                         'Therefore there is no valid lattice edge connecting `site2` and `site1`.'
            return np.asarray([site_index, direction], dtype=np.int32)
        # diff negative --> move in backward direction (negative direction for open b.c.)
        else:
            return np.asarray([site_index, -(direction + 1)], dtype=np.int32)

    # PLOTTING FUNCTIONALITY
    def site_to_real_space(self, site: list) -> np.ndarray:
        """Converts an abstract site index to the actual vector. Used for plotting.

        Parameters
        ----------
        site : array-like
            A vector of the indices labelling the site in terms of the lattice basis vectors

        Returns
        -------
        numpy.ndarray
            A numpy array which contains the lattice sites coordinates in real space as floating
            point values.
        """
        basis_vectors = np.array([np.array(v).squeeze() for v in self.vectors]).transpose()
        site_vector = np.array(site).squeeze()

        return np.matmul(basis_vectors, site_vector)

    def plot_2d(self, annotate_sites=True, annotate_edges=True):
        """Plots closed lattices in 1 or 2 dimensions.

        Parameters
        ----------
        annotate_sites : bool
            If true, site labels as they appear in self._sites will be shown next to each site.
        annotate_edges : bool
            If true, edge labels as they appear in self._edges will be shown next to each
            edge in the format (site_index, direction).

        Returns
        -------
        matplotlib axes object
        """
        assert self.ndim <= 2, 'Only works for 1d and 2d-lattices.'
        assert len(self.sites) > 0, 'Sites are empty'
        fig, ax = plt.subplots()

        # Color periodic dimensions in green
        if self.boundary_cond == 'periodic' or self.boundary_cond == 'semi-periodic':
            boundary_color = 'green'
        else:
            boundary_color = 'red'

        if self.ndim == 2:
            for i, site in enumerate(self.sites):
                x, y = self.site_to_real_space(site)
                ax.scatter(x, y, color='blue', alpha=0.4)
                if annotate_sites:
                    ax.annotate(self.site_index(site), (x + 0.02, y + 0.02))

            for edge in self._edges:
                # Handle the case of incoming edges (negative directions)
                site1, site2 = self.edge_endsites(edge, project=True)
                x1, y1 = site1[0], site1[1]
                x2, y2 = site2[0], site2[1]
                x1r, y1r = self.site_to_real_space([x1, y1])
                x2r, y2r = self.site_to_real_space([x2, y2])

                # Flag incoming edges
                incoming_flag = False
                if edge[1] < 0:
                    incoming_flag = True

                if incoming_flag:
                    if (x2 < x1):
                        print(x1, x2)
                        x2r, y2r = self.site_to_real_space([x2 + 0.7, y2])
                        ax.plot([x2r, x1r], [y2r, y1r], color=boundary_color, alpha=0.2)
                        if annotate_edges:
                            ax.annotate(str(edge),
                                        ((x1r + x2r) / 2., (y1r + y2r) / 2.))
                    if (y2 < y1):
                        print(x1, x2)
                        x2r, y2r = self.site_to_real_space([x2, y2 + 0.7])
                        ax.plot([x2r, x1r], [y2r, y1r], color=boundary_color, alpha=0.2)
                        if annotate_edges:
                            ax.annotate(str(edge),
                                        ((x1r + x2r) / 2., (y1r + y2r) / 2.))


                elif (x2 < x1):  # horizontal boundary edge
                    x3r, y3r = self.site_to_real_space([x1 + 0.3, y2])
                    x4r, y4r = self.site_to_real_space([x2 - 0.3, y1])
                    ax.plot([x1r, x3r], [y1r, y3r], color=boundary_color, alpha=0.2)
                    ax.plot([x4r, x2r], [y4r, y2r], color=boundary_color, alpha=0.2)
                    if annotate_edges:
                        ax.annotate(str(edge),
                                    ((x1r + x3r) / 2., (y1r + y3r) / 2.))

                elif (y2 < y1):  # vertical boundary edge
                    x3r, y3r = self.site_to_real_space([x2, y1 + 0.3])
                    x4r, y4r = self.site_to_real_space([x1, y2 - 0.3])
                    ax.plot([x1r, x3r], [y1r, y3r], color=boundary_color, alpha=0.2)
                    ax.plot([x4r, x2r], [y4r, y2r], color=boundary_color, alpha=0.2)
                    if annotate_edges:
                        ax.annotate(str(edge),
                                    ((x1r + x3r) / 2., (y1r + y3r) / 2.))

                else:  # normal case for edge in bulk
                    ax.plot([x1r, x2r], [y1r, y2r], color='red', alpha=0.2)
                    if annotate_edges:
                        ax.annotate(str(edge),
                                    ((x1r + x2r) / 2., (y1r + y2r) / 2.))

        elif self.ndim == 1:
            ax.scatter(self._sites, np.zeros_like(self._sites))
            for i, site in enumerate(self._sites):
                if annotate_sites:
                    ax.annotate(self.site_index(site), (site[0] + 0.02, 0.001))

            for edge in self._edges:
                # Flag incoming edges
                incoming_flag = False
                if edge[1] < 0:
                    incoming_flag = True

                x1, x2 = self.edge_endsites(edge, project=True)

                if incoming_flag:
                    x2 = x2 + 0.7
                    ax.plot([x2, x1], [0, 0], color=boundary_color, alpha=0.2)
                    if annotate_edges:
                        ax.annotate(str(edge), (x1 - 0.3, 0))

                elif (x2 < x1):  # horizontal boundary edge
                    ax.plot([x1, x1 + 0.3], [0, 0], color=boundary_color, alpha=0.2)
                    ax.plot([x2 - 0.3, x2], [0, 0], color=boundary_color, alpha=0.2)
                    if annotate_edges:
                        ax.annotate(str(edge), (x1 + 0.2, 0))

                else:
                    ax.plot([x1, x2], [0, 0], color='red', alpha=0.2)
                    if annotate_edges:
                        ax.annotate(str(edge), ((x1 + x2) / 2., 0))

        plt.title('Lattice ' + str(self.layout))
        return ax


class SquareLattice(Lattice):
    """Subclass of Lattice
    Provides a simpler syntax for creating square lattices.
    """

    def __init__(self, layout: list, lattice_const: float = 1., bc: str = 'closed', store_sites=True,
                 periodic_dims=None):
        # Set basis vectors
        basis_vectors = []
        for i in range(len(layout)):
            ei = np.zeros(len(layout))
            ei[i] = lattice_const
            basis_vectors.append(ei)

        Lattice.__init__(self, basis_vectors, layout, bc, store_sites, periodic_dims)

    def __str__(self):
        """Returns the basic facts about the given lattice"""
        full_str = 'SquareLattice object \n'
        full_str += 'Dim:'.ljust(14) + '%d \n' % self.ndim
        full_str += 'Layout:'.ljust(14) + '%s \n' % str(self.layout)
        full_str += 'Nsites:'.ljust(14) + '%d \n' % self.nsites
        full_str += 'Nedges:'.ljust(14) + '%d \n' % self.nedges
        full_str += 'BC:'.ljust(14) + '%s \n' % self.boundary_cond
        if self.boundary_cond == 'semi-periodic':
            full_str += 'Periodic dim:'.ljust(14) + str(self.periodic_dims) + '\n'
        full_str += 'Basis: %s' % str(self.vectors)
        return full_str