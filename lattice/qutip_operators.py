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
import qutip as qt
import numpy as np

__all__ = ['identity_string', 'site_operator', 'edge_operator', 'jordan_wigner_transform',
           'psi', 'psi_dag', 'S_plus', 'S_minus', 'S_z']


def identity_string(n, subsys_dim=2):
    """Retunrs a string of `n` identity operators, each with dimension `subsys_dim`."""
    if n == 0: return 1
    return qt.tensor([qt.identity(subsys_dim) for i in range(n)])


def site_operator(site, operator, lattice, edge_dim=2):
    """Implements `operator` on a single lattice site."""

    # locate site in lattice site list
    if isinstance(site, (int, np.integer)):
        assert site < lattice.nsites
        site_index = site
    else:
        site_index = lattice.site_index(site)
    site_dim = operator.dims[0][0]

    # add identity operators on sites before, after and on edges
    if 0 < site_index < lattice.nsites - 1:
        pre_identity = identity_string(site_index, site_dim)
        post_identity = qt.tensor([identity_string(lattice.nedges, edge_dim),
                                   identity_string(lattice.nsites - site_index - 1, site_dim)])
        return qt.tensor([post_identity, operator, pre_identity])

    elif site_index == 0:
        post_identity = qt.tensor([identity_string(lattice.nsites - 1, site_dim),
                                   identity_string(lattice.nedges, edge_dim)])
        return qt.tensor([post_identity, operator])

    elif site_index == lattice.nsites - 1:
        pre_identity = identity_string(site_index, site_dim)
        post_identity = identity_string(lattice.nedges, edge_dim)
        return qt.tensor([post_identity, operator, pre_identity])


def edge_operator(edge, operator, lattice, site_dim=2):
    """Implements `operator` on a single lattice site."""

    # locate edge in lattice edge list
    # case 1:  if `edge` is given in (index) format
    if isinstance(edge, (int, np.integer)):
        assert edge < lattice.nedges
        edge_index = edge
    # case 2: if `edge` is given in (site, direction) format
    else:
        edge_index = lattice.edge_index(edge)

    # get the Hilbertspace dimension for a edge from the shape of the operator
    edge_dim = operator.dims[0][0]

    # handle the exceptional case where the lattice has only 1 edge
    if lattice.nedges == 1:
        return qt.tensor([operator, identity_string(lattice.nsites, site_dim)])

    # handle all other cases (more than 1 edge)
    # add identity operators on sites before, and edges before and after
    if 0 < edge_index < lattice.nedges - 1:
        pre_identity = qt.tensor([identity_string(edge_index, edge_dim),
                                  identity_string(lattice.nsites, site_dim)])
        post_identity = identity_string(lattice.nedges - edge_index - 1, edge_dim)
        return qt.tensor([post_identity, operator, pre_identity])

    elif edge_index == 0:
        pre_identity = identity_string(lattice.nsites, site_dim)
        post_identity = identity_string(lattice.nedges - 1, edge_dim)
        return qt.tensor([post_identity, operator, pre_identity])

    elif edge_index == lattice.nedges - 1:
        pre_identity = qt.tensor([identity_string(edge_index, edge_dim),
                                  identity_string(lattice.nsites, site_dim)])
        return qt.tensor([operator, pre_identity])


def jordan_wigner_transform(j: int, lattice_length: int):
    """Performs the jordan wigner transform of the lowering operator psi_j in arbitrary dimensions"""
    sigmaz = qt.sigmaz()
    sigmam = qt.sigmam()
    identity = qt.identity(2)
    assert j < lattice_length, 'No site %d in lattice of length %d. Indexing from 0.' % (j, lattice_length)

    operators = []
    for k in range(j):
        operators.append(sigmaz)
    operators.append(sigmam)
    for k in range(lattice_length - j - 1):
        operators.append(identity)

    return -qt.tensor(operators[::-1])


def psi(site, lattice, edge_dim=2):
    """Returns lowering operator on site `site`"""
    # locate site in lattice site list
    if isinstance(site, (int, np.integer)):
        assert site < lattice.nsites
        site_index = site
    else:
        site_index = lattice.site_index(site)

    if edge_dim == 0:
        return qt.tensor([jordan_wigner_transform(site_index, lattice.nsites)])
    else:
        return qt.tensor([identity_string(lattice.nedges, edge_dim),
                          jordan_wigner_transform(site_index, lattice.nsites)])


def psi_dag(site, lattice, edge_dim=2):
    """Returns raising operator on site `site`"""
    # locate site in lattice site list
    if isinstance(site, (int, np.integer)):
        assert site < lattice.nsites
        site_index = site
    else:
        site_index = lattice.site_index(site)

    if edge_dim == 0:
        return qt.tensor([qt.dag(jordan_wigner_transform(site_index, lattice.nsites))])
    else:
        return qt.tensor([identity_string(lattice.nedges, edge_dim),
                          qt.dag(jordan_wigner_transform(site_index, lattice.nsites))])


def S_plus(edge, lattice, S=1 / 2):
    """Returns the S+ operator on edge `edge` for a spin S system"""
    return edge_operator(edge, qt.jmat(S, '+'), lattice, site_dim=2)


def normalized_plus(edge, lattice, S=1 / 2):
    normed_plus = qt.Qobj(inpt=np.diag(np.ones(int(2 * S)), 1))
    return edge_operator(edge, normed_plus, lattice, site_dim=2)


def normalized_minus(edge, lattice, S=1 / 2):
    normed_minus = qt.Qobj(inpt=np.diag(np.ones(int(2 * S)), -1))
    return edge_operator(edge, normed_minus, lattice, site_dim=2)


def S_minus(edge, lattice, S=1 / 2):
    """Returns the S- operator on edge `edge` for a spin S system"""
    return edge_operator(edge, qt.jmat(S, '-'), lattice, site_dim=2)


def S_z(edge, lattice, S=1 / 2):
    """Returns the Sz operator on edge `edge` for a spin S system"""
    return edge_operator(edge, qt.jmat(S, 'z'), lattice, site_dim=2)


def identity(lattice, site_dim=2, edge_dim=2):
    """Returns the identity operator on the entire lattice."""
    if edge_dim == 0:
        return qt.tensor([identity_string(lattice.nsites, site_dim)])
    else:
        return qt.tensor([identity_string(lattice.nedges, edge_dim),
                          identity_string(lattice.nsites, site_dim)])
