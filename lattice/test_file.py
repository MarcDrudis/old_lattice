from lattice import Lattice, SquareLattice
import numpy as np
import matplotlib.pyplot as plt


def test_lattice(lattice, name):
    print('Test', name)
    for i, site in enumerate(lattice.sites):
        lattice.in_sites(site)
        assert i == lattice.site_index(site), '{0}: {1}, {2}'.format(name, i, lattice.site_index(site))
        assert np.array_equal(site, lattice.site_vector(i)), '{0}, {1}, {2}'.format(name, site, lattice.site_vector(i))
    print(name, ': site test passed')
    for j, edge in enumerate(lattice.edges):
        lattice.in_edges(edge)
        assert j == lattice.edge_index(edge), '{0}: {1}, {2}'.format(name, j, lattice.edge_index(edge))
        assert np.array_equal(edge, lattice.edge_vector(j)), '{0}: {1}, {2}'.format(name, edge, lattice.edge_vector(i))
    print(name, ': edge test passed')


a = Lattice([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
test_lattice(a, 'a')

b = Lattice([[1, 0, 0], [0, 1, 0], [0, 0, 1]], bc='periodic')
test_lattice(b, 'b')

c = SquareLattice(layout=[3, 5, 3])
test_lattice(c, 'c')
ce1 = c.edges[13]
ce2 = c.edges[18]
ce3 = c.edges[3]

d = SquareLattice(layout=[3, 3, 3], bc='periodic')
test_lattice(d, 'd')
edge1 = d.edges[19]

e = SquareLattice([4, 3])
test_lattice(e, 'e')
e.plot_2d()
plt.show()

f = SquareLattice([3, 5, 1, 2], bc='periodic')
test_lattice(f, 'f')

g = SquareLattice([2, 2], bc='open')
test_lattice(g, 'g')
g.plot_2d()
plt.show()

h = SquareLattice(layout=[3, 5, 3], bc='semi-periodic', periodic_dims=[True, True, False])
test_lattice(h, 'h')

i = SquareLattice([2, 3], bc='periodic')
test_lattice(i, 'i')
i.plot_2d()
plt.show()

j = SquareLattice([4, 3], bc='semi-periodic', periodic_dims=[True, False])
test_lattice(j, 'j')
j.plot_2d()
plt.show()

j = SquareLattice([4])
test_lattice(j, 'j')
j.plot_2d()
plt.show()

j = SquareLattice([4], bc='periodic')
test_lattice(j, 'j')
j.plot_2d()
plt.show()

j = SquareLattice([4], bc='open')
test_lattice(j, 'j')
j.plot_2d()
plt.show()

large_lat = SquareLattice([64, 64, 64], store_sites=False)

huge_lat = SquareLattice([100, 100, 100, 100, 100], store_sites=False)