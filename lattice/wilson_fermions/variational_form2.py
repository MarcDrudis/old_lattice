from qiskit.aqua.components.variational_forms import VariationalForm
#from qiskit.aqua.utils.validation import validate_min, validate_in_set
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.aqua.operators import WeightedPauliOperator
import numpy as np
import qutip as qt

from lattice.operators.qiskit_aqua_operator_utils import operator_sum
from lattice.wilson_fermions.basic_operators import psi, psidag, U, Udag, standard_basis, fermion_id, link_id
from lattice import Lattice
from lattice.wilson_fermions.qiskit_utils import construct_trotter_step

__all__ = ['WilsonLGT2',"WilsonLGT6_double_string_pair_S_1_5","WilsonLGT6_double_string_pair_N_1","WilsonLGT4_new_pair",'WilsonStringLike',"WilsonLGT6_double_string_pair","WilsonLGT6_double_string", "WilsonLGT6_string","WilsonLGT6","WilsonLGT4","WilsonLGT4_new",'uniform_sampler', "WilsonHardwareEfficientLGT", "hopping_like"]

# TODO 1: Construct a parametrized evolution circuit (from operator.construct_evolution_circuit) - LATER
# TODO 2: Adapt WilsonLGT to work with hopping_like terms instead of requiring hopping terms - NOW - DONE
# TODO 3: Decide whether to build a pluggable circuit for the hopping terms e.g. for (XIYZI)IIIXIIYII,
#       but this should basically work anyways if I parametrize my evolution circuit
# TODO 4: Implement a qutip/matrix method that does the same thing - NOW - DONE
# TODO 5: Benchmark my circuit on the VM (for S=0, 0.5, 1.) and closed lattices: - NOW
#            x---x
#            |   |      x--x        x--x--x         x--x--x--x
#            x---x
# TODO 6: See if I want to implement the Bravyi-kitaev / parity mappings for fermionic operators - LATER
# TODO 7: Implement the linear encoding of the spins for the S = whole-number cases. - LATER


################################################################################################
# 1. Set up functionality to construct the hopping terms form the lattice
################################################################################################

def atleast_4d(*arys):
    """
    View inputs as arrays with at least four dimensions. An extension of numpys atleast_1d, atleast_2d, atleast_3d.
    Arrays with fewer than 4 dimensions have 4-ary.ndim new axes of length 1 prepended.

    Args:
        ary1, ary2, ... (array_like): One or more array-like sequences.  Non-array inputs are converted to
            arrays.  Arrays that already have four or more dimensions are preserved.

    Returns:
        res (list or np.ndarray):
            If one array was given as input, then this array is returned interpreted as 4d np.ndarray.
            If several arrays were given as input, then a list is returned in which each of these arrays is
            interpreted as 4d np.ndarray.
            Arrays with fewer than 4 dimensions have 4-ary.ndim new axes of length 1 prepended.
    """
    newaxis = np.newaxis
    res = []
    for ary in arys:
        ary = np.asanyarray(ary)
        if ary.ndim == 0:
            result = ary.reshape(1, 1, 1, 1)
        elif ary.ndim == 1:
            result = ary[newaxis, newaxis, newaxis, :]
        elif ary.ndim == 2:
            result = ary[newaxis, newaxis, :,:]
        elif ary.ndim == 3:
            result = ary[newaxis, :, :, :]
        else:
            result = ary
        res.append(result)
    if len(res) == 1:
        return res[0]
    else:
        return res

def phase_like(site, lattice, S, copmat=np.array([[1,0],[0,1]]), output='qiskit'):
    # generate phase term, note if copmat=identity then it is just the site number operator
    nqubit=lattice.nsites * 2 + lattice.nedges * int(np.ceil(np.log2(2 * S + 1)))

    summands = []
    # 1. sum over all spinor components at the given lattice site
    for alpha in range(2):
        for beta in range(2):
            summands.append(psidag(site, alpha, lattice) * copmat[alpha, beta] * psi(site, beta,lattice))
    mx = operator_sum(summands)

    if S > 0:
        mx = mx @ link_id(S, lattice)
    return  mx.to_qubit_operator(output=output)

def pair_like(site, lattice, S, output='qiskit'):
    pair_op= psidag(site, 0, lattice= lattice) * psi(site, 1, lattice= lattice)
    if S > 0:
        pair_op= pair_op @ link_id(S, lattice)
    pair_op=pair_op.to_qubit_operator(output=output)

    if output == 'qiskit':
        # nota che come un pirla devo fare sta cosa, ma basterebbe aggiungere l'hermitian conjugate ! 
        for i,pair in enumerate(pair_op.paulis):
               if np.imag(pair[0]) != 0 :
                    pair_op.paulis[i][0] *= -1j
        pair_op.chop()
  
    return pair_op


def hopping_like(edge, lattice, S, mixmat, hopp_coeff=1., output='qiskit'):
    """
    Generates the hopping like terms along the given `edge` in `lattice` scaled by
    the `hopp_coeff`.

    Args:
        edge (np.ndarray): The edge in the lattice along which the hopping takes place
        lattice (Lattice): The lattice object on which the lattice gauge theory is defined.
        S (float): The spin truncation of the Quantum Link model
        mixmat (np.ndarray): The mixing matrix between the hopping terms. Must be of shape
            (ncomp, ncomp) where ncomp is the number of components of the spinors used in the
            lattice gauge theory.
        hopp_coeff (float): A scaling parameter for the hopping-like term
        output (str): The desired output format. Must be one of ['qiskit', 'qutip', 'matrix', 'spmatrix']

    Returns:
        (qiskit.aqua.Operator or qutip.Qobj or np.ndarray)
        The desired hopping_like term in the format specified by `output`
    """

    # 1. Extract the site and direction of the edge along which hopping occurs:
    site = lattice.site_vector(edge[0])
    hopping_dim = edge[1]

    # 2. Generate the hopping term as a sum over the fermionic (spinor) components
    summands = []
    # 2.1 Check for boundary sites
    # Treat open and closed boundary conditions (skip term if edge is at boundary)
    is_at_boundary = lattice.is_boundary_along(site, hopping_dim, direction='positive')
    if is_at_boundary:
        # print('encountered boundary at {}, {}'.format(site, j))
        raise UserWarning('The given `site` and `hopping_dim` combination goes outside the lattice.')

    # 2.2Get the edge along which hopping takes place and the next site
    next_site = lattice.project(site + standard_basis(hopping_dim, lattice.ndim))

    # 3 Sum over all spinor components:
    for alpha in range(2):
        for beta in range(2):
            # Skip cases with zero coefficients
            if mixmat[alpha, beta] == 0:
                # print('skipped ab {}, {}'.format(alpha, beta))
                continue

            # Generate the fermionic hopping terms
            bwd_hopp = (hopp_coeff
                        * psidag(site, alpha, lattice)
                        * mixmat[alpha, beta]
                        * psi(next_site, beta, lattice))
            fwd_hopp = bwd_hopp.dag()

            # If gauge field is present, tensor with the hopping terms with the link operators
            if S > 0:
                bwd_hopp = bwd_hopp @ U(edge, S, lattice)
                fwd_hopp = fwd_hopp @ Udag(edge, S, lattice)

            # Convert the operators to qubit operators and add them to the total hopping term
            if not output == 'abstract':
                bwd_hopp = bwd_hopp.to_qubit_operator(output=output)
                fwd_hopp = fwd_hopp.to_qubit_operator(output=output)

            summands += [fwd_hopp, bwd_hopp]

    # 5. Sum up all hopping terms and return the hopping hamilton_qiskit
    hopping_term = operator_sum(summands)
    if output == 'qiskit':
        hopping_term.chop()
    return hopping_term


def string_like(edges, lattice, S, mixmat, hopp_coeff=1., output='qiskit'):
    """
    Generates the hopping like terms along the given `edge` in `lattice` scaled by
    the `hopp_coeff`.

    Args:
        edges (list(np.ndarray)): The ordered list of edges in the lattice along which the string is constructed
        lattice (Lattice): The lattice object on which the lattice gauge theory is defined.
        S (float): The spin truncation of the Quantum Link model
        mixmat (np.ndarray): The mixing matrix between the hopping terms. Must be of shape
            (ncomp, ncomp) where ncomp is the number of components of the spinors used in the
            lattice gauge theory.
        hopp_coeff (float): A scaling parameter for the hopping-like term
        output (str): The desired output format. Must be one of ['qiskit', 'qutip', 'matrix', 'spmatrix']

    Returns:
        (qiskit.aqua.Operator)
        The desired hopping_like term in the format specified by `output`
    """

    # 1. Extract the site and direction of the edge along which hopping occurs:
    start_site = lattice.site_vector(edges[0][0])
    end_site = lattice.project(lattice.site_vector(edges[-1][0]) + standard_basis(edges[-1][1], lattice.ndim))

    # 2. Generate the hopping term as a sum over the fermionic (spinor) components
    summands = []
    # 2.1 Check for boundary sites
    # Treat open and closed boundary conditions (skip term if edge is at boundary)
    # is_at_boundary = lattice.is_boundary_along(site, hopping_dim, direction='positive')
    # if is_at_boundary:
    #     # print('encountered boundary at {}, {}'.format(site, j))
    #     raise UserWarning('The given `site` and `hopping_dim` combination goes outside the lattice.')


    # 3 Sum over all spinor components:
    for alpha in range(2):
        for beta in range(2):
            # Skip cases with zero coefficients
            if mixmat[alpha, beta] == 0:
                # print('skipped ab {}, {}'.format(alpha, beta))
                continue

            # Generate the fermionic hopping terms
            bwd_hopp = (hopp_coeff
                        * psidag(start_site, alpha, lattice)
                        * mixmat[alpha, beta]
                        * psi(end_site, beta, lattice))
            fwd_hopp = bwd_hopp.dag()

            # If gauge field is present, tensor with the hopping terms with the link operators
            if S > 0:
                if output == 'qiskit':
                    bwd_string = (bwd_hopp @ link_id(S, lattice)).to_qubit_operator(output=output)
                    fwd_string = (fwd_hopp @ link_id(S, lattice)).to_qubit_operator(output=output)

                    for edge in edges:

                        bwd_string_element = (fermion_id(lattice) @ U(edge, S, lattice)).to_qubit_operator(output=output)
                        fwd_string_element = (fermion_id(lattice) @ Udag(edge, S, lattice)).to_qubit_operator(output=output)

                        bwd_string = bwd_string.multiply(bwd_string_element)
                        fwd_string = fwd_string.multiply(fwd_string_element)

                else:
                    # TODO: Implement a multiplication operator for multiplying two SpinOperators
                    raise NotImplementedError

            elif not output == 'abstract':
                bwd_string = bwd_hopp.to_qubit_operator(output=output)
                fwd_string = fwd_hopp.to_qubit_operator(output=output)

            # Convert the operators to qubit operators and add them to the total hopping term

            summands += [fwd_string, bwd_string]

    # 5. Sum up all hopping terms and return the hopping hamilton_qiskit
    string_term = operator_sum(summands)
    if output == 'qiskit':
        string_term.chop()
    return string_term






def get_list_edge_pairs(lattice):
    """
    Generates a full list of adjacent edge pairs on the lattice.

    Args:
        lattice (Lattice): The lattice object on which the lattice gauge theory is defined.

    Returns:
        (list)
        The desired list of edge pairs [pair_1, pair_2, ...] where pair_i = [edge_1, edge_2]
    """
    list_edge_pairs = []
    for site in lattice.sites:

        for dim1 in range(lattice.ndim):
            # positive directions 1

            # Check for boundary sites
            # Treat open and closed boundary conditions (skip term if edge is at boundary)
            is_at_boundary = lattice.is_boundary_along(site, dim1, direction='positive')
            if is_at_boundary:
                continue

            edge1 = np.ndarray(2, dtype='int64')
            edge1[0] = lattice.site_index(site)
            edge1[1] = dim1

            for dim2 in range(dim1 + 1, lattice.ndim):
                # positive directions 2
                # Check for boundary sites
                # Treat open and closed boundary conditions (skip term if edge is at boundary)
                is_at_boundary = lattice.is_boundary_along(site, dim2, direction='positive')
                if is_at_boundary:
                    continue

                edge2 = np.ndarray(2, dtype='int64')
                edge2[0] = lattice.site_index(site)
                edge2[1] = dim2

                list_edge_pairs.append([edge1, edge2])

            for dim2 in range(lattice.ndim):
                # negative directions 2
                # Check for boundary sites
                # Treat open and closed boundary conditions (skip term if edge is at boundary)
                is_at_boundary = lattice.is_boundary_along(site, dim2, direction='negative')
                if is_at_boundary:
                    continue

                previous_site = site - standard_basis(dim2, lattice.ndim)

                edge2 = np.ndarray(2, dtype='int64')
                edge2[0] = lattice.site_index(previous_site)
                edge2[1] = dim2

                list_edge_pairs.append([edge1, edge2])

        for dim1 in range(lattice.ndim):
            # negative directions 1

            # Check for boundary sites
            # Treat open and closed boundary conditions (skip term if edge is at boundary)
            is_at_boundary = lattice.is_boundary_along(site, dim1, direction='negative')
            if is_at_boundary:
                continue

            previous_site = site - standard_basis(dim1, lattice.ndim)

            edge1 = np.ndarray(2, dtype='int64')
            edge1[0] = lattice.site_index(previous_site)
            edge1[1] = dim1

            for dim2 in range(dim1 + 1, lattice.ndim):
                # negative directions 2
                # Check for boundary sites
                # Treat open and closed boundary conditions (skip term if edge is at boundary)
                is_at_boundary = lattice.is_boundary_along(site, dim2, direction='negative')
                if is_at_boundary:
                    continue

                previous_site = site - standard_basis(dim2, lattice.ndim)

                edge2 = np.ndarray(2, dtype='int64')
                edge2[0] = lattice.site_index(previous_site)
                edge2[1] = dim2

                list_edge_pairs.append([edge1, edge2])

    return list_edge_pairs


def multimode_normal_sampler(size, modes=(-1, 1), width=np.pi/8):
    """
    Draws samples with the specified `size` from the distribution N(mode[0], width) + ... + N(mode[-1], width),
    where N denotes the normal distribution.

    Args:
        size (int or list): The size of the randomly sampled numpy array that is returned
        modes (array-like): A list of the modes of the distribution
        width (float): The width parameter of the normal distribution

    Returns:
        numpy.ndarray
    """
    return np.random.randn(size)*width + np.random.choice(modes, size)


def uniform_sampler(size, lb=-np.pi, ub=np.pi):
    """
    Draws random samples with the specified `size` from the uniform distribution U(lb, ub).

    Args:
        size (int or list): The size of the randomly sampled numpy array that is returned
        lb (float): The lower bound
        ub (float): The upper bound

    Returns:
        numpy.ndarray
    """
    return np.random.rand(size)*(ub-lb) + lb


### similar to WilsonLGT2 but with phase matrix diag(lamba0, lambda1)

class WilsonLGT2bis(VariationalForm):
    """A variational form constructed from parametrized evolution with single-edge hopping terms
    in the kinetic part of the Hamiltonian."""
    CONFIGURATION = {
        'name': 'WilsonLGT',
        'description': 'LGT Variational Form for lattice gauge theories',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'lgt_schema',
            'type': 'object',
            'properties': {
                'depth': {
                    'type': 'integer',
                    'default': 1,
                    'minimum': 0
                },
            },
            'additionalProperties': False
        },
        'depends': [
            {
                'pluggable_type': 'initial_state',
                'default': {
                    'name': 'ZERO',
                }
            },
        ],
    }

    def __init__(self, lattice, S, rep=None, coupling_matrices=None, depth=1, initial_state=None):
        """Constructor for a hopping term based variational form for U(1) Lattice gauge theories
        with wilson fermions.

        Args:
            lattice (Lattice): The lattice on which the lattice gauge theory is fomulated
            S (float): The spin truncation value of the Quantum Link Model used to model the gauge fields.
                Must be a non-negative integer or half-integer.
            rep (dict): A dictionary specifying the representation of the Clifford algebra to be used
            depth (int): The depth of the variational circuit
            initial_state (qiskit.InitialState): The initial state object
        """
        #self.validate(locals())
        #validate_min('depth', depth, 0)
        super().__init__()

        # 1 Initialize the varform-circuit parameters
        self._num_qubits = lattice.nsites * 2 + lattice.nedges * int(np.ceil(np.log2(2 * S + 1)))
        self._depth = depth

        # 2 Initialize the var-form lattice parameters
        self._lattice = lattice
        self._S = S
        self._initial_state = initial_state

        # 3 Set the parametrized couplings. Either `rep` or `coupling_matrices` must be given as argument.
        # xor these two arguments:
        assert (rep is not None) ^ (coupling_matrices is not None), \
            'Either `rep` or `coupling_matrices` must be given.'

        # Note: The attribute self._coupling_matrices will be a 4d - numpy array of shape
        # (ndim, ncouplings_per_dim, ncomponents, ncomponents), i.e.
        # self._coupling_matrices[2][3] returns the `3`rd coupling matrix for hopping along
        # lattice dimension `2` as a (ncomponents, ncomponents) numpy array.
        #praticamente se non 
        if rep is not None:
            gamma = [val for val in rep.values()]
            self._coupling_matrices = np.array([np.array([-1j * gamma[0] @ gamma[hopping_dim + 1]])
                                           for hopping_dim in range(self._lattice.ndim)])

        elif coupling_matrices is not None:
            self._coupling_matrices = atleast_4d(coupling_matrices)

        # 4 Parse the coupling_matrices:
        # Each dimension should have a list of coupling matrices:
        if not self._coupling_matrices.shape[0] == self._lattice.ndim:
            raise UserWarning('Error with coupling matrices. The given lattice has {} dimensions but only coupling'
                              'matrices were given for {} dimensions. Please check that `coupling_matrices` has the '
                              'shape (lattice.ndim, n_couplings_per_dim, ncomponents, ncomponents)'.format(
                self._lattice.ndim, self._coupling_matrices.shape[0]))
        # And each coupling matrix should be a square matrices of size (ncomp, ncomp). Currently only 2 component
        # spinors are supported.
        if not self._coupling_matrices.shape[2:] == (2,2): # TODO: Generalize for more components spinors
            raise UserWarning('Error with coupling matrices. Your coupling matrices couple {} component spinors,'
                              'but currently only 2 component spinors are supported.'.format(
                self._coupling_matrices.shape[2]))

        # 5 Extract the number of parametrized couplings for every dimension
        self._num_couplings = self._coupling_matrices.shape[1]

        # 6 Count the total number of parameters of the variational form:
        # one parameter for every edge (along which hopp. occurs) and every parametrized coupling plus
        # one parameter for final z-rotation at the end of each layer
 ##### occhio faccio per due per mettere due lambda       
        self._num_parameters = (lattice.nedges * self._num_couplings             # hopping terms
                                + lattice.nsites * 2) * depth          # + spinor component z-rotations

        # 7 Set bounds for the parameters
        hopping_bounds = [(-np.pi * self.hopper_correction,
                                     np.pi * self.hopper_correction)] * self.num_hoppings_per_layer
        phase_bounds = [(-np.pi, np.pi)] * self.num_phases_per_layer
        self._bounds = operator_sum([hopping_bounds + phase_bounds] * self._depth)

        # 8 Set up the hopping terms along the edges
        self._construct_hopping_operators()
        self._construct_phase_operators()
        self._matrix_hopping_terms = []

    @property
    def num_hoppings_per_layer(self):
        return self._lattice.nedges

# per ogni sito (rappresentato da due qubit) metto una phase per canale, quindi due per sito 
    @property
    def num_phases_per_layer(self):
        return self._lattice.nsites * 2 

    @property
    def num_hopping_parameters(self):
        return self.num_hoppings_per_layer * self._num_couplings * self._depth


    @property
    def num_phase_parameters(self):
        return self.num_phases_per_layer * self._depth

    @property
    def hopper_correction(self) -> float:
        """ The correction for the norms of the hopping terms to make sure that all hopping strengths can be
        reached within the boundaries of (-np.pi, np.pi)"""
        if self._S == 0:
            return 1.
        else:
            return np.sqrt((self._S + 1) * self._S) * 2

    def _construct_hopping_operators(self, mode='qiskit'):
        """
        Constructs the Hamiltonian inspired hopping operators which are exponentiated when building the circuit and
        stores them in `self._hopping_terms` (for 'qiskit' mode) or `self._matrix_hopping_terms` (for 'qutip' mode).

        Args:
            mode (str): Must be one of ['qiskit', 'qutip']

        Returns:
            None
        """
        if mode == 'qiskit':
            self._hopping_terms = [] # will be a list of shape (nedges * ncouplings)
        elif mode == 'qutip':
            self._matrix_hopping_terms = [] # will be a list of shape (nedges * ncouplings)
        else:
            raise UserWarning("mode must be one of ['qiskit', 'qutip']")

        # 1. Iterate through all lattice edges
        for edge in self._lattice.edges:
            hopping_dim = edge[1]

            # 2. Iterate through all coupling matrices (mixmatrices) along the given dimension of the edge
            for mixmatrix in self._coupling_matrices[hopping_dim]:
                # Build up the hopping term along this edge with the specified coupling
                hopper_edge = hopping_like(edge, lattice=self._lattice, S=self._S, mixmat=mixmatrix, output=mode)
                # Simplify the operator if in qiskit mode.
                if mode == 'qiskit':
                    hopper_edge.chop()
                    self._hopping_terms.append(hopper_edge)
                elif mode == 'qutip':
                    self._matrix_hopping_terms.append(hopper_edge)


    def _construct_phase_operators(self):
        self._phase_terms = []
        for site in self._lattice.sites:

            phase_first_comp= phase_like(site, lattice=self._lattice, S=self._S, copmat=np.array([[1,0],[0,0]]) , output='qiskit')

            phase_first_comp.chop()
            
            phase_second_comp= phase_like(site, lattice=self._lattice, S=self._S, copmat=np.array([[0,0],[0,1]]) , output='qiskit')

            phase_second_comp.chop()

            self._phase_terms.append(phase_first_comp)
            self._phase_terms.append(phase_second_comp)
             



    def construct_circuit(self, parameters, q=None):
        """
        Construct the variational form, given its parameters.
        Args:
            parameters (numpy.ndarray): circuit parameters.
            coupling_matrices (list): list of np.ndarray for the coupling matrices
            q (QuantumRegister): Quantum Register for the circuit.
        Returns:
            QuantumCircuit: a quantum circuit with given `parameters`
        Raises:
            ValueError: the number of parameters is incorrect.
        """

        # 1 Parse inputs
        if len(parameters) != self._num_parameters:
            raise ValueError('The number of parameters has to be {}'.format(self._num_parameters))

        # 2 Set up the quantum register and the circuit
        if q is None:
            if self._initial_state is not None:
                circuit = self._initial_state.construct_circuit('circuit')
                q = circuit.qregs
            else:
                q = QuantumRegister(self._num_qubits, name='q')
                circuit = QuantumCircuit(q)
        else:
            circuit = self._initial_state.construct_circuit('circuit', q)
        all_qubits = circuit.qubits
        fermion_qubits = circuit.qubits[:self._lattice.nsites*2]

        # 3 Build up the variational form with the parameters
        
        hoppings_per_layer = self.num_hoppings_per_layer
        couplings_per_hopping = self._num_couplings
        phases_per_layer = self.num_phases_per_layer
        params_per_layer = couplings_per_hopping * hoppings_per_layer + phases_per_layer

        # 4. Iterate over depth to construct the individual blocks
        for d in range(self._depth):
            # 4.1 For each depth-block add parametrized exponentials of hopping terms
            hopping_params = parameters[d * params_per_layer: (d+1) * params_per_layer - phases_per_layer]
            phase_params = parameters[(d+1) * params_per_layer - phases_per_layer: (d+1) * params_per_layer]

            for i, hopper, parameter in zip(np.arange(hoppings_per_layer),
                                            self._hopping_terms,
                                            hopping_params):

                # print(hopper)
                var_term = construct_trotter_step(hopper,
                                                  parameter,
                                                  name='h_{}({})'.format(self._lattice.edges[i//couplings_per_hopping],
                                                                         parameter))
                # Se con piu di 2 site non funziona controlla init_state ceh sia adattato                                                       
                circuit.append(var_term, qargs=all_qubits)

            # 4.3 Add z-rotations for every second qubit at the end of one layer
            for i, phase_term in zip( np.arange(len(self._phase_terms)), self._phase_terms):

                var_term = construct_trotter_step(phase_term, phase_params[i], name='phase_{}({})'.format(i,phase_params[i]))
                circuit.append(var_term, qargs=all_qubits)

            
                

        return circuit



################################################################################################
# 2. Set up the variational form that is based on the lattice hopping terms to conserve
#    the Gauss law (gauge-invariance)
################################################################################################

class WilsonLGT2(VariationalForm):
    """A variational form constructed from parametrized evolution with single-edge hopping terms
    in the kinetic part of the Hamiltonian."""
    CONFIGURATION = {
        'name': 'WilsonLGT',
        'description': 'LGT Variational Form for lattice gauge theories',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'lgt_schema',
            'type': 'object',
            'properties': {
                'depth': {
                    'type': 'integer',
                    'default': 1,
                    'minimum': 0
                },
            },
            'additionalProperties': False
        },
        'depends': [
            {
                'pluggable_type': 'initial_state',
                'default': {
                    'name': 'ZERO',
                }
            },
        ],
    }

    def __init__(self, lattice, S, rep=None, coupling_matrices=None, depth=1, initial_state=None):
        """Constructor for a hopping term based variational form for U(1) Lattice gauge theories
        with wilson fermions.

        Args:
            lattice (Lattice): The lattice on which the lattice gauge theory is fomulated
            S (float): The spin truncation value of the Quantum Link Model used to model the gauge fields.
                Must be a non-negative integer or half-integer.
            rep (dict): A dictionary specifying the representation of the Clifford algebra to be used
            depth (int): The depth of the variational circuit
            initial_state (qiskit.InitialState): The initial state object
        """
        #self.validate(locals())
        #validate_min('depth', depth, 0)
        super().__init__()

        # 1 Initialize the varform-circuit parameters
        self._num_qubits = lattice.nsites * 2 + lattice.nedges * int(np.ceil(np.log2(2 * S + 1)))
        self._depth = depth

        # 2 Initialize the var-form lattice parameters
        self._lattice = lattice
        self._S = S
        self._initial_state = initial_state

        # 3 Set the parametrized couplings. Either `rep` or `coupling_matrices` must be given as argument.
        # xor these two arguments:
        assert (rep is not None) ^ (coupling_matrices is not None), \
            'Either `rep` or `coupling_matrices` must be given.'

        # Note: The attribute self._coupling_matrices will be a 4d - numpy array of shape
        # (ndim, ncouplings_per_dim, ncomponents, ncomponents), i.e.
        # self._coupling_matrices[2][3] returns the `3`rd coupling matrix for hopping along
        # lattice dimension `2` as a (ncomponents, ncomponents) numpy array.
        #praticamente se non 
        if rep is not None:
            gamma = [val for val in rep.values()]
            self._coupling_matrices = np.array([np.array([-1j * gamma[0] @ gamma[hopping_dim + 1]])
                                           for hopping_dim in range(self._lattice.ndim)])

        elif coupling_matrices is not None:
            self._coupling_matrices = atleast_4d(coupling_matrices)

        # 4 Parse the coupling_matrices:
        # Each dimension should have a list of coupling matrices:
        if not self._coupling_matrices.shape[0] == self._lattice.ndim:
            raise UserWarning('Error with coupling matrices. The given lattice has {} dimensions but only coupling'
                              'matrices were given for {} dimensions. Please check that `coupling_matrices` has the '
                              'shape (lattice.ndim, n_couplings_per_dim, ncomponents, ncomponents)'.format(
                self._lattice.ndim, self._coupling_matrices.shape[0]))
        # And each coupling matrix should be a square matrices of size (ncomp, ncomp). Currently only 2 component
        # spinors are supported.
        if not self._coupling_matrices.shape[2:] == (2,2): # TODO: Generalize for more components spinors
            raise UserWarning('Error with coupling matrices. Your coupling matrices couple {} component spinors,'
                              'but currently only 2 component spinors are supported.'.format(
                self._coupling_matrices.shape[2]))

        # 5 Extract the number of parametrized couplings for every dimension
        self._num_couplings = self._coupling_matrices.shape[1]
        ###################### Attenzione ################
        ## num couplings é il numero di thetass 
        #non farlo perhcé aumentano anche i parametri che almeno quelli sono giusti
        #self._num_couplings = self._coupling_matrices.shape[1] * lattice.nedges
        ###################### Attenzione######

        # 6 Count the total number of parameters of the variational form:
        # one parameter for every edge (along which hopp. occurs) and every parametrized coupling plus
        # one parameter for final z-rotation at the end of each layer
        self._num_parameters = (lattice.nedges * self._num_couplings             # hopping terms
                                + lattice.nsites) * depth          # + spinor component z-rotations

        # 7 Set bounds for the parameters
        hopping_bounds = [(-np.pi * self.hopper_correction,
                                     np.pi * self.hopper_correction)] * self.num_hoppings_per_layer
        phase_bounds = [(-np.pi, np.pi)] * self.num_phases_per_layer
        self._bounds = operator_sum([hopping_bounds + phase_bounds] * self._depth)

        # 8 Set up the hopping terms along the edges
        self._construct_hopping_operators()
        self._matrix_hopping_terms = []

###### Atttenzione ########  sotto ne contava solo uno  
    @property
    def num_hoppings_per_layer(self):
        #return self._lattice.nedges
        #assumo che per ogni coupling matrice c'é un coupling matrice
        return self._num_couplings
###### Attenzione ##########
    @property
    def num_phases_per_layer(self):
        return self._lattice.nsites

    @property
    def num_hopping_parameters(self):
        ################Atttenzione ############ Vengono contati doppi visto che ho modificato prima
        #return self.num_hoppings_per_layer * self._num_couplings * self._depth
        return self.num_hoppings_per_layer * self._depth
        ################Atttenzione ############
    @property
    def num_phase_parameters(self):
        return self.num_phases_per_layer * self._depth

    @property
    def hopper_correction(self) -> float:
        """ The correction for the norms of the hopping terms to make sure that all hopping strengths can be
        reached within the boundaries of (-np.pi, np.pi)"""
        if self._S == 0:
            return 1.
        else:
            return np.sqrt((self._S + 1) * self._S) * 2

    def _construct_hopping_operators(self, mode='qiskit'):
        """
        Constructs the Hamiltonian inspired hopping operators which are exponentiated when building the circuit and
        stores them in `self._hopping_terms` (for 'qiskit' mode) or `self._matrix_hopping_terms` (for 'qutip' mode).

        Args:
            mode (str): Must be one of ['qiskit', 'qutip']

        Returns:
            None
        """
        if mode == 'qiskit':
            self._hopping_terms = [] # will be a list of shape (nedges * ncouplings)
        elif mode == 'qutip':
            self._matrix_hopping_terms = [] # will be a list of shape (nedges * ncouplings)
        else:
            raise UserWarning("mode must be one of ['qiskit', 'qutip']")

        # 1. Iterate through all lattice edges
        for edge in self._lattice.edges:
            hopping_dim = edge[1]

            # 2. Iterate through all coupling matrices (mixmatrices) along the given dimension of the edge
            for mixmatrix in self._coupling_matrices[hopping_dim]:
                # Build up the hopping term along this edge with the specified coupling
                hopper_edge = hopping_like(edge, lattice=self._lattice, S=self._S, mixmat=mixmatrix, output=mode)
                # Simplify the operator if in qiskit mode.
                if mode == 'qiskit':
                    hopper_edge.chop()
                    self._hopping_terms.append(hopper_edge)
                elif mode == 'qutip':
                    self._matrix_hopping_terms.append(hopper_edge)

    def _construct_parametrized_evolution_circuits(self):
        # TODO
        pass

    def initialize_parameters(self, mode='uniform', hopping_sampler=None, phase_sampler=None):
        """
        Retruns a numpy array of randomly initialized parameters for the given variational form. The parameters
        are drawn from the specified `distribution`
        Args:
            mode (str): The initialization mode. Must be one of ['uniform', 'multimode', 'normal', 'multimode-normal',
                'custom']. If custom mode is used, distribution functions for the `hopping_sampler` and the
                `phase_sampler` must be given.
            hopping_sampler (function): The distribution function for the hopping parameters.
            phase_sampler (function): The distribution function for the single qubit z-rotation parameters at the
                end of a block

        Returns:
            np.ndarray:
                A random numpy array distributed according to the given distributions of size `self.num_parameters`
        """

        if mode == 'uniform':
            return uniform_sampler(self.num_parameters,
                                   lb=-np.pi/(2*self.hopper_correction),
                                   ub=np.pi/(2*self.hopper_correction))

        elif mode == 'multimode':
            return multimode_normal_sampler(self.num_parameters,
                                            modes=[-1,1],
                                            width=np.pi/(8*self.hopper_correction))

        elif mode == 'normal':
            return multimode_normal_sampler(self.num_parameters,
                                            modes=(0),
                                            width=1/self.hopper_correction)

        elif mode =='multimode-uniform':
            hopping_sampler = lambda size: multimode_normal_sampler(size,
                                                                    modes=[-1,1],
                                                                    width=np.pi/(8*self.hopper_correction))
            phase_sampler = lambda size: uniform_sampler(size,
                                                         lb=-np.pi / (2 * self.hopper_correction),
                                                         ub=np.pi / (2 * self.hopper_correction))

        elif mode == 'custom':
            if hopping_sampler is None or phase_sampler is None:
                raise UserWarning("Must provide functions for `hopping_sampler` and `phase_sampler` in 'custom' mode.")

        else:
            raise UserWarning("`mode` must be one of ['uniform', 'normal', 'multimode',"
                              " 'multimode-uniform', 'custom']")

        # Generate the random parameters for the first depth-block
        hoppings1 = hopping_sampler(self.num_hoppings_per_layer * self._num_couplings)
        phases1 = phase_sampler(self.num_phases_per_layer)
        params= np.hstack((hoppings1, phases1))

        # Generate the random parameters for the following blocks.
        for d in range(self._depth-1):
            hoppingsd = hopping_sampler(self.num_hoppings_per_layer * self._num_couplings)
            phasesd = phase_sampler(self.num_phases_per_layer)
            params = np.hstack((params, hoppingsd, phasesd))
        return params

    def construct_circuit(self, parameters, q=None, num_time_slices=1):
        """
        Construct the variational form, given its parameters.
        Args:
            parameters (numpy.ndarray): circuit parameters.
            coupling_matrices (list): list of np.ndarray for the coupling matrices
            q (QuantumRegister): Quantum Register for the circuit.
        Returns:
            QuantumCircuit: a quantum circuit with given `parameters`
        Raises:
            ValueError: the number of parameters is incorrect.
        """

        # 1 Parse inputs
        if len(parameters) != self._num_parameters:
            raise ValueError('The number of parameters has to be {}'.format(self._num_parameters))

        # 2 Set up the quantum register and the circuit
        if q is None:
            if self._initial_state is not None:
                circuit = self._initial_state.construct_circuit('circuit')
                q = circuit.qregs
            else:
                q = QuantumRegister(self._num_qubits, name='q')
                circuit = QuantumCircuit(q)
        else:
            circuit = self._initial_state.construct_circuit('circuit', q)
        all_qubits = circuit.qubits
        fermion_qubits = circuit.qubits[:self._lattice.nsites*2]

        # 3 Build up the variational form with the parameters
        #hoppings_per_layer = self.num_hoppings_per_layer
        # attenzione metto per lattice, perché lui mette 4 invece di 8  
        hoppings_per_layer = self.num_hoppings_per_layer * self._lattice.nedges
        couplings_per_hopping = self._num_couplings
        phases_per_layer = self.num_phases_per_layer
        #### per 2 sites con 2 couplingmatrices, ci sono 4 param 
        #per layer cioé per edges 
        #params_per_layer = couplings_per_hopping * hoppings_per_layer + phases_per_layer
        params_per_layer =  hoppings_per_layer + phases_per_layer

        # 4. Iterate over depth to construct the individual blocks
        for d in range(self._depth):
            # 4.1 For each depth-block add parametrized exponentials of hopping terms
            hopping_params = parameters[d * params_per_layer: (d+1) * params_per_layer - phases_per_layer]
            phase_params = parameters[(d+1) * params_per_layer - phases_per_layer: (d+1) * params_per_layer]

            for i, hopper, parameter in zip(np.arange(hoppings_per_layer),
                                            self._hopping_terms,
                                            hopping_params):
                #print(i, "hopp", hopper, parameter)
                var_term = construct_trotter_step(hopper,
                                                  parameter,
                                                  name='h_{}({})'.format(self._lattice.edges[i//couplings_per_hopping],
                                                                         parameter), num_time_slices= num_time_slices)
                # se come init stae metto Dirac qui  if len(qargs) != self.num_qubits, mi da errore  
                # check initial state of varform!!                                                         
                circuit.append(var_term, qargs=all_qubits)

            # 4.3 Add z-rotations for every second qubit at the end of one layer
            for qbit, phase_param in zip(fermion_qubits[1::2], phase_params):
                # circuit.u1(phase_param, qbit)
                circuit.rz(phase_param, qbit)

        return circuit

    @staticmethod
    def _site_rotation_matrix(site_index, lattice, S, comp=1):
        """
        Perform a z-rotation on the `comp`-th qubit making up a site. This method is meant to be used to simulate
        and verify the qiskit circuit, not for any other purposes.

        Args:
            site_index (int): The index of the site for which we want to rotate one of the components
            lattice (Lattice): The lattice object
            S (float): The spin truncation of the Abelian Quantum Link Model
            comp (int): The component qubit which we want to rotate

        Returns:
            qutip.Qobj: The matrix representation of the pauli string II...IZI...II, where the Z pauli operator is
                acting on the qubit representing the spinor component `comp` at the lattice site indexed by `site_index`
        """
    

        #dim_S = int(2 * S + 1)
        dim_S = int(np.ceil(np.log2(2 * S + 1)))
        ncomp = 2
        assert 0 <= site_index <= lattice.nsites - 1, '`site_index` out of bounds ' \
                                                      'for lattice with {} sites'.format(lattice.nsites)

        ops = [qt.identity(2)] * (ncomp * site_index + comp) \
              + [qt.sigmaz()] \
              + [qt.identity(2)] * (ncomp * lattice.nsites - (ncomp * site_index + comp) - 1)

        

        #qui c'é un errore (oppure no se é pensato in linear, cmq se ho dim_2=2 e faccio qt.identity(2) ho una matrice che prende
        # solo due elementi ma dovrei fare 2**dim_S, perché per ogni qubit ho due var 
        #ops += [qt.identity(dim_S)] * lattice.nedges
        if S > 0:
            ops += [qt.identity(2**dim_S)] * lattice.nedges
        
        return qt.tensor(ops[::-1])

    @staticmethod
    def _edge_rotation_matrix(edge_index, lattice, S):
        """
        Perform a z-rotation on the `edge_index`-th edge. This method is meant to be used to simulate
        and verify the qiskit circuit, not for any other purposes.

        Args:
            edge_index (int): The index of the edge for which we want to perform a z-rotation
            lattice (Lattice): The lattice object
            S (float): The spin truncation of the Abelian Quantum Link Model

        Returns:
            qutip.Qobj:
                The matrix representation of the z-rotation on the `edge_index`-th edge.
        """

        dim_S = int(2*S+1)
        ncomp = 2
        assert 0 <= edge_index <= lattice.nedges - 1, '`edge_index` out of bounds ' \
                                                      'for lattice with {} edges'.format(lattice.nedges)

        ops = [qt.identity(2)] * (ncomp * lattice.nsites) \
              + [qt.identity(dim_S)] * edge_index \
              + [qt.jmat(S, 'z')]  \
              + [qt.identity(dim_S)] * (lattice.nedges - edge_index - 1)

        return qt.tensor(ops[::-1])

    def _eval_matrix_varform(self, parameters, init_state):
        """
        Constructs the variational state in qutip. Used to verify qiskit simulations.

        Args:
            parameters (list or np.ndarray):
            init_state (qutip.Qobj):

        Returns:

        
        """
        if len(parameters) != self._num_parameters:
            raise ValueError('The number of parameters has to be {}'.format(self._num_parameters))

        # 1. If hopping operators are not yet built, build them
        if self._matrix_hopping_terms == []:
            self._construct_hopping_operators(mode='qutip')

        # 2 Initialize the var_form matrix
        var_form = init_state.copy()
        mat_form = qt.identity(2**self.num_qubits)
        # 3. Build up the variational form with the parameters
        #hoppings_per_layer = len(self._matrix_hopping_terms)
        #couplings_per_hopping = self._coupling_matrices.shape[1]
        ####### occhio
        #phases_per_layer = self.num_qubits
        ####occhio che dopo esce [0:0]
        #params_per_layer = couplings_per_hopping * hoppings_per_layer + phases_per_layer
        hoppings_per_layer = self.num_hoppings_per_layer * self._lattice.nedges
        couplings_per_hopping = self._num_couplings
        phases_per_layer = self.num_phases_per_layer
        params_per_layer =  hoppings_per_layer + phases_per_layer
        # 3.1 Iterate over depth
        for d in range(self._depth):
            # 3.2 For each depth-block add parametrized exponentials of hopping terms
            hopping_params = parameters[d * params_per_layer: (d + 1) * params_per_layer - phases_per_layer]
            phase_params = parameters[(d + 1) * params_per_layer - phases_per_layer: (d + 1) * params_per_layer]
            ##### cambio len(hopping_param) con len(self._matrix_hopping_terms)
            for i, hopper, parameter in zip(np.arange(len(self._matrix_hopping_terms)),
                                            self._matrix_hopping_terms,
                                            hopping_params):
                #trasform hopper in the right dims:  dims [16,1,1,1,,2,2,] --> [1024,1024]
                hopper = qt.Qobj(hopper.full())
                var_form = (1j * parameter * hopper).expm() \
                             * var_form           
                mat_form *= (1j * parameter * hopper).expm()

            # 3.3 At the end add single qubit rotations for each site and edge
            for site_index, param in zip(np.arange(self._lattice.nsites), phase_params[:self._lattice.nsites]):
                # perform z-rotations on every 2nd qubit
                rot_matrix_right_dim = qt.Qobj(self._site_rotation_matrix(site_index, self._lattice, self._S).full())
                #occhio metto quel -(param/2) perché cosi é uguale al aprametro usato da rz(param)
                var_form = (1j * -(param/2) * rot_matrix_right_dim).expm() \
                            * var_form
                mat_form *= (1j * -(param/2) * rot_matrix_right_dim).expm()

            # if self._S > 0:
            #     for edge_index, param in zip(np.arange(self._lattice.nedges), phase_params[self._lattice.nsites:]):
            #         # perform z-rotations on every 2nd qubit
            #         var_form = (1j * param * self._edge_rotation_matrix(edge_index, self._lattice, self._S)).expm() \
            #                     * var_form

        return var_form, mat_form


class WilsonHardwareEfficientLGT(VariationalForm):
    """A hardware efficient variational form constructed from parametrized evolution with single-edge hopping terms
    in the kinetic part of the Hamiltonian. Currently, the coupling matrix is given by
    coupling_matrices = np.array([[0,0],[1,0]])
    """

    CONFIGURATION = {
        'name': 'WilsonLGT',
        'description': 'LGT Variational Form for lattice gauge theories',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'lgt_schema',
            'type': 'object',
            'properties': {
                'depth': {
                    'type': 'integer',
                    'default': 1,
                    'minimum': 0
                },
            },
            'additionalProperties': False
        },
        'depends': [
            {
                'pluggable_type': 'initial_state',
                'default': {
                    'name': 'ZERO',
                }
            },
        ],
    }

    def __init__(self, lattice, S, depth=1, initial_state=None):
        """Constructor for a hopping term based variational form for U(1) Lattice gauge theories
        with wilson fermions.

        Args:
            lattice (Lattice): The lattice on which the lattice gauge theory is fomulated
            S (float): The spin truncation value of the Quantum Link Model used to model the gauge fields.
                Must be a non-negative integer or half-integer.
            depth (int): The depth of the variational circuit
            initial_state (qiskit.InitialState): The initial state object
        """
        super().__init__()

        # 1 Initialize the varform-circuit parameters
        self._num_qubits = lattice.nsites * 2 + lattice.nedges * int(np.ceil(np.log2(2 * S + 1))) - 2
        self._depth = depth

        # 2 Initialize the var-form lattice parameters
        self._lattice = lattice
        self._S = S
        self._initial_state = initial_state

        # 3. count the number of parameters assuming coupling_matrices = np.array([[0,0],[1,0]])
        self._num_parameters = (lattice.nedges + lattice.nsites - 1) * depth

        # 4. Set bounds for the parameters
        hopping_bounds = [(-np.pi * self.hopper_correction,
                           np.pi * self.hopper_correction)] * self.num_hoppings_per_layer
        phase_bounds = [(-np.pi, np.pi)] * self.num_phases_per_layer
        self._bounds = operator_sum([hopping_bounds + phase_bounds] * self._depth)

        # 5. Set up the hopping terms along the edges
        self._construct_hopping_operators()

    @property
    def hopper_correction(self) -> float:
        """ The correction for the norms of the hopping terms to make sure that all hopping strengths can be
        reached within the boundaries of (-np.pi, np.pi)"""
        if self._S == 0:
            return 1.
        else:
            return np.sqrt((self._S + 1) * self._S) * 2

    @property
    def num_hoppings_per_layer(self):
        return self._lattice.nedges

    @property
    def num_phases_per_layer(self):
        # assuming coupling_matrices = np.array([[0, 0], [1, 0]]),
        # and factoring out the phase rotation of the last fermionic qubit
        return self._lattice.nsites - 1

    @property
    def num_parameters(self):
        return self._num_parameters

    def _construct_hopping_operators(self, mode='qiskit'):
        """
        Constructs the Hamiltonian inspired hopping operators which are exponentiated when building the circuit and
        stores them in `self._hopping_terms` (for 'qiskit' mode).

        Args:
            mode (str): Must be one of ['qiskit']

        Returns:
            None
        """
        if mode == 'qiskit':
            self._hopping_terms = []  # will be a list of shape (nedges * ncouplings)
        else:
            raise UserWarning("mode must be one of ['qiskit']")

        # 1. Iterate through all lattice edges
        for edge in self._lattice.edges:

            # 2. Iterate through all coupling matrices (mixmatrices) along the given dimension of the edge
            mixmatrix = np.array([[0, 0], [1, 0]])
            # Build up the hopping term along this edge with the specified coupling
            hopper_edge = hopping_like(edge, lattice=self._lattice, S=self._S, mixmat=mixmatrix, output=mode)
            new_paulis = hopper_edge.paulis
            for pauli in new_paulis:
                pauli[1].delete_qubits([0, self._lattice.nsites * 2 - 1])
            hopper_edge = WeightedPauliOperator(new_paulis)
            # Simplify the operator if in qiskit mode.
            if mode == 'qiskit':
                hopper_edge.chop()
                self._hopping_terms.append(hopper_edge)

    def construct_circuit(self, parameters, q=None, num_time_slices=1):
        """
        Construct the variational form, given its parameters.
        Args:
            parameters (numpy.ndarray): circuit parameters.
            q (QuantumRegister): Quantum Register for the circuit.
        Returns:
            QuantumCircuit: a quantum circuit with given `parameters`
        Raises:
            ValueError: the number of parameters is incorrect.
        """

        # 1 Parse inputs
        if len(parameters) != self._num_parameters:
            raise ValueError('The number of parameters has to be {}'.format(self._num_parameters))

        # 2 Get the original WilsonLGT2 varform and its circuit
        #         original_varform = WilsonLGT2(self._lattice, self._S, coupling_matrices=np.array([[0,0],[1,0]]), \
        #                                       initial_state=self._initial_state)

        #         original_params = copy.deepcopy(parameters)
        #         np.append(original_params, 0)
        #         original_varform_circ = original_varform.construct_circuit(original_params)

        # 2 Set up the quantum register and the circuit
        if q is None:
            if self._initial_state is not None:
                circuit = self._initial_state.construct_circuit('circuit')
                q = circuit.qregs
            else:
                q = QuantumRegister(self._num_qubits, name='q')
                circuit = QuantumCircuit(q)
        else:
            circuit = self._initial_state.construct_circuit('circuit', q)
        all_qubits = circuit.qubits
        fermion_qubits = circuit.qubits[
                         :(self._lattice.nsites * 2 - 2)]  # HardwareEfficient: excludes first and last qubit

        # 3 Build up the variational form with the parameters
        hoppings_per_layer = self.num_hoppings_per_layer
        couplings_per_hopping = 1
        phases_per_layer = self.num_phases_per_layer
        params_per_layer = couplings_per_hopping * hoppings_per_layer + phases_per_layer

        # 4. Iterate over depth to construct the individual blocks
        for d in range(self._depth):
            # 4.1 For each depth-block add parametrized exponentials of hopping terms
            hopping_params = parameters[d * params_per_layer: (d + 1) * params_per_layer - phases_per_layer]
            phase_params = parameters[(d + 1) * params_per_layer - phases_per_layer: (d + 1) * params_per_layer]

            for i, hopper, parameter in zip(np.arange(hoppings_per_layer),
                                            self._hopping_terms,
                                            hopping_params):
                # print(hopper)
                var_term = construct_trotter_step(hopper,
                                                  parameter,
                                                  name='h_{}({})'.format(
                                                      self._lattice.edges[i // couplings_per_hopping],
                                                      parameter), num_time_slices=num_time_slices)
                circuit.append(var_term, qargs=all_qubits)

            # 4.3 Add z-rotations for every second qubit at the end of one layer
            for qbit, phase_param in zip(fermion_qubits[::2], phase_params):
                circuit.u1(phase_param, qbit)

        return circuit


class WilsonLGT3(VariationalForm):
    """A variational form constructed from parametrized evolution with single-edge hopping terms
    in the kinetic part of the Hamiltonian."""
    CONFIGURATION = {
        'name': 'WilsonLGT',
        'description': 'LGT Variational Form for lattice gauge theories',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'lgt_schema',
            'type': 'object',
            'properties': {
                'depth': {
                    'type': 'integer',
                    'default': 1,
                    'minimum': 0
                },
            },
            'additionalProperties': False
        },
        'depends': [
            {
                'pluggable_type': 'initial_state',
                'default': {
                    'name': 'ZERO',
                }
            },
        ],
    }

    def __init__(self, lattice, S, rep=None, coupling_matrices=None, depth=1, initial_state=None):
        """Constructor for a hopping term based variational form for U(1) Lattice gauge theories
        with wilson fermions.

        Args:
            lattice (Lattice): The lattice on which the lattice gauge theory is fomulated
            S (float): The spin truncation value of the Quantum Link Model used to model the gauge fields.
                Must be a non-negative integer or half-integer.
            rep (dict): A dictionary specifying the representation of the Clifford algebra to be used
            depth (int): The depth of the variational circuit
            initial_state (qiskit.InitialState): The initial state object
        """
        # self.validate(locals())
        # validate_min('depth', depth, 0)
        super().__init__()

        # 1 Initialize the varform-circuit parameters
        self._num_qubits = lattice.nsites * 2 + lattice.nedges * int(np.ceil(np.log2(2 * S + 1)))
        self._depth = depth

        # 2 Initialize the var-form lattice parameters
        self._lattice = lattice
        self._S = S
        self._initial_state = initial_state

        # 3 Set the parametrized couplings. Either `rep` or `coupling_matrices` must be given as argument.
        # xor these two arguments:
        assert (rep is not None) ^ (coupling_matrices is not None), \
            'Either `rep` or `coupling_matrices` must be given.'

        # Note: The attribute self._coupling_matrices will be a 4d - numpy array of shape
        # (ndim, ncouplings_per_dim, ncomponents, ncomponents), i.e.
        # self._coupling_matrices[2][3] returns the `3`rd coupling matrix for hopping along
        # lattice dimension `2` as a (ncomponents, ncomponents) numpy array.
        if rep is not None:
            gamma = [val for val in rep.values()]
            self._coupling_matrices = np.array([np.array([-1j * gamma[0] @ gamma[hopping_dim + 1]])
                                                for hopping_dim in range(self._lattice.ndim)])

        elif coupling_matrices is not None:
            self._coupling_matrices = atleast_4d(coupling_matrices)

        # 4 Parse the coupling_matrices:
        # Each dimension should have a list of coupling matrices:
        if not self._coupling_matrices.shape[0] == self._lattice.ndim:
            raise UserWarning('Error with coupling matrices. The given lattice has {} dimensions but only coupling'
                              'matrices were given for {} dimensions. Please check that `coupling_matrices` has the '
                              'shape (lattice.ndim, n_couplings_per_dim, ncomponents, ncomponents)'.format(
                self._lattice.ndim, self._coupling_matrices.shape[0]))
        # And each coupling matrix should be a square matrices of size (ncomp, ncomp). Currently only 2 component
        # spinors are supported.
        if not self._coupling_matrices.shape[2:] == (2, 2):  # TODO: Generalize for more components spinors
            raise UserWarning('Error with coupling matrices. Your coupling matrices couple {} component spinors,'
                              'but currently only 2 component spinors are supported.'.format(
                self._coupling_matrices.shape[2]))

        # 5 Extract the number of parametrized couplings for every dimension
        self._num_couplings = self._coupling_matrices.shape[1]

        # 6 Count the total number of parameters of the variational form:
        # one parameter for every edge (along which hopp. occurs) and every parametrized coupling plus
        # one parameter for final z-rotation at the end of each layer
        self._num_parameters = (lattice.nedges * lattice.nedges**2 * self._num_couplings  # hopping terms
                                + lattice.nsites) * depth  # + spinor component z-rotations

        # 7 Set bounds for the parameters
        hopping_bounds = [(-np.pi * self.hopper_correction,
                           np.pi * self.hopper_correction)] * self.num_hoppings_per_layer \
                         * self.num_hoppings_per_layer**2

        phase_bounds = [(-np.pi, np.pi)] * self.num_phases_per_layer
        self._bounds = operator_sum([hopping_bounds + phase_bounds] * self._depth)

        # 8 Set up the hopping terms along the edges
        self._construct_hopping_operators()
        self._matrix_hopping_terms = []

    @property
    def num_hoppings_per_layer(self):
        return self._lattice.nedges

    @property
    def num_phases_per_layer(self):
        return self._lattice.nsites

    @property
    def num_hopping_parameters(self):
        return self.num_hoppings_per_layer * self._num_couplings * self._depth

    @property
    def num_phase_parameters(self):
        return self.num_phases_per_layer * self._depth

    @property
    def hopper_correction(self) -> float:
        """ The correction for the norms of the hopping terms to make sure that all hopping strengths can be
        reached within the boundaries of (-np.pi, np.pi)"""
        if self._S == 0:
            return 1.
        else:
            return np.sqrt((self._S + 1) * self._S) * 2

    def _construct_hopping_operators(self, mode='qiskit'):
        """
        Constructs the Hamiltonian inspired hopping operators which are exponentiated when building the circuit and
        stores them in `self._hopping_terms` (for 'qiskit' mode) or `self._matrix_hopping_terms` (for 'qutip' mode).

        Args:
            mode (str): Must be one of ['qiskit', 'qutip']

        Returns:
            None
        """
        if mode == 'qiskit':
            self._hopping_terms = []  # will be a list of shape (nedges * ncouplings)
        elif mode == 'qutip':
            self._matrix_hopping_terms = []  # will be a list of shape (nedges * ncouplings)
        else:
            raise UserWarning("mode must be one of ['qiskit', 'qutip']")

        # 1. Iterate through all lattice edges
        for edge in self._lattice.edges:
            hopping_dim = edge[1]

            # 2. Iterate through all coupling matrices (mixmatrices) along the given dimension of the edge
            for mixmatrix in self._coupling_matrices[hopping_dim]:
                # Build up the hopping term along this edge with the specified coupling
                hopper_edge = hopping_like(edge, lattice=self._lattice, S=self._S, mixmat=mixmatrix, output=mode)
                # Simplify the operator if in qiskit mode.
                if mode == 'qiskit':
                    hopper_edge.chop()
                    self._hopping_terms.append(hopper_edge)
                elif mode == 'qutip':
                    self._matrix_hopping_terms.append(hopper_edge)

    def _construct_parametrized_evolution_circuits(self):
        # TODO
        pass

    def initialize_parameters(self, mode='uniform', hopping_sampler=None, phase_sampler=None):
        """
        Retruns a numpy array of randomly initialized parameters for the given variational form. The parameters
        are drawn from the specified `distribution`
        Args:
            mode (str): The initialization mode. Must be one of ['uniform', 'multimode', 'normal', 'multimode-normal',
                'custom']. If custom mode is used, distribution functions for the `hopping_sampler` and the
                `phase_sampler` must be given.
            hopping_sampler (function): The distribution function for the hopping parameters.
            phase_sampler (function): The distribution function for the single qubit z-rotation parameters at the
                end of a block

        Returns:
            np.ndarray:
                A random numpy array distributed according to the given distributions of size `self.num_parameters`
        """

        if mode == 'uniform':
            return uniform_sampler(self.num_parameters,
                                   lb=-np.pi / (2 * self.hopper_correction),
                                   ub=np.pi / (2 * self.hopper_correction))

        elif mode == 'multimode':
            return multimode_normal_sampler(self.num_parameters,
                                            modes=[-1, 1],
                                            width=np.pi / (8 * self.hopper_correction))

        elif mode == 'normal':
            return multimode_normal_sampler(self.num_parameters,
                                            modes=(0),
                                            width=1 / self.hopper_correction)

        elif mode == 'multimode-uniform':
            hopping_sampler = lambda size: multimode_normal_sampler(size,
                                                                    modes=[-1, 1],
                                                                    width=np.pi / (8 * self.hopper_correction))
            phase_sampler = lambda size: uniform_sampler(size,
                                                         lb=-np.pi / (2 * self.hopper_correction),
                                                         ub=np.pi / (2 * self.hopper_correction))

        elif mode == 'custom':
            if hopping_sampler is None or phase_sampler is None:
                raise UserWarning(
                    "Must provide functions for `hopping_sampler` and `phase_sampler` in 'custom' mode.")

        else:
            raise UserWarning("`mode` must be one of ['uniform', 'normal', 'multimode',"
                              " 'multimode-uniform', 'custom']")

        # Generate the random parameters for the first depth-block
        hoppings1 = hopping_sampler(self.num_hoppings_per_layer * self._num_couplings)
        phases1 = phase_sampler(self.num_phases_per_layer)
        params = np.hstack((hoppings1, phases1))

        # Generate the random parameters for the following blocks.
        for d in range(self._depth - 1):
            hoppingsd = hopping_sampler(self.num_hoppings_per_layer * self._num_couplings)
            phasesd = phase_sampler(self.num_phases_per_layer)
            params = np.hstack((params, hoppingsd, phasesd))
        return params

    def construct_circuit(self, parameters, q=None):
        """
        Construct the variational form, given its parameters.
        Args:
            parameters (numpy.ndarray): circuit parameters.
            coupling_matrices (list): list of np.ndarray for the coupling matrices
            q (QuantumRegister): Quantum Register for the circuit.
        Returns:
            QuantumCircuit: a quantum circuit with given `parameters`
        Raises:
            ValueError: the number of parameters is incorrect.
        """

        # 1 Parse inputs
        if len(parameters) != self._num_parameters:
            raise ValueError('The number of parameters has to be {}'.format(self._num_parameters))

        # 2 Set up the quantum register and the circuit
        if q is None:
            if self._initial_state is not None:
                circuit = self._initial_state.construct_circuit('circuit')
                q = circuit.qregs
            else:
                q = QuantumRegister(self._num_qubits, name='q')
                circuit = QuantumCircuit(q)
        else:
            circuit = self._initial_state.construct_circuit('circuit', q)
        all_qubits = circuit.qubits
        fermion_qubits = circuit.qubits[:self._lattice.nsites * 2]

        # 3 Build up the variational form with the parameters
        hoppings_per_layer = self.num_hoppings_per_layer
        mixed_hoppings_per_layer = self.num_hoppings_per_layer**2
        couplings_per_hopping = self._num_couplings
        phases_per_layer = self.num_phases_per_layer
        params_per_layer = couplings_per_hopping * hoppings_per_layer + phases_per_layer + mixed_hoppings_per_layer

        # 4. Iterate over depth to construct the individual blocks
        for d in range(self._depth):
            # 4.1 For each depth-block add parametrized exponentials of hopping terms
            hopping_params = parameters[d * params_per_layer: (d + 1) * params_per_layer
                                                              - (phases_per_layer + mixed_hoppings_per_layer)]
            mixed_hopping_params = parameters[(d + 1) * params_per_layer - (phases_per_layer + mixed_hoppings_per_layer):
                                   (d + 1) * params_per_layer - phases_per_layer]
            phase_params = parameters[(d + 1) * params_per_layer - phases_per_layer: (d + 1) * params_per_layer]

            for i, hopper1, parameter in zip(np.arange(hoppings_per_layer), self._hopping_terms, hopping_params):

                for j, hopper2 in enumerate(self._hopping_terms):
                    mixed_hopper = hopper1 * hopper2
                    mixed_param = mixed_hopping_params[j + i * hoppings_per_layer]

                    var_term = construct_trotter_step(mixed_hopper, mixed_param)
                    circuit.append(var_term, qargs=all_qubits)

                # print(hopper)
                var_term = construct_trotter_step(hopper1,
                                                  parameter,
                                                  name='h_{}({})'.format(
                                                      self._lattice.edges[i // couplings_per_hopping],
                                                      parameter))
                circuit.append(var_term, qargs=all_qubits)

            # 4.3 Add z-rotations for every second qubit at the end of one layer
            for qbit, phase_param in zip(fermion_qubits[1::2], phase_params):
                # circuit.u1(phase_param, qbit)
                circuit.rz(phase_param, qbit)

        return circuit


class WilsonLGT4(VariationalForm):
    """A variational form constructed from parametrized evolution with single-edge hopping terms
    in the kinetic part of the Hamiltonian."""
    CONFIGURATION = {
        'name': 'WilsonLGT',
        'description': 'LGT Variational Form for lattice gauge theories',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'lgt_schema',
            'type': 'object',
            'properties': {
                'depth': {
                    'type': 'integer',
                    'default': 1,
                    'minimum': 0
                },
            },
            'additionalProperties': False
        },
        'depends': [
            {
                'pluggable_type': 'initial_state',
                'default': {
                    'name': 'ZERO',
                }
            },
        ],
    }

    def __init__(self, lattice, S, rep=None, coupling_matrices=None, depth=1, initial_state=None):
        """Constructor for a hopping term based variational form for U(1) Lattice gauge theories
        with wilson fermions.

        Args:
            lattice (Lattice): The lattice on which the lattice gauge theory is fomulated
            S (float): The spin truncation value of the Quantum Link Model used to model the gauge fields.
                Must be a non-negative integer or half-integer.
            rep (dict): A dictionary specifying the representation of the Clifford algebra to be used
            depth (int): The depth of the variational circuit
            initial_state (qiskit.InitialState): The initial state object
        """
        # self.validate(locals())
        # validate_min('depth', depth, 0)
        super().__init__()

        # 1 Initialize the varform-circuit parameters
        self._num_qubits = lattice.nsites * 2 + lattice.nedges * int(np.ceil(np.log2(2 * S + 1)))
        self._depth = depth

        # 2 Initialize the var-form lattice parameters
        self._lattice = lattice
        self._S = S
        self._initial_state = initial_state

        # 3 Set the parametrized couplings. Either `rep` or `coupling_matrices` must be given as argument.
        # xor these two arguments:
        assert (rep is not None) ^ (coupling_matrices is not None), \
            'Either `rep` or `coupling_matrices` must be given.'

        # Note: The attribute self._coupling_matrices will be a 4d - numpy array of shape
        # (ndim, ncouplings_per_dim, ncomponents, ncomponents), i.e.
        # self._coupling_matrices[2][3] returns the `3`rd coupling matrix for hopping along
        # lattice dimension `2` as a (ncomponents, ncomponents) numpy array.
        if rep is not None:
            gamma = [val for val in rep.values()]
            self._coupling_matrices = np.array([np.array([-1j * gamma[0] @ gamma[hopping_dim + 1]])
                                                for hopping_dim in range(self._lattice.ndim)])

        elif coupling_matrices is not None:
            self._coupling_matrices = atleast_4d(coupling_matrices)

        # 4 Parse the coupling_matrices:
        # Each dimension should have a list of coupling matrices:
        if not self._coupling_matrices.shape[0] == self._lattice.ndim:
            raise UserWarning('Error with coupling matrices. The given lattice has {} dimensions but only coupling'
                              'matrices were given for {} dimensions. Please check that `coupling_matrices` has the '
                              'shape (lattice.ndim, n_couplings_per_dim, ncomponents, ncomponents)'.format(
                self._lattice.ndim, self._coupling_matrices.shape[0]))
        # And each coupling matrix should be a square matrices of size (ncomp, ncomp). Currently only 2 component
        # spinors are supported.
        if not self._coupling_matrices.shape[2:] == (2, 2):  # TODO: Generalize for more components spinors
            raise UserWarning('Error with coupling matrices. Your coupling matrices couple {} component spinors,'
                              'but currently only 2 component spinors are supported.'.format(
                self._coupling_matrices.shape[2]))

        # 5 Extract the number of parametrized couplings for every dimension
        self._num_couplings = self._coupling_matrices.shape[1]

        # 8 Set up the hopping terms along the edges
        self._construct_hopping_operators()
        self._construct_string_operators()
        self._matrix_hopping_terms = []
        self._matrix_string_terms = []

        # 6 Count the total number of parameters of the variational form:
        # one parameter for every edge (along which hopp. occurs) and every parametrized coupling plus
        # one parameter for final z-rotation at the end of each layer
        self._num_parameters = (lattice.nedges * self._num_couplings  # hopping terms
                                + self.num_strings_per_layer  # string terms
                                + lattice.nsites *2 ) * depth  # + aggiungiamo 2 per ogni qubit 

        # 7 Set bounds for the parameters
        hopping_bounds = [(-np.pi * self.hopper_correction,
                           np.pi * self.hopper_correction)] * self.num_hoppings_per_layer

        string_bounds = [(-np.pi * self.hopper_correction**2,
                           np.pi * self.hopper_correction**2)] * self.num_strings_per_layer

        phase_bounds = [(-np.pi, np.pi)] * self.num_phases_per_layer
        self._bounds = operator_sum([hopping_bounds + string_bounds + phase_bounds] * self._depth)

    @property
    def num_hoppings_per_layer(self):
        return self._lattice.nedges

    @property
    def num_strings_per_layer(self):
        return len(self._string_terms)

    @property
    def num_phases_per_layer(self):
        return self._lattice.nsites * 2 #aggiungiamo uno per ogni qubit 

    @property
    def num_hopping_parameters(self):
        return self.num_hoppings_per_layer * self._num_couplings * self._depth

    @property
    def num_strings_parameters(self):
        return self.num_strings_per_layer * self._depth

    @property
    def num_phase_parameters(self):
        return self.num_phases_per_layer * self._depth

    @property
    def hopper_correction(self) -> float:
        """ The correction for the norms of the hopping terms to make sure that all hopping strengths can be
        reached within the boundaries of (-np.pi, np.pi)"""
        if self._S == 0:
            return 1.
        else:
            return np.sqrt((self._S + 1) * self._S) * 2

    def _construct_hopping_operators(self, mode='qiskit'):
        """
        Constructs the Hamiltonian inspired hopping operators which are exponentiated when building the circuit and
        stores them in `self._hopping_terms` (for 'qiskit' mode) or `self._matrix_hopping_terms` (for 'qutip' mode).

        Args:
            mode (str): Must be one of ['qiskit', 'qutip']

        Returns:
            None
        """
        if mode == 'qiskit':
            self._hopping_terms = []  # will be a list of shape (nedges * ncouplings)
        elif mode == 'qutip':
            self._matrix_hopping_terms = []  # will be a list of shape (nedges * ncouplings)
        else:
            raise UserWarning("mode must be one of ['qiskit', 'qutip']")

        # 1. Iterate through all lattice edges
        for edge in self._lattice.edges:
            hopping_dim = edge[1]

            # 2. Iterate through all coupling matrices (mixmatrices) along the given dimension of the edge
            for mixmatrix in self._coupling_matrices[hopping_dim]:
                # Build up the hopping term along this edge with the specified coupling
                hopper_edge = hopping_like(edge, lattice=self._lattice, S=self._S, mixmat=mixmatrix, output=mode)
                # Simplify the operator if in qiskit mode.
                if mode == 'qiskit':
                    hopper_edge.chop()
                    self._hopping_terms.append(hopper_edge)
                elif mode == 'qutip':
                    self._matrix_hopping_terms.append(hopper_edge)

    def _construct_string_operators(self, mode='qiskit'):
        if mode == 'qiskit':
            self._string_terms = []  # will be a list of shape (nedges * ncouplings)
        elif mode == 'qutip':
            self._matrix_string_terms = []  # will be a list of shape (nedges * ncouplings)
        else:
            raise UserWarning("mode must be one of ['qiskit', 'qutip']")

        # 1. Iterate through all pairs of lattice edges
        # list_edge_pairs = get_list_edge_pairs(self._lattice)

        # Warning: Hardcoded version for the specific case of a 2 x 2 lattice
        edge1 = np.ndarray(2, dtype='int64')
        edge1[0] = 0
        edge1[1] = 0
        edge2 = np.ndarray(2, dtype='int64')
        edge2[0] = 1
        edge2[1] = 1
        edge3 = np.ndarray(2, dtype='int64')
        edge3[0] = 0
        edge3[1] = 1
        edge4 = np.ndarray(2, dtype='int64')
        edge4[0] = 2
        edge4[1] = 0
        list_edge_pairs = [[edge1, edge2], [edge3, edge4]]
        """
        for edge_pair in list_edge_pairs:

            # 2. Iterate through all coupling matrices (mixmatrices) along the given dimension of the edge
            for mixmatrix in self._coupling_matrices[0]:  # TODO: generalize for arbitrary coupling matrices
                # Build up the hopping term along this edge with the specified coupling
                string_edge = string_like(edge_pair, lattice=self._lattice, S=self._S, mixmat=mixmatrix, output=mode)
                # Simplify the operator if in qiskit mode
                if mode == 'qiskit':
                    string_edge.chop()
                    self._string_terms.append(string_edge)
                elif mode == 'qutip':
                    self._matrix_string_terms.append(string_edge)
        """
        mixmat = self._coupling_matrices[0][0] #così dovrebbe prendere solo una matrice 
        string_edge1 = string_like(list_edge_pairs[0], lattice=self._lattice, S=self._S, mixmat=mixmat, output=mode)
        string_edge1.chop()
        string_edge2 = string_like(list_edge_pairs[1], lattice=self._lattice, S=self._S, mixmat=mixmat, output=mode)
        string_edge2.chop()


        #mixstring1 = string_edge1.multiply(string_edge2)
        #for i,mix in enumerate(mixstring1.paulis):
        #    if np.imag(mix[0]) != 0 :
        #        mixstring1.paulis[i][0] *= 1j
        #mixstring1.chop()        

        self._string_terms.append(string_edge1)
        self._string_terms.append(string_edge2)

        #if not mix_term:
        #    print("\nNon c'é string term\n")
        #elif mix_term: 
        #    print("\nMettiamo mixed term\n")
        #    mixstring1 = string_edge1.multiply(string_edge2)
        #    for i,mix in enumerate(mixstring1.paulis):
        #       if np.imag(mix[0]) != 0 :
        #       mixstring1.paulis[i][0] *= 1j
        #    mixstring1.chop()  
        #    self._string_terms.append(mixstring1)
    """"
    def _construct_string_operators(self, mode='qiskit'):
        if mode == 'qiskit':
            self._string_terms = []  # will be a list of shape (nedges * ncouplings)
        elif mode == 'qutip':
            self._matrix_string_terms = []  # will be a list of shape (nedges * ncouplings)
        else:
            raise UserWarning("mode must be one of ['qiskit', 'qutip']")

        # 1. Iterate through all pairs of lattice edges
        # list_edge_pairs = get_list_edge_pairs(self._lattice)

        # Warning: Hardcoded version for the specific case of a 2 x 2 lattice
        edge1 = np.ndarray(2, dtype='int64')
        edge1[0] = 0
        edge1[1] = 0
        edge2 = np.ndarray(2, dtype='int64')
        edge2[0] = 1
        edge2[1] = 1
        edge3 = np.ndarray(2, dtype='int64')
        edge3[0] = 0
        edge3[1] = 1
        edge4 = np.ndarray(2, dtype='int64')
        edge4[0] = 2
        edge4[1] = 0
        list_edge_pairs = [[edge1, edge2], [edge3, edge4]]

        for edge_pair in list_edge_pairs:

            # 2. Iterate through all coupling matrices (mixmatrices) along the given dimension of the edge
            for mixmatrix in self._coupling_matrices[0]:  # TODO: generalize for arbitrary coupling matrices
                # Build up the hopping term along this edge with the specified coupling
                string_edge = string_like(edge_pair, lattice=self._lattice, S=self._S, mixmat=mixmatrix, output=mode)
                # Simplify the operator if in qiskit mode
                if mode == 'qiskit':
                    string_edge.chop()
                    self._string_terms.append(string_edge)
                elif mode == 'qutip':
                    self._matrix_string_terms.append(string_edge)

        mixmat = self._coupling_matrices[0][0]
        string_edge1 = string_like(list_edge_pairs[0], lattice=self._lattice, S=self._S, mixmat=mixmat, output=mode)
        string_edge2 = string_like(list_edge_pairs[1], lattice=self._lattice, S=self._S, mixmat=mixmat, output=mode)

        mixstring1 = string_edge1.multiply(string_edge2)
        mixstring2 = string_edge2.multiply(string_edge1)

        self._string_terms.append(mixstring1)
        self._string_terms.append(mixstring2)
    """

    def _construct_parametrized_evolution_circuits(self):
        # TODO
        pass

    def initialize_parameters(self, mode='uniform', hopping_sampler=None, phase_sampler=None):
        """
        Retruns a numpy array of randomly initialized parameters for the given variational form. The parameters
        are drawn from the specified `distribution`
        Args:
            mode (str): The initialization mode. Must be one of ['uniform', 'multimode', 'normal', 'multimode-normal',
                'custom']. If custom mode is used, distribution functions for the `hopping_sampler` and the
                `phase_sampler` must be given.
            hopping_sampler (function): The distribution function for the hopping parameters.
            phase_sampler (function): The distribution function for the single qubit z-rotation parameters at the
                end of a block

        Returns:
            np.ndarray:
                A random numpy array distributed according to the given distributions of size `self.num_parameters`
        """

        if mode == 'uniform':
            return uniform_sampler(self.num_parameters,
                                   lb=-np.pi / (2 * self.hopper_correction),
                                   ub=np.pi / (2 * self.hopper_correction))

        elif mode == 'multimode':
            return multimode_normal_sampler(self.num_parameters,
                                            modes=[-1, 1],
                                            width=np.pi / (8 * self.hopper_correction))

        elif mode == 'normal':
            return multimode_normal_sampler(self.num_parameters,
                                            modes=(0),
                                            width=1 / self.hopper_correction)

        elif mode == 'multimode-uniform':
            hopping_sampler = lambda size: multimode_normal_sampler(size,
                                                                    modes=[-1, 1],
                                                                    width=np.pi / (8 * self.hopper_correction))
            phase_sampler = lambda size: uniform_sampler(size,
                                                         lb=-np.pi / (2 * self.hopper_correction),
                                                         ub=np.pi / (2 * self.hopper_correction))

        elif mode == 'custom':
            if hopping_sampler is None or phase_sampler is None:
                raise UserWarning(
                    "Must provide functions for `hopping_sampler` and `phase_sampler` in 'custom' mode.")

        else:
            raise UserWarning("`mode` must be one of ['uniform', 'normal', 'multimode',"
                              " 'multimode-uniform', 'custom']")

        # Generate the random parameters for the first depth-block
        hoppings1 = hopping_sampler(self.num_hoppings_per_layer * self._num_couplings)
        phases1 = phase_sampler(self.num_phases_per_layer)
        params = np.hstack((hoppings1, phases1))

        # Generate the random parameters for the following blocks.
        for d in range(self._depth - 1):
            hoppingsd = hopping_sampler(self.num_hoppings_per_layer * self._num_couplings)
            phasesd = phase_sampler(self.num_phases_per_layer)
            params = np.hstack((params, hoppingsd, phasesd))
        return params

    def construct_circuit(self, parameters, q=None, num_time_slices=1):
        """
        Construct the variational form, given its parameters.
        Args:
            parameters (numpy.ndarray): circuit parameters.
            coupling_matrices (list): list of np.ndarray for the coupling matrices
            q (QuantumRegister): Quantum Register for the circuit.
        Returns:
            QuantumCircuit: a quantum circuit with given `parameters`
        Raises:
            ValueError: the number of parameters is incorrect.
        """
        # 1 Parse inputs
        if len(parameters) != self._num_parameters:
            raise ValueError('The number of parameters has to be {}'.format(self._num_parameters))

        # 2 Set up the quantum register and the circuit
        if q is None:
            if self._initial_state is not None:
                circuit = self._initial_state.construct_circuit('circuit')
                q = circuit.qregs
            else:
                q = QuantumRegister(self._num_qubits, name='q')
                circuit = QuantumCircuit(q)
        else:
            circuit = self._initial_state.construct_circuit('circuit', q)
        all_qubits = circuit.qubits
        fermion_qubits = circuit.qubits[:self._lattice.nsites * 2]

        # 3 Build up the variational form with the parameters
        hoppings_per_layer = self.num_hoppings_per_layer
        strings_per_layer = self.num_strings_per_layer
        couplings_per_hopping = self._num_couplings
        phases_per_layer = self.num_phases_per_layer
        params_per_layer = couplings_per_hopping * hoppings_per_layer + phases_per_layer + strings_per_layer

        # 4. Iterate over depth to construct the individual blocks
        for d in range(self._depth):
            # 4.1 For each depth-block add parametrized exponentials of hopping terms
            hopping_params = parameters[d * params_per_layer: (d + 1) * params_per_layer
                                                              - (phases_per_layer + strings_per_layer)]
            #print("hopping_params",hopping_params) 
            string_params = parameters[(d + 1) * params_per_layer - (phases_per_layer + strings_per_layer):
                                   (d + 1) * params_per_layer - phases_per_layer]
            #print("string_params",string_params) 
            phase_params = parameters[(d + 1) * params_per_layer - phases_per_layer: (d + 1) * params_per_layer]
            #print("phase_params",phase_params)
            #print("np.arange(hoppings_per_layer)", np.arange(hoppings_per_layer),"len(self._hopping_terms)", len(self._hopping_terms), "hopping_params ",hopping_params )
            #for i, hopper, parameter in zip(np.arange(hoppings_per_layer), self._hopping_terms, hopping_params):
            for i, hopper, parameter in zip(np.arange(len(self._hopping_terms)), self._hopping_terms, hopping_params):
                var_term = construct_trotter_step(hopper,
                                                  parameter,
                                                  name='h_{}({})'.format(
                                                      self._lattice.edges[i // couplings_per_hopping],
                                                      parameter), num_time_slices= num_time_slices)
                circuit.append(var_term, qargs=all_qubits)
            for string_term, parameter in zip(self._string_terms, string_params):
                var_term = construct_trotter_step(string_term, parameter, num_time_slices= num_time_slices)
                circuit.append(var_term, qargs=all_qubits)

            # 4.3 Add z-rotations for every second qubit at the end of one layer
            #for qbit, phase_param in zip(fermion_qubits[1::2], phase_params):
            for qbit, phase_param in zip(fermion_qubits, phase_params):
                circuit.u1(phase_param, qbit)

        return circuit

class WilsonLGT4_new(VariationalForm):
    """A variational form constructed from parametrized evolution with single-edge hopping terms
    in the kinetic part of the Hamiltonian."""
    CONFIGURATION = {
        'name': 'WilsonLGT',
        'description': 'LGT Variational Form for lattice gauge theories',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'lgt_schema',
            'type': 'object',
            'properties': {
                'depth': {
                    'type': 'integer',
                    'default': 1,
                    'minimum': 0
                },
            },
            'additionalProperties': False
        },
        'depends': [
            {
                'pluggable_type': 'initial_state',
                'default': {
                    'name': 'ZERO',
                }
            },
        ],
    }

    def __init__(self, lattice, S, rep=None, coupling_matrices=None, depth=1, initial_state=None, mix_bool = False , depth_phase=-1):
        """Constructor for a hopping term based variational form for U(1) Lattice gauge theories
        with wilson fermions.

        Args:
            lattice (Lattice): The lattice on which the lattice gauge theory is fomulated
            S (float): The spin truncation value of the Quantum Link Model used to model the gauge fields.
                Must be a non-negative integer or half-integer.
            rep (dict): A dictionary specifying the representation of the Clifford algebra to be used
            depth (int): The depth of the variational circuit
            initial_state (qiskit.InitialState): The initial state object
        """
        # self.validate(locals())
        # validate_min('depth', depth, 0)
        super().__init__()

        # 1 Initialize the varform-circuit parameters
        self._num_qubits = lattice.nsites * 2 + lattice.nedges * int(np.ceil(np.log2(2 * S + 1)))
        self._depth = depth

        # 2 Initialize the var-form lattice parameters
        self._lattice = lattice
        self._S = S
        self._initial_state = initial_state
        self._mix_bool = mix_bool
        self._depth_phase = depth_phase

        # 3 Set the parametrized couplings. Either `rep` or `coupling_matrices` must be given as argument.
        # xor these two arguments:
        assert (rep is not None) ^ (coupling_matrices is not None), \
            'Either `rep` or `coupling_matrices` must be given.'

        # Note: The attribute self._coupling_matrices will be a 4d - numpy array of shape
        # (ndim, ncouplings_per_dim, ncomponents, ncomponents), i.e.
        # self._coupling_matrices[2][3] returns the `3`rd coupling matrix for hopping along
        # lattice dimension `2` as a (ncomponents, ncomponents) numpy array.
        if rep is not None:
            gamma = [val for val in rep.values()]
            self._coupling_matrices = np.array([np.array([-1j * gamma[0] @ gamma[hopping_dim + 1]])
                                                for hopping_dim in range(self._lattice.ndim)])

        elif coupling_matrices is not None:
            self._coupling_matrices = atleast_4d(coupling_matrices)

        # 4 Parse the coupling_matrices:
        # Each dimension should have a list of coupling matrices:
        if not self._coupling_matrices.shape[0] == self._lattice.ndim:
            raise UserWarning('Error with coupling matrices. The given lattice has {} dimensions but only coupling'
                              'matrices were given for {} dimensions. Please check that `coupling_matrices` has the '
                              'shape (lattice.ndim, n_couplings_per_dim, ncomponents, ncomponents)'.format(
                self._lattice.ndim, self._coupling_matrices.shape[0]))
        # And each coupling matrix should be a square matrices of size (ncomp, ncomp). Currently only 2 component
        # spinors are supported.
        if not self._coupling_matrices.shape[2:] == (2, 2):  # TODO: Generalize for more components spinors
            raise UserWarning('Error with coupling matrices. Your coupling matrices couple {} component spinors,'
                              'but currently only 2 component spinors are supported.'.format(
                self._coupling_matrices.shape[2]))

        # 5 Extract the number of parametrized couplings for every dimension
        self._num_couplings = self._coupling_matrices.shape[1]

        # 8 Set up the hopping terms along the edges
        self._construct_hopping_operators()
        self._construct_string_operators()
        self._matrix_hopping_terms = []
        self._matrix_string_terms = []

        # 6 Count the total number of parameters of the variational form:
        # one parameter for every edge (along which hopp. occurs) and every parametrized coupling plus
        # one parameter for final z-rotation at the end of each layer
        if self._depth_phase == -1:
            self._num_parameters = (lattice.nedges * self._num_couplings  # hopping terms
                                + self.num_strings_per_layer  # string terms
                                + lattice.nsites * 2 ) * depth  # + aggiungiamo 2 per ogni qubit 
        elif self._depth_phase != -1:
            self._num_parameters = (lattice.nedges * self._num_couplings  # hopping terms
                                + self.num_strings_per_layer  # string terms
                                + lattice.nsites * 2 + lattice.nsites * 2 * depth_phase ) * depth 


        # 7 Set bounds for the parameters
        hopping_bounds = [(-np.pi * self.hopper_correction,
                           np.pi * self.hopper_correction)] * self.num_hoppings_per_layer

        string_bounds = [(-np.pi * self.hopper_correction**2,
                           np.pi * self.hopper_correction**2)] * self.num_strings_per_layer

        phase_bounds = [(-np.pi, np.pi)] * self.num_phases_per_layer
        self._bounds = operator_sum([hopping_bounds + string_bounds + phase_bounds] * self._depth)

    @property
    def num_hoppings_per_layer(self):
        return self._lattice.nedges

    @property
    def num_strings_per_layer(self):
        return len(self._string_terms)

    @property
    def num_phases_per_layer(self):
        if self._depth_phase != -1:
            return self._lattice.nsites * 2 + self._lattice.nsites * 2 * self._depth_phase
        elif self._depth_phase == -1:
            return self._lattice.nsites * 2
             

    @property
    def num_hopping_parameters(self):
        return self.num_hoppings_per_layer * self._num_couplings * self._depth

    @property
    def num_strings_parameters(self):
        return self.num_strings_per_layer * self._depth

    @property
    def num_phase_parameters(self):
        return self.num_phases_per_layer * self._depth

    @property
    def hopper_correction(self) -> float:
        """ The correction for the norms of the hopping terms to make sure that all hopping strengths can be
        reached within the boundaries of (-np.pi, np.pi)"""
        if self._S == 0:
            return 1.
        else:
            return np.sqrt((self._S + 1) * self._S) * 2

    def _construct_hopping_operators(self, mode='qiskit'):
        """
        Constructs the Hamiltonian inspired hopping operators which are exponentiated when building the circuit and
        stores them in `self._hopping_terms` (for 'qiskit' mode) or `self._matrix_hopping_terms` (for 'qutip' mode).

        Args:
            mode (str): Must be one of ['qiskit', 'qutip']

        Returns:
            None
        """
        if mode == 'qiskit':
            self._hopping_terms = []  # will be a list of shape (nedges * ncouplings)
        elif mode == 'qutip':
            self._matrix_hopping_terms = []  # will be a list of shape (nedges * ncouplings)
        else:
            raise UserWarning("mode must be one of ['qiskit', 'qutip']")

        # 1. Iterate through all lattice edges
        for edge in self._lattice.edges:
            hopping_dim = edge[1]

            # 2. Iterate through all coupling matrices (mixmatrices) along the given dimension of the edge
            for mixmatrix in self._coupling_matrices[hopping_dim]:
                # Build up the hopping term along this edge with the specified coupling
                hopper_edge = hopping_like(edge, lattice=self._lattice, S=self._S, mixmat=mixmatrix, output=mode)
                # Simplify the operator if in qiskit mode.
                if mode == 'qiskit':
                    hopper_edge.chop()
                    self._hopping_terms.append(hopper_edge)
                elif mode == 'qutip':
                    self._matrix_hopping_terms.append(hopper_edge)

    def _construct_string_operators(self, mode='qiskit'):
        if mode == 'qiskit':
            self._string_terms = []  # will be a list of shape (nedges * ncouplings)
        elif mode == 'qutip':
            self._matrix_string_terms = []  # will be a list of shape (nedges * ncouplings)
        else:
            raise UserWarning("mode must be one of ['qiskit', 'qutip']")

        # 1. Iterate through all pairs of lattice edges
        # list_edge_pairs = get_list_edge_pairs(self._lattice)

        # Warning: Hardcoded version for the specific case of a 2 x 2 lattice
        edge1 = np.ndarray(2, dtype='int64')
        edge1[0] = 0
        edge1[1] = 0
        edge2 = np.ndarray(2, dtype='int64')
        edge2[0] = 1
        edge2[1] = 1
        edge3 = np.ndarray(2, dtype='int64')
        edge3[0] = 0
        edge3[1] = 1
        edge4 = np.ndarray(2, dtype='int64')
        edge4[0] = 2
        edge4[1] = 0
        list_edge_pairs = [[edge1, edge2], [edge3, edge4]]
        """
        for edge_pair in list_edge_pairs:

            # 2. Iterate through all coupling matrices (mixmatrices) along the given dimension of the edge
            for mixmatrix in self._coupling_matrices[0]:  # TODO: generalize for arbitrary coupling matrices
                # Build up the hopping term along this edge with the specified coupling
                string_edge = string_like(edge_pair, lattice=self._lattice, S=self._S, mixmat=mixmatrix, output=mode)
                # Simplify the operator if in qiskit mode
                if mode == 'qiskit':
                    string_edge.chop()
                    self._string_terms.append(string_edge)
                elif mode == 'qutip':
                    self._matrix_string_terms.append(string_edge)
        """
        mixmat = self._coupling_matrices[0][0] #così dovrebbe prendere solo una matrice 
        string_edge1 = string_like(list_edge_pairs[0], lattice=self._lattice, S=self._S, mixmat=mixmat, output=mode)
        string_edge1.chop()
        string_edge2 = string_like(list_edge_pairs[1], lattice=self._lattice, S=self._S, mixmat=mixmat, output=mode)
        string_edge2.chop()


        #mixstring1 = string_edge1.multiply(string_edge2)
        #for i,mix in enumerate(mixstring1.paulis):
        #    if np.imag(mix[0]) != 0 :
        #        mixstring1.paulis[i][0] *= 1j
        #mixstring1.chop()        

        self._string_terms.append(string_edge1)
        self._string_terms.append(string_edge2)

        if not self._mix_bool:
            print("\nNon c'é mixed term\n")
        elif self._mix_bool: 
            print("\nMettiamo mixed term\n")
            mixstring1 = string_edge1.multiply(string_edge2)
            for i,mix in enumerate(mixstring1.paulis):
               if np.imag(mix[0]) != 0 :
                mixstring1.paulis[i][0] *= 1j
            mixstring1.chop()  
            self._string_terms.append(mixstring1)
    """"
    def _construct_string_operators(self, mode='qiskit'):
        if mode == 'qiskit':
            self._string_terms = []  # will be a list of shape (nedges * ncouplings)
        elif mode == 'qutip':
            self._matrix_string_terms = []  # will be a list of shape (nedges * ncouplings)
        else:
            raise UserWarning("mode must be one of ['qiskit', 'qutip']")

        # 1. Iterate through all pairs of lattice edges
        # list_edge_pairs = get_list_edge_pairs(self._lattice)

        # Warning: Hardcoded version for the specific case of a 2 x 2 lattice
        edge1 = np.ndarray(2, dtype='int64')
        edge1[0] = 0
        edge1[1] = 0
        edge2 = np.ndarray(2, dtype='int64')
        edge2[0] = 1
        edge2[1] = 1
        edge3 = np.ndarray(2, dtype='int64')
        edge3[0] = 0
        edge3[1] = 1
        edge4 = np.ndarray(2, dtype='int64')
        edge4[0] = 2
        edge4[1] = 0
        list_edge_pairs = [[edge1, edge2], [edge3, edge4]]

        for edge_pair in list_edge_pairs:

            # 2. Iterate through all coupling matrices (mixmatrices) along the given dimension of the edge
            for mixmatrix in self._coupling_matrices[0]:  # TODO: generalize for arbitrary coupling matrices
                # Build up the hopping term along this edge with the specified coupling
                string_edge = string_like(edge_pair, lattice=self._lattice, S=self._S, mixmat=mixmatrix, output=mode)
                # Simplify the operator if in qiskit mode
                if mode == 'qiskit':
                    string_edge.chop()
                    self._string_terms.append(string_edge)
                elif mode == 'qutip':
                    self._matrix_string_terms.append(string_edge)

        mixmat = self._coupling_matrices[0][0]
        string_edge1 = string_like(list_edge_pairs[0], lattice=self._lattice, S=self._S, mixmat=mixmat, output=mode)
        string_edge2 = string_like(list_edge_pairs[1], lattice=self._lattice, S=self._S, mixmat=mixmat, output=mode)

        mixstring1 = string_edge1.multiply(string_edge2)
        mixstring2 = string_edge2.multiply(string_edge1)

        self._string_terms.append(mixstring1)
        self._string_terms.append(mixstring2)
    """

    def _construct_parametrized_evolution_circuits(self):
        # TODO
        pass

    def initialize_parameters(self, mode='uniform', hopping_sampler=None, phase_sampler=None):
        """
        Retruns a numpy array of randomly initialized parameters for the given variational form. The parameters
        are drawn from the specified `distribution`
        Args:
            mode (str): The initialization mode. Must be one of ['uniform', 'multimode', 'normal', 'multimode-normal',
                'custom']. If custom mode is used, distribution functions for the `hopping_sampler` and the
                `phase_sampler` must be given.
            hopping_sampler (function): The distribution function for the hopping parameters.
            phase_sampler (function): The distribution function for the single qubit z-rotation parameters at the
                end of a block

        Returns:
            np.ndarray:
                A random numpy array distributed according to the given distributions of size `self.num_parameters`
        """

        if mode == 'uniform':
            return uniform_sampler(self.num_parameters,
                                   lb=-np.pi / (2 * self.hopper_correction),
                                   ub=np.pi / (2 * self.hopper_correction))

        elif mode == 'multimode':
            return multimode_normal_sampler(self.num_parameters,
                                            modes=[-1, 1],
                                            width=np.pi / (8 * self.hopper_correction))

        elif mode == 'normal':
            return multimode_normal_sampler(self.num_parameters,
                                            modes=(0),
                                            width=1 / self.hopper_correction)

        elif mode == 'multimode-uniform':
            hopping_sampler = lambda size: multimode_normal_sampler(size,
                                                                    modes=[-1, 1],
                                                                    width=np.pi / (8 * self.hopper_correction))
            phase_sampler = lambda size: uniform_sampler(size,
                                                         lb=-np.pi / (2 * self.hopper_correction),
                                                         ub=np.pi / (2 * self.hopper_correction))

        elif mode == 'custom':
            if hopping_sampler is None or phase_sampler is None:
                raise UserWarning(
                    "Must provide functions for `hopping_sampler` and `phase_sampler` in 'custom' mode.")

        else:
            raise UserWarning("`mode` must be one of ['uniform', 'normal', 'multimode',"
                              " 'multimode-uniform', 'custom']")

        # Generate the random parameters for the first depth-block
        hoppings1 = hopping_sampler(self.num_hoppings_per_layer * self._num_couplings)
        phases1 = phase_sampler(self.num_phases_per_layer)
        params = np.hstack((hoppings1, phases1))

        # Generate the random parameters for the following blocks.
        for d in range(self._depth - 1):
            hoppingsd = hopping_sampler(self.num_hoppings_per_layer * self._num_couplings)
            phasesd = phase_sampler(self.num_phases_per_layer)
            params = np.hstack((params, hoppingsd, phasesd))
        return params

    def construct_circuit(self, parameters, q=None, num_time_slices=1):
        """
        Construct the variational form, given its parameters.
        Args:
            parameters (numpy.ndarray): circuit parameters.
            coupling_matrices (list): list of np.ndarray for the coupling matrices
            q (QuantumRegister): Quantum Register for the circuit.
        Returns:
            QuantumCircuit: a quantum circuit with given `parameters`
        Raises:
            ValueError: the number of parameters is incorrect.
        """
        # 1 Parse inputs
        if len(parameters) != self._num_parameters:
            raise ValueError('The number of parameters has to be {}'.format(self._num_parameters))

        # 2 Set up the quantum register and the circuit
        if q is None:
            if self._initial_state is not None:
                circuit = self._initial_state.construct_circuit('circuit')
                q = circuit.qregs
            else:
                q = QuantumRegister(self._num_qubits, name='q')
                circuit = QuantumCircuit(q)
        else:
            circuit = self._initial_state.construct_circuit('circuit', q)
        all_qubits = circuit.qubits
        fermion_qubits = circuit.qubits[:self._lattice.nsites * 2]

        # 3 Build up the variational form with the parameters
        hoppings_per_layer = self.num_hoppings_per_layer
        strings_per_layer = self.num_strings_per_layer
        couplings_per_hopping = self._num_couplings
        phases_per_layer = self.num_phases_per_layer
        params_per_layer = couplings_per_hopping * hoppings_per_layer + phases_per_layer + strings_per_layer

        # 4. Iterate over depth to construct the individual blocks
        for d in range(self._depth):
            # 4.1 For each depth-block add parametrized exponentials of hopping terms
            hopping_params = parameters[d * params_per_layer: (d + 1) * params_per_layer
                                                              - (phases_per_layer + strings_per_layer)]
            #print("hopping_params",hopping_params) 
            string_params = parameters[(d + 1) * params_per_layer - (phases_per_layer + strings_per_layer):
                                   (d + 1) * params_per_layer - phases_per_layer]
            #print("string_params",string_params) 
            phase_params = parameters[(d + 1) * params_per_layer - phases_per_layer: (d + 1) * params_per_layer]
            #print("phase_params",phase_params)
            #print("np.arange(hoppings_per_layer)", np.arange(hoppings_per_layer),"len(self._hopping_terms)", len(self._hopping_terms), "hopping_params ",hopping_params )
            #for i, hopper, parameter in zip(np.arange(hoppings_per_layer), self._hopping_terms, hopping_params):
            if self._depth_phase == -1: 
                for i, hopper, parameter in zip(np.arange(len(self._hopping_terms)), self._hopping_terms, hopping_params):
                    var_term = construct_trotter_step(hopper,
                                                  parameter,
                                                  name='h_{}({})'.format(
                                                      self._lattice.edges[i // couplings_per_hopping],
                                                      parameter), num_time_slices= num_time_slices)
                    circuit.append(var_term, qargs=all_qubits)
                for string_term, parameter in zip(self._string_terms, string_params):
                    var_term = construct_trotter_step(string_term, parameter, num_time_slices= num_time_slices)
                    circuit.append(var_term, qargs=all_qubits)

                for qbit, phase_param in zip(fermion_qubits, phase_params):
                    circuit.u1(phase_param, qbit)


            elif self._depth_phase != -1: 
                #phase_params_last = phase_params[ -self._lattice.nsites * 2 :]
                #phase_params = phase_params[ :self._lattice.nsites * 2 ]
                var_terms_circ = []
                for i, hopper, parameter in zip(np.arange(len(self._hopping_terms)), self._hopping_terms, hopping_params):
                    var_terms_circ.append(construct_trotter_step(hopper,
                                                  parameter,
                                                  name='h_{}({})'.format(
                                                      self._lattice.edges[i // couplings_per_hopping],
                                                      parameter), num_time_slices= num_time_slices))
                string_terms_circ = []
                for string_term, parameter in zip(self._string_terms, string_params):
                    string_terms_circ.append(construct_trotter_step(string_term, parameter, num_time_slices= num_time_slices))
                

                phases_per_layer_phase = self._lattice.nsites * 2 
                #print(phases_per_layer_phase , int(phases_per_layer_phase ), phases_per_layer, self._depth_phase  )
                #occhio che in GT& é diverso len(varform._hopping_terms)hoppings_per_layer
                if self._depth_phase <= len(self._hopping_terms):
                    #forward
                    n_rest = len(self._hopping_terms) - self._depth_phase
                    #print(n_rest)
                    #print(len(self._hopping_terms) )
                    for var_term in var_terms_circ[:n_rest]:
                        circuit.append(var_term, qargs=all_qubits)

                    for d in range(self._depth_phase):
                        phase_params_per_d_phase = phase_params[d * phases_per_layer_phase : (d+1) * phases_per_layer_phase ]
                        
                        circuit.append(var_terms_circ[n_rest+d], qargs=all_qubits)
                        
                        for qbit, phase_param in zip(fermion_qubits, phase_params_per_d_phase):
                            circuit.rz(phase_param, qbit)
                    
                    for var_term in string_terms_circ:
                        circuit.append(var_term, qargs=all_qubits)

                    

                    for qbit, phase_param in zip(fermion_qubits, phase_params[(self._depth_phase) * self._lattice.nsites * 2  :]):
                        circuit.u1(phase_param, qbit)
                    

        return circuit


class WilsonStringLike(VariationalForm):
    """A variational form constructed from parametrized evolution with single-edge hopping terms
    in the kinetic part of the Hamiltonian."""
    CONFIGURATION = {
        'name': 'WilsonStringLike',
        'description': 'LGT String-like Variational Form for lattice gauge theories',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'lgt_schema',
            'type': 'object',
            'properties': {
                'depth': {
                    'type': 'integer',
                    'default': 1,
                    'minimum': 0
                },
            },
            'additionalProperties': False
        },
        'depends': [
            {
                'pluggable_type': 'initial_state',
                'default': {
                    'name': 'ZERO',
                }
            },
        ],
    }

    def __init__(self, lattice, S, rep=None, coupling_matrices=None, string_length=2, depth=1, initial_state=None):
        """Constructor for a hopping term based variational form for U(1) Lattice gauge theories
        with wilson fermions.

        Args:
            lattice (Lattice): The lattice on which the lattice gauge theory is fomulated
            S (float): The spin truncation value of the Quantum Link Model used to model the gauge fields.
                Must be a non-negative integer or half-integer.
            rep (dict): A dictionary specifying the representation of the Clifford algebra to be used
            string_length (int): Length of the String-like terms, i.e. the number of edges connecting two sites.
            depth (int): The depth of the variational circuit
            initial_state (qiskit.InitialState): The initial state object
        """
        # self.validate(locals())
        # validate_min('depth', depth, 0)
        super().__init__()

        # 1 Initialize the varform-circuit parameters
        self._num_qubits = lattice.nsites * 2 + lattice.nedges * int(np.ceil(np.log2(2 * S + 1)))
        self._depth = depth

        # 2 Initialize the var-form lattice parameters
        self._lattice = lattice
        self._S = S
        self._initial_state = initial_state
        self._string_length = string_length

        # 3 Set the parametrized couplings. Either `rep` or `coupling_matrices` must be given as argument.
        # xor these two arguments:
        assert (rep is not None) ^ (coupling_matrices is not None), \
            'Either `rep` or `coupling_matrices` must be given.'

        # Note: The attribute self._coupling_matrices will be a 4d - numpy array of shape
        # (ndim, ncouplings_per_dim, ncomponents, ncomponents), i.e.
        # self._coupling_matrices[2][3] returns the `3`rd coupling matrix for hopping along
        # lattice dimension `2` as a (ncomponents, ncomponents) numpy array.
        if rep is not None:
            gamma = [val for val in rep.values()]
            self._coupling_matrices = np.array([np.array([-1j * gamma[0] @ gamma[hopping_dim + 1]])
                                                for hopping_dim in range(self._lattice.ndim)])

        elif coupling_matrices is not None:
            self._coupling_matrices = atleast_4d(coupling_matrices)
            #print("self._coupling_matrices ", self._coupling_matrices  )
        # 4 Parse the coupling_matrices:
        # Each dimension should have a list of coupling matrices:
        if not self._coupling_matrices.shape[0] == self._lattice.ndim:
            raise UserWarning('Error with coupling matrices. The given lattice has {} dimensions but only coupling'
                              'matrices were given for {} dimensions. Please check that `coupling_matrices` has the '
                              'shape (lattice.ndim, n_couplings_per_dim, ncomponents, ncomponents)'.format(
                self._lattice.ndim, self._coupling_matrices.shape[0]))
        # And each coupling matrix should be a square matrices of size (ncomp, ncomp). Currently only 2 component
        # spinors are supported.
        if not self._coupling_matrices.shape[2:] == (2, 2):  # TODO: Generalize for more components spinors
            raise UserWarning('Error with coupling matrices. Your coupling matrices couple {} component spinors,'
                              'but currently only 2 component spinors are supported.'.format(
                self._coupling_matrices.shape[2]))

        # 5 Extract the number of parametrized couplings for every dimension
        self._num_couplings = self._coupling_matrices.shape[1]
        
        # 8 Set up the string terms along the edges
        self._construct_string_operators()
        self._matrix_string_terms = []

        # 6 Count the total number of parameters of the variational form:
        # one parameter for every edge (along which hopp. occurs) and every parametrized coupling plus
        # one parameter for final z-rotation at the end of each layer
        self._num_parameters = (self.num_strings_per_layer * self._num_couplings) * depth

        # 7 Set bounds for the parameters
        string_bounds = [(-np.pi * self.hopper_correction**2,
                          np.pi * self.hopper_correction**2)] * self.num_strings_per_layer * self._num_couplings

        self._bounds = operator_sum([string_bounds] * self._depth)

    @property
    def num_strings_per_layer(self):
        return len(self._string_terms)

    @property
    def num_strings_parameters(self):
        return self.num_strings_per_layer * self._num_couplings * self._depth

    @property
    def hopper_correction(self) -> float:
        """ The correction for the norms of the hopping terms to make sure that all hopping strengths can be
        reached within the boundaries of (-np.pi, np.pi)"""
        if self._S == 0:
            return 1.
        else:
            return np.sqrt((self._S + 1) * self._S) * 2

    def _construct_string_operators(self, mode='qiskit'):
        """
        Constructs the Hamiltonian inspired string operators which are exponentiated when building the circuit and
        stores them in `self._hopping_terms` (for 'qiskit' mode) or `self._matrix_hopping_terms` (for 'qutip' mode).

        Args:
            mode (str): Must be one of ['qiskit', 'qutip']

        Returns:
            None
        """
        if mode == 'qiskit':
            self._string_terms = []  # will be a list of shape (nedges * ncouplings)
        elif mode == 'qutip':
            self._matrix_string_terms = []  # will be a list of shape (nedges * ncouplings)
        else:
            raise UserWarning("mode must be one of ['qiskit', 'qutip']")

        # 1. Iterate through all pairs of lattice edges
        # list_edge_pairs = get_list_edge_pairs(self._lattice)

        # Warning: Hardcoded version for the specific case of a 2 x 2 lattice
        """
        edge1 = np.ndarray(2, dtype='int64')
        edge1[0] = 0
        edge1[1] = 0
        edge2 = np.ndarray(2, dtype='int64')
        edge2[0] = 1
        edge2[1] = 1
        edge3 = np.ndarray(2, dtype='int64')
        edge3[0] = 0
        edge3[1] = 1
        edge4 = np.ndarray(2, dtype='int64')
        edge4[0] = 2
        edge4[1] = 0
        list_edge_pairs = [[edge1, edge2], [edge3, edge4]]
        list_edge_pairs = [[[0,0], [1,1]],[[0,1], [2,0]]]
        list_edge_pairs = [[0,0], [1,1]]
        """
        #for edge_pair in list_edge_pairs:

            # 2. Iterate through all coupling matrices (mixmatrices) along the given dimension of the edge
            #for mixmatrix in self._coupling_matrices[0]:  # TODO: generalize for arbitrary coupling matrices
                # Build up the hopping term along this edge with the specified coupling
            #    string_edge = string_like(edge_pair, lattice=self._lattice, S=self._S, mixmat=mixmatrix, output=mode)
                # Simplify the operator if in qiskit mode
            #    if mode == 'qiskit':
            #        string_edge.chop()
            #        self._string_terms.append(string_edge)
            #    elif mode == 'qutip':
            #        self._matrix_string_terms.append(string_edge)
        mixmat = self._coupling_matrices[0][0]

        edges_list= []
        N = self._lattice.nsites
        for m in range(2, N):
            for i in range(N-m):
                v = np.arange(i,i + m )
                el= []
                for v_el in v:
                    el.append([v_el,0])
                edges_list.append(el)
        
        for i in range(len(edges_list)):
            print(len(edges_list))
            string_edge = string_like(edges_list[i], lattice=self._lattice, S=self._S, mixmat=mixmat, output=mode)
            string_edge.chop()
            self._string_terms.append(string_edge)

        
   
        #string_edge1 = string_like(list_edge_pairs, lattice=self._lattice, S=self._S, mixmat=mixmat, output=mode)
        #string_edge2 = string_like(list_edge_pairs[1], lattice=self._lattice, S=self._S, mixmat=mixmat, output=mode)
        #self._string_terms.append(string_edge1)
        #self._string_terms.append(string_edge2)
        #
        #mixstring1 = string_edge1.multiply(string_edge2)
        #mixstring2 = string_edge2.multiply(string_edge1)

        #for i,mix in enumerate(mixstring1.paulis):
        #    if np.imag(mix[0]) != 0 :
        #        mixstring1.paulis[i][0] *= 1j

        #for i,mix in enumerate(mixstring2.paulis):
        #    if np.imag(mix[0]) != 0 :
        #        mixstring2.paulis[i][0] *= 1j


        #mixstring1.chop()
        #self._string_terms.append(mixstring1)
        #self._string_terms.append(mixstring2)

    def _construct_parametrized_evolution_circuits(self):
        # TODO
        pass

    def initialize_parameters(self, mode='uniform', string_sampler=None):
        """
        Retruns a numpy array of randomly initialized parameters for the given variational form. The parameters
        are drawn from the specified `distribution`
        Args:
            mode (str): The initialization mode. Must be one of ['uniform', 'multimode', 'normal', 'multimode-normal',
                'custom']. If custom mode is used, distribution functions for the `hopping_sampler` and the
                `phase_sampler` must be given.
            string_sampler (function): The distribution function for the strings parameters.

        Returns:
            np.ndarray:
                A random numpy array distributed according to the given distributions of size `self.num_parameters`
        """

        if mode == 'uniform':
            return uniform_sampler(self.num_parameters,
                                   lb=-np.pi / (2 * self.hopper_correction),
                                   ub=np.pi / (2 * self.hopper_correction))

        elif mode == 'multimode':
            return multimode_normal_sampler(self.num_parameters,
                                            modes=[-1, 1],
                                            width=np.pi / (8 * self.hopper_correction))

        elif mode == 'normal':
            return multimode_normal_sampler(self.num_parameters,
                                            modes=(0),
                                            width=1 / self.hopper_correction)

        elif mode == 'custom':
            if string_sampler is None:
                raise UserWarning(
                    "Must provide functions for `string_sampler`  in 'custom' mode.")

        else:
            raise UserWarning("`mode` must be one of ['uniform', 'normal', 'multimode', 'custom']")

        # Generate the random parameters for the first depth-block
        strings1 = string_sampler(self.num_strings_per_layer * self._num_couplings)
        params = strings1

        # Generate the random parameters for the following blocks.
        for d in range(self._depth - 1):
            stringsd = string_sampler(self.num_strings_per_layer * self._num_couplings)
            params = np.hstack((params, stringsd))
        return params

    def construct_circuit(self, parameters, q=None, num_time_slices=1):
        """
        Construct the variational form, given its parameters.
        Args:
            parameters (numpy.ndarray): circuit parameters.
            q (QuantumRegister): Quantum Register for the circuit.
        Returns:
            QuantumCircuit: a quantum circuit with given `parameters`
        Raises:
            ValueError: the number of parameters is incorrect.
        """
        edges_list= []
        N = self._lattice.nsites
        for m in range(2, N):
            for i in range(N-m):
                v = np.arange(i,i + m )
                el= []
                for v_el in v:
                    el.append([v_el,0])
                edges_list.append(el)
        #print(edges_list[i])
        #print(edges_list[i][0][0], edges_list[i][-1][0])

        # 1 Parse inputs
        if len(parameters) != self._num_parameters:
            raise ValueError('The number of parameters has to be {}'.format(self._num_parameters))

        # 2 Set up the quantum register and the circuit
        if q is None:
            if self._initial_state is not None:
                circuit = self._initial_state.construct_circuit('circuit')
                q = circuit.qregs
            else:
                q = QuantumRegister(self._num_qubits, name='q')
                circuit = QuantumCircuit(q)
        else:
            circuit = self._initial_state.construct_circuit('circuit', q)
        all_qubits = circuit.qubits
        fermion_qubits = circuit.qubits[:self._lattice.nsites * 2]

        # 3 Build up the variational form with the parameters
        strings_per_layer = self.num_strings_per_layer
        couplings_per_string = self._num_couplings
        params_per_layer = couplings_per_string * strings_per_layer

        # 4. Iterate over depth to construct the individual blocks
        for d in range(self._depth):
            # 4.1 For each depth-block add parametrized exponentials of hopping terms
            string_params = parameters[d * params_per_layer: (d + 1) * params_per_layer]

            for i,string_term, parameter in zip(np.arange(len(self._string_terms)),self._string_terms, string_params):
                var_term = construct_trotter_step(string_term, parameter, num_time_slices=num_time_slices, name='s_[{},{}]({})'.format(edges_list[i][0][0],(edges_list[i][-1][0]+1),
                                                                         parameter))
                circuit.append(var_term, qargs=all_qubits)

        return circuit

class WilsonLGT6(VariationalForm):
   
    CONFIGURATION = {
        'name': 'WilsonLGT',
        'description': 'LGT Variational Form for lattice gauge theories',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'lgt_schema',
            'type': 'object',
            'properties': {
                'depth': {
                    'type': 'integer',
                    'default': 1,
                    'minimum': 0
                },
            },
            'additionalProperties': False
        },
        'depends': [
            {
                'pluggable_type': 'initial_state',
                'default': {
                    'name': 'ZERO',
                }
            },
        ],
    }

    def __init__(self, lattice, S, rep=None, coupling_matrices=None, depth=1, initial_state=None, depth_phase=1):
        """Constructor for a hopping term based variational form for U(1) Lattice gauge theories
        with wilson fermions.

        Args:
            lattice (Lattice): The lattice on which the lattice gauge theory is fomulated
            S (float): The spin truncation value of the Quantum Link Model used to model the gauge fields.
                Must be a non-negative integer or half-integer.
            rep (dict): A dictionary specifying the representation of the Clifford algebra to be used
            depth (int): The depth of the variational circuit
            initial_state (qiskit.InitialState): The initial state object
        """
        #self.validate(locals())
        #validate_min('depth', depth, 0)
        super().__init__()

        # 1 Initialize the varform-circuit parameters
        self._num_qubits = lattice.nsites * 2 + lattice.nedges * int(np.ceil(np.log2(2 * S + 1)))
        self._depth = depth
        self._depth_phase = depth_phase

        # 2 Initialize the var-form lattice parameters
        self._lattice = lattice
        self._S = S
        self._initial_state = initial_state

        # 3 Set the parametrized couplings. Either `rep` or `coupling_matrices` must be given as argument.
        # xor these two arguments:
        assert (rep is not None) ^ (coupling_matrices is not None), \
            'Either `rep` or `coupling_matrices` must be given.'

        # Note: The attribute self._coupling_matrices will be a 4d - numpy array of shape
        # (ndim, ncouplings_per_dim, ncomponents, ncomponents), i.e.
        # self._coupling_matrices[2][3] returns the `3`rd coupling matrix for hopping along
        # lattice dimension `2` as a (ncomponents, ncomponents) numpy array.
        #praticamente se non 
        if rep is not None:
            gamma = [val for val in rep.values()]
            self._coupling_matrices = np.array([np.array([-1j * gamma[0] @ gamma[hopping_dim + 1]])
                                           for hopping_dim in range(self._lattice.ndim)])

        elif coupling_matrices is not None:
            self._coupling_matrices = atleast_4d(coupling_matrices)

        # 4 Parse the coupling_matrices:
        # Each dimension should have a list of coupling matrices:
        if not self._coupling_matrices.shape[0] == self._lattice.ndim:
            raise UserWarning('Error with coupling matrices. The given lattice has {} dimensions but only coupling'
                              'matrices were given for {} dimensions. Please check that `coupling_matrices` has the '
                              'shape (lattice.ndim, n_couplings_per_dim, ncomponents, ncomponents)'.format(
                self._lattice.ndim, self._coupling_matrices.shape[0]))
        # And each coupling matrix should be a square matrices of size (ncomp, ncomp). Currently only 2 component
        # spinors are supported.
        if not self._coupling_matrices.shape[2:] == (2,2): # TODO: Generalize for more components spinors
            raise UserWarning('Error with coupling matrices. Your coupling matrices couple {} component spinors,'
                              'but currently only 2 component spinors are supported.'.format(
                self._coupling_matrices.shape[2]))

        # 5 Extract the number of parametrized couplings for every dimension
        self._num_couplings = self._coupling_matrices.shape[1]
        ###################### Attenzione ################
        ## num couplings é il numero di thetass 
        #non farlo perhcé aumentano anche i parametri che almeno quelli sono giusti
        #self._num_couplings = self._coupling_matrices.shape[1] * lattice.nedges
        ###################### Attenzione######

        # 6 Count the total number of parameters of the variational form:
        # one parameter for every edge (along which hopp. occurs) and every parametrized coupling plus
        # one parameter for final z-rotation at the end of each layer

        if depth_phase == -1: 
            self._num_parameters = (lattice.nedges * self._num_couplings             # hopping terms
                                + lattice.nsites*2) * depth          #  occhio metto una rotazione per fermionqubit+ spinor component z-rotations
        elif depth_phase != -1: 
            self._num_parameters = (lattice.nedges * self._num_couplings + lattice.nsites* 2  * depth_phase  ) * depth 
       
        
        # 7 Set bounds for the parameters
        hopping_bounds = [(-np.pi * self.hopper_correction,
                                     np.pi * self.hopper_correction)] * self.num_hoppings_per_layer
        phase_bounds = [(-np.pi, np.pi)] * self.num_phases_per_layer
        self._bounds = operator_sum([hopping_bounds + phase_bounds] * self._depth)

        # 8 Set up the hopping terms along the edges
        self._construct_hopping_operators()
        self._matrix_hopping_terms = []

###### Atttenzione ########  sotto ne contava solo uno  
    @property
    def num_hoppings_per_layer(self):
        #return self._lattice.nedges
        #assumo che per ogni coupling matrice c'é un coupling matrice
        return self._num_couplings
###### Attenzione ##########
    @property
    def num_phases_per_layer(self):
        if self._depth_phase == -1: 
            return (self._lattice.nsites * 2) 
        elif self._depth_phase != -1:
            return (self._lattice.nsites * 2)  * self._depth_phase 
        #ne metto una per qubit 

    @property
    def num_hopping_parameters(self):
        ################Atttenzione ############ Vengono contati doppi visto che ho modificato prima
        #return self.num_hoppings_per_layer * self._num_couplings * self._depth
        return self.num_hoppings_per_layer * self._depth
        ################Atttenzione ############
    @property
    def num_phase_parameters(self):
        return self.num_phases_per_layer * self._depth


    @property
    def hopper_correction(self) -> float:
        """ The correction for the norms of the hopping terms to make sure that all hopping strengths can be
        reached within the boundaries of (-np.pi, np.pi)"""
        if self._S == 0:
            return 1.
        else:
            return np.sqrt((self._S + 1) * self._S) * 2

    def _construct_hopping_operators(self, mode='qiskit'):
        """
        Constructs the Hamiltonian inspired hopping operators which are exponentiated when building the circuit and
        stores them in `self._hopping_terms` (for 'qiskit' mode) or `self._matrix_hopping_terms` (for 'qutip' mode).

        Args:
            mode (str): Must be one of ['qiskit', 'qutip']

        Returns:
            None
        """
        if mode == 'qiskit':
            self._hopping_terms = [] # will be a list of shape (nedges * ncouplings)
        elif mode == 'qutip':
            self._matrix_hopping_terms = [] # will be a list of shape (nedges * ncouplings)
        else:
            raise UserWarning("mode must be one of ['qiskit', 'qutip']")

        # 1. Iterate through all lattice edges
        for edge in self._lattice.edges:
            hopping_dim = edge[1]

            # 2. Iterate through all coupling matrices (mixmatrices) along the given dimension of the edge
            for mixmatrix in self._coupling_matrices[hopping_dim]:
                # Build up the hopping term along this edge with the specified coupling
                hopper_edge = hopping_like(edge, lattice=self._lattice, S=self._S, mixmat=mixmatrix, output=mode)
                # Simplify the operator if in qiskit mode.
                if mode == 'qiskit':
                    hopper_edge.chop()
                    self._hopping_terms.append(hopper_edge)
                elif mode == 'qutip':
                    self._matrix_hopping_terms.append(hopper_edge)

    def _construct_parametrized_evolution_circuits(self):
        # TODO
        pass

    def initialize_parameters(self, mode='uniform', hopping_sampler=None, phase_sampler=None):
        """
        Retruns a numpy array of randomly initialized parameters for the given variational form. The parameters
        are drawn from the specified `distribution`
        Args:
            mode (str): The initialization mode. Must be one of ['uniform', 'multimode', 'normal', 'multimode-normal',
                'custom']. If custom mode is used, distribution functions for the `hopping_sampler` and the
                `phase_sampler` must be given.
            hopping_sampler (function): The distribution function for the hopping parameters.
            phase_sampler (function): The distribution function for the single qubit z-rotation parameters at the
                end of a block

        Returns:
            np.ndarray:
                A random numpy array distributed according to the given distributions of size `self.num_parameters`
        """

        if mode == 'uniform':
            return uniform_sampler(self.num_parameters,
                                   lb=-np.pi/(2*self.hopper_correction),
                                   ub=np.pi/(2*self.hopper_correction))

        elif mode == 'multimode':
            return multimode_normal_sampler(self.num_parameters,
                                            modes=[-1,1],
                                            width=np.pi/(8*self.hopper_correction))

        elif mode == 'normal':
            return multimode_normal_sampler(self.num_parameters,
                                            modes=(0),
                                            width=1/self.hopper_correction)

        elif mode =='multimode-uniform':
            hopping_sampler = lambda size: multimode_normal_sampler(size,
                                                                    modes=[-1,1],
                                                                    width=np.pi/(8*self.hopper_correction))
            phase_sampler = lambda size: uniform_sampler(size,
                                                         lb=-np.pi / (2 * self.hopper_correction),
                                                         ub=np.pi / (2 * self.hopper_correction))

        elif mode == 'custom':
            if hopping_sampler is None or phase_sampler is None:
                raise UserWarning("Must provide functions for `hopping_sampler` and `phase_sampler` in 'custom' mode.")

        else:
            raise UserWarning("`mode` must be one of ['uniform', 'normal', 'multimode',"
                              " 'multimode-uniform', 'custom']")

        # Generate the random parameters for the first depth-block
        hoppings1 = hopping_sampler(self.num_hoppings_per_layer * self._num_couplings)
        phases1 = phase_sampler(self.num_phases_per_layer)
        params= np.hstack((hoppings1, phases1))

        # Generate the random parameters for the following blocks.
        for d in range(self._depth-1):
            hoppingsd = hopping_sampler(self.num_hoppings_per_layer * self._num_couplings)
            phasesd = phase_sampler(self.num_phases_per_layer)
            params = np.hstack((params, hoppingsd, phasesd))
        return params

    def construct_circuit(self, parameters, q=None, num_time_slices=1):
        """
        Construct the variational form, given its parameters.
        Args:
            parameters (numpy.ndarray): circuit parameters.
            coupling_matrices (list): list of np.ndarray for the coupling matrices
            q (QuantumRegister): Quantum Register for the circuit.
        Returns:
            QuantumCircuit: a quantum circuit with given `parameters`
        Raises:
            ValueError: the number of parameters is incorrect.
        """

        # 1 Parse inputs
        if len(parameters) != self._num_parameters:
            raise ValueError('The number of parameters has to be {}'.format(self._num_parameters))

        # 2 Set up the quantum register and the circuit
        if q is None:
            if self._initial_state is not None:
                circuit = self._initial_state.construct_circuit('circuit')
                q = circuit.qregs
            else:
                q = QuantumRegister(self._num_qubits, name='q')
                circuit = QuantumCircuit(q)
        else:
            circuit = self._initial_state.construct_circuit('circuit', q)
        all_qubits = circuit.qubits
        fermion_qubits = circuit.qubits[:self._lattice.nsites*2]

        # 3 Build up the variational form with the parameters
        #hoppings_per_layer = self.num_hoppings_per_layer
        # attenzione metto per lattice, perché lui mette 4 invece di 8  
        hoppings_per_layer = self.num_hoppings_per_layer * self._lattice.nedges
        couplings_per_hopping = self._num_couplings
        phases_per_layer = self.num_phases_per_layer
        #### per 2 sites con 2 couplingmatrices, ci sono 4 param 
        #per layer cioé per edges 
        #params_per_layer = couplings_per_hopping * hoppings_per_layer + phases_per_layer
        params_per_layer =  hoppings_per_layer + phases_per_layer
        """
        # 4. Iterate over depth to construct the individual blocks
        if self._delpth_phase == -1 : 
            for d in range(self._depth):
            # 4.1 For each depth-block add parametrized exponentials of hopping terms
            hopping_params = parameters[d * params_per_layer: (d+1) * params_per_layer - phases_per_layer]
            phase_params = parameters[(d+1) * params_per_layer - phases_per_layer: (d+1) * params_per_layer]

            for i, hopper, parameter in zip(np.arange(hoppings_per_layer),
                                            self._hopping_terms,
                                            hopping_params):
                #print(i, "hopp", hopper, parameter)
                var_term = construct_trotter_step(hopper,
                                                  parameter,
                                                  name='h_{}({})'.format(self._lattice.edges[i//couplings_per_hopping],
                                                                         parameter), num_time_slices= num_time_slices)
                # se come init stae metto Dirac qui  if len(qargs) != self.num_qubits, mi da errore  
                # check initial state of varform!!                                                         
                circuit.append(var_term, qargs=all_qubits)

            # 4.3 Add z-rotations for every second qubit at the end of one layer
            #tutti i fermionic qubit, ometto fermion_qubits[1::2]
            for qbit, phase_param in zip(fermion_qubits, phase_params):
                # circuit.u1(phase_param, qbit)
                circuit.rz(phase_param, qbit)
        elif self._delpth_phase != 1 : 
            for d in range(self._depth):
            # 4.1 For each depth-block add parametrized exponentials of hopping terms
            hopping_params = parameters[d * params_per_layer: (d+1) * params_per_layer - phases_per_layer]
            phase_params = parameters[(d+1) * params_per_layer - phases_per_layer: (d+1) * params_per_layer]

            for i, hopper, parameter in zip(np.arange(hoppings_per_layer),
                                            self._hopping_terms,
                                            hopping_params):
                #print(i, "hopp", hopper, parameter)
                var_term = construct_trotter_step(hopper,
                                                  parameter,
                                                  name='h_{}({})'.format(self._lattice.edges[i//couplings_per_hopping],
                                                                         parameter), num_time_slices= num_time_slices)
                # se come init stae metto Dirac qui  if len(qargs) != self.num_qubits, mi da errore  
                # check initial state of varform!!                                                         
                circuit.append(var_term, qargs=all_qubits)
    
            for d in range(self._depth_phase):
                for qbit, phase_param in zip(fermion_qubits, phase_params):
                # circuit.u1(phase_param, qbit)
                circuit.rz(phase_param, qbit)
        """

        """
        old working 
        for d in range(self._depth):
            # 4.1 For each depth-block add parametrized exponentials of hopping terms
            hopping_params = parameters[d * params_per_layer: (d+1) * params_per_layer - phases_per_layer]
            phase_params = parameters[(d+1) * params_per_layer - phases_per_layer: (d+1) * params_per_layer]

            for i, hopper, parameter in zip(np.arange(hoppings_per_layer),
                                            self._hopping_terms,
                                            hopping_params):
                #print(i, "hopp", hopper, parameter)
                var_term = construct_trotter_step(hopper,
                                                  parameter,
                                                  name='h_{}({})'.format(self._lattice.edges[i//couplings_per_hopping],
                                                                         parameter), num_time_slices= num_time_slices)
                # se come init stae metto Dirac qui  if len(qargs) != self.num_qubits, mi da errore  
                # check initial state of varform!!                                                         
                circuit.append(var_term, qargs=all_qubits)

            # 4.3 Add z-rotations for every second qubit at the end of one layer
            #tutti i fermionic qubit, ometto fermion_qubits[1::2]
            if self._depth_phase == -1:
                for qbit, phase_param in zip(fermion_qubits, phase_params):
                # circuit.u1(phase_param, qbit)
                    circuit.rz(phase_param, qbit)
            if self._depth_phase != -1:
                phases_per_layer_phase = int(phases_per_layer / self._depth_phase)
                #print(phases_per_layer_phase , int(phases_per_layer_phase ), phases_per_layer, self._depth_phase  )
                for d in range(self._depth_phase):
                    phase_params_per_d_phase = phase_params[d * phases_per_layer_phase : (d+1) * phases_per_layer_phase ]
                    for qbit, phase_param in zip(fermion_qubits, phase_params_per_d_phase):
                        # circuit.u1(phase_param, qbit)
                        circuit.rz(phase_param, qbit)
        
        return circuit
        """

        for d in range(self._depth):
            # 4.1 For each depth-block add parametrized exponentials of hopping terms
            hopping_params = parameters[d * params_per_layer: (d+1) * params_per_layer - phases_per_layer]
            phase_params = parameters[(d+1) * params_per_layer - phases_per_layer: (d+1) * params_per_layer]
            var_terms = [] 
            for i, hopper, parameter in zip(np.arange(hoppings_per_layer),
                                            self._hopping_terms,
                                            hopping_params):
                #print(i, "hopp", hopper, parameter)
                var_terms.append(construct_trotter_step(hopper,
                                                  parameter,
                                                  name='h_{}({})'.format(self._lattice.edges[i//couplings_per_hopping],
                                                                         parameter), num_time_slices= num_time_slices))
                # se come init stae metto Dirac qui  if len(qargs) != self.num_qubits, mi da errore  
                # check initial state of varform!!                                                         
                #circuit.append(var_term, qargs=all_qubits)

            # 4.3 Add z-rotations for every second qubit at the end of one layer
            #tutti i fermionic qubit, ometto fermion_qubits[1::2]
            if self._depth_phase == -1:
                for var_term in var_terms:
                    if var_term != var_terms[-1]:
                        circuit.append(var_term, qargs=all_qubits)
                    elif var_term == var_terms[-1]:
                        for qbit, phase_param in zip(fermion_qubits, phase_params):
                             circuit.rz(phase_param, qbit)
                        circuit.append(var_term, qargs=all_qubits)

                #for qbit, phase_param in zip(fermion_qubits, phase_params):
                 #   circuit.rz(phase_param, qbit)
                

            if self._depth_phase != -1:
                phases_per_layer_phase = int(phases_per_layer / self._depth_phase)
                #print(phases_per_layer_phase , int(phases_per_layer_phase ), phases_per_layer, self._depth_phase  )

                if self._depth_phase <= hoppings_per_layer:
                    #forward
                    """
                    n_rest =  self._depth_phase
                    for d in range(self._depth_phase):
                        phase_params_per_d_phase = phase_params[d * phases_per_layer_phase : (d+1) * phases_per_layer_phase ]
                        circuit.append(var_terms[d], qargs=all_qubits)
                        for qbit, phase_param in zip(fermion_qubits, phase_params_per_d_phase):
                            circuit.rz(phase_param, qbit)
                    for var_term in var_terms[n_rest:]:
                        circuit.append(var_term, qargs=all_qubits)
                    """  
                    n_rest =  hoppings_per_layer - self._depth_phase
                    for var_term in var_terms[:n_rest]:
                        circuit.append(var_term, qargs=all_qubits)
                    for d in range(self._depth_phase):
                        phase_params_per_d_phase = phase_params[d * phases_per_layer_phase : (d+1) * phases_per_layer_phase ]
                        circuit.append(var_terms[n_rest+d], qargs=all_qubits)
                        
                        for qbit, phase_param in zip(fermion_qubits, phase_params_per_d_phase):
                            circuit.rz(phase_param, qbit)
                    
                    

                if self._depth_phase > hoppings_per_layer:
                    for i,var_term in enumerate(var_terms):
                        phase_params_per_d_phase = phase_params[i * phases_per_layer_phase : (i+1) * phases_per_layer_phase ]
                        circuit.append(var_term, qargs=all_qubits)
                        for qbit, phase_param in zip(fermion_qubits, phase_params_per_d_phase):
                            circuit.rz(phase_param, qbit)
                        
                    for d in np.arange(hoppings_per_layer, self._depth_phase):
                        phase_params_per_d_phase = phase_params[d * phases_per_layer_phase : (d+1) * phases_per_layer_phase ]
                        for qbit, phase_param in zip(fermion_qubits, phase_params_per_d_phase):
                            circuit.rz(phase_param, qbit)


        
        return circuit

    @staticmethod
    def _site_rotation_matrix(site_index, lattice, S, comp=1):
        """
        Perform a z-rotation on the `comp`-th qubit making up a site. This method is meant to be used to simulate
        and verify the qiskit circuit, not for any other purposes.

        Args:
            site_index (int): The index of the site for which we want to rotate one of the components
            lattice (Lattice): The lattice object
            S (float): The spin truncation of the Abelian Quantum Link Model
            comp (int): The component qubit which we want to rotate

        Returns:
            qutip.Qobj: The matrix representation of the pauli string II...IZI...II, where the Z pauli operator is
                acting on the qubit representing the spinor component `comp` at the lattice site indexed by `site_index`
        """
    

        #dim_S = int(2 * S + 1)
        dim_S = int(np.ceil(np.log2(2 * S + 1)))
        ncomp = 2
        assert 0 <= site_index <= lattice.nsites - 1, '`site_index` out of bounds ' \
                                                      'for lattice with {} sites'.format(lattice.nsites)

        ops = [qt.identity(2)] * (ncomp * site_index + comp) \
              + [qt.sigmaz()] \
              + [qt.identity(2)] * (ncomp * lattice.nsites - (ncomp * site_index + comp) - 1)

        

        #qui c'é un errore (oppure no se é pensato in linear, cmq se ho dim_2=2 e faccio qt.identity(2) ho una matrice che prende
        # solo due elementi ma dovrei fare 2**dim_S, perché per ogni qubit ho due var 
        #ops += [qt.identity(dim_S)] * lattice.nedges
        if S > 0:
            ops += [qt.identity(2**dim_S)] * lattice.nedges
        
        return qt.tensor(ops[::-1])

    @staticmethod
    def _edge_rotation_matrix(edge_index, lattice, S):
        """
        Perform a z-rotation on the `edge_index`-th edge. This method is meant to be used to simulate
        and verify the qiskit circuit, not for any other purposes.

        Args:
            edge_index (int): The index of the edge for which we want to perform a z-rotation
            lattice (Lattice): The lattice object
            S (float): The spin truncation of the Abelian Quantum Link Model

        Returns:
            qutip.Qobj:
                The matrix representation of the z-rotation on the `edge_index`-th edge.
        """

        dim_S = int(2*S+1)
        ncomp = 2
        assert 0 <= edge_index <= lattice.nedges - 1, '`edge_index` out of bounds ' \
                                                      'for lattice with {} edges'.format(lattice.nedges)

        ops = [qt.identity(2)] * (ncomp * lattice.nsites) \
              + [qt.identity(dim_S)] * edge_index \
              + [qt.jmat(S, 'z')]  \
              + [qt.identity(dim_S)] * (lattice.nedges - edge_index - 1)

        return qt.tensor(ops[::-1])

    def _eval_matrix_varform(self, parameters, init_state):
        """
        Constructs the variational state in qutip. Used to verify qiskit simulations.

        Args:
            parameters (list or np.ndarray):
            init_state (qutip.Qobj):

        Returns:

        
        """
        if len(parameters) != self._num_parameters:
            raise ValueError('The number of parameters has to be {}'.format(self._num_parameters))

        # 1. If hopping operators are not yet built, build them
        if self._matrix_hopping_terms == []:
            self._construct_hopping_operators(mode='qutip')

        # 2 Initialize the var_form matrix
        var_form = init_state.copy()
        mat_form = qt.identity(2**self.num_qubits)
        # 3. Build up the variational form with the parameters
        #hoppings_per_layer = len(self._matrix_hopping_terms)
        #couplings_per_hopping = self._coupling_matrices.shape[1]
        ####### occhio
        #phases_per_layer = self.num_qubits
        ####occhio che dopo esce [0:0]
        #params_per_layer = couplings_per_hopping * hoppings_per_layer + phases_per_layer
        hoppings_per_layer = self.num_hoppings_per_layer * self._lattice.nedges
        couplings_per_hopping = self._num_couplings
        phases_per_layer = self.num_phases_per_layer
        params_per_layer =  hoppings_per_layer + phases_per_layer
        # 3.1 Iterate over depth
        for d in range(self._depth):
            # 3.2 For each depth-block add parametrized exponentials of hopping terms
            hopping_params = parameters[d * params_per_layer: (d + 1) * params_per_layer - phases_per_layer]
            phase_params = parameters[(d + 1) * params_per_layer - phases_per_layer: (d + 1) * params_per_layer]
            ##### cambio len(hopping_param) con len(self._matrix_hopping_terms)
            for i, hopper, parameter in zip(np.arange(len(self._matrix_hopping_terms)),
                                            self._matrix_hopping_terms,
                                            hopping_params):
                #trasform hopper in the right dims:  dims [16,1,1,1,,2,2,] --> [1024,1024]
                hopper = qt.Qobj(hopper.full())
                var_form = (1j * parameter * hopper).expm() \
                             * var_form           
                mat_form *= (1j * parameter * hopper).expm()

            # 3.3 At the end add single qubit rotations for each site and edge
            for site_index, param in zip(np.arange(self._lattice.nsites), phase_params[:self._lattice.nsites]):
                # perform z-rotations on every 2nd qubit
                rot_matrix_right_dim = qt.Qobj(self._site_rotation_matrix(site_index, self._lattice, self._S).full())
                #occhio metto quel -(param/2) perché cosi é uguale al aprametro usato da rz(param)
                var_form = (1j * -(param/2) * rot_matrix_right_dim).expm() \
                            * var_form
                mat_form *= (1j * -(param/2) * rot_matrix_right_dim).expm()

            # if self._S > 0:
            #     for edge_index, param in zip(np.arange(self._lattice.nedges), phase_params[self._lattice.nsites:]):
            #         # perform z-rotations on every 2nd qubit
            #         var_form = (1j * param * self._edge_rotation_matrix(edge_index, self._lattice, self._S)).expm() \
            #                     * var_form

        return var_form, mat_form


class WilsonLGT6_string(VariationalForm):
    

    def __init__(self, lattice, S, rep=None, coupling_matrices=None, depth=1, initial_state=None, depth_phase=-1):
      
     
        super().__init__()

        # 1 Initialize the varform-circuit parameters
        self._num_qubits = lattice.nsites * 2 + lattice.nedges * int(np.ceil(np.log2(2 * S + 1)))
        self._depth = depth
        self._depth_phase = depth_phase

        # 2 Initialize the var-form lattice parameters
        self._lattice = lattice
        self._S = S
        self._initial_state = initial_state

        # 3 Set the parametrized couplings. Either `rep` or `coupling_matrices` must be given as argument.
        # xor these two arguments:
        assert (rep is not None) ^ (coupling_matrices is not None), \
            'Either `rep` or `coupling_matrices` must be given.'

        # Note: The attribute self._coupling_matrices will be a 4d - numpy array of shape
        # (ndim, ncouplings_per_dim, ncomponents, ncomponents), i.e.
        # self._coupling_matrices[2][3] returns the `3`rd coupling matrix for hopping along
        # lattice dimension `2` as a (ncomponents, ncomponents) numpy array.
        if rep is not None:
            gamma = [val for val in rep.values()]
            self._coupling_matrices = np.array([np.array([-1j * gamma[0] @ gamma[hopping_dim + 1]])
                                                for hopping_dim in range(self._lattice.ndim)])

        elif coupling_matrices is not None:
            self._coupling_matrices = atleast_4d(coupling_matrices)

        # 4 Parse the coupling_matrices:
        # Each dimension should have a list of coupling matrices:
        if not self._coupling_matrices.shape[0] == self._lattice.ndim:
            raise UserWarning('Error with coupling matrices. The given lattice has {} dimensions but only coupling'
                              'matrices were given for {} dimensions. Please check that `coupling_matrices` has the '
                              'shape (lattice.ndim, n_couplings_per_dim, ncomponents, ncomponents)'.format(
                self._lattice.ndim, self._coupling_matrices.shape[0]))
        # And each coupling matrix should be a square matrices of size (ncomp, ncomp). Currently only 2 component
        # spinors are supported.
        if not self._coupling_matrices.shape[2:] == (2, 2):  # TODO: Generalize for more components spinors
            raise UserWarning('Error with coupling matrices. Your coupling matrices couple {} component spinors,'
                              'but currently only 2 component spinors are supported.'.format(
                self._coupling_matrices.shape[2]))

        # 5 Extract the number of parametrized couplings for every dimension
        self._num_couplings = self._coupling_matrices.shape[1]

        # 8 Set up the hopping terms along the edges
        self._construct_hopping_operators()
        self._construct_string_operators()
        self._matrix_hopping_terms = []
        self._matrix_string_terms = []

        # 6 Count the total number of parameters of the variational form:
        # one parameter for every edge (along which hopp. occurs) and every parametrized coupling plus
        # one parameter for final z-rotation at the end of each layer

        if depth_phase == -1: 
            self._num_parameters = (lattice.nedges * self._num_couplings             # hopping terms
                                + lattice.nsites*2) * depth  + self.num_strings_per_layer + lattice.nsites* 2        #  occhio metto una rotazione per fermionqubit+ spinor component z-rotations + last side 
        elif depth_phase != -1: 
            self._num_parameters = (lattice.nedges * self._num_couplings  +lattice.nsites* 2  * depth_phase  ) * depth + self.num_strings_per_layer + lattice.nsites* 2

          
        

        # 7 Set bounds for the parameters
        hopping_bounds = [(-np.pi * self.hopper_correction,
                           np.pi * self.hopper_correction)] * self.num_hoppings_per_layer

        string_bounds = [(-np.pi * self.hopper_correction**2,
                           np.pi * self.hopper_correction**2)] * self.num_strings_per_layer

        phase_bounds = [(-np.pi, np.pi)] * self.num_phases_per_layer
        self._bounds = operator_sum([hopping_bounds + string_bounds + phase_bounds] * self._depth)

    @property
    def num_hoppings_per_layer(self):
        return self._lattice.nedges

    @property
    def num_strings_per_layer(self):
        return len(self._string_terms)

    @property
    def num_phases_per_layer(self):
        if self._depth_phase == -1: 
            return (self._lattice.nsites * 2) 
        elif self._depth_phase != -1:
            return (self._lattice.nsites * 2)  * self._depth_phase 

    @property
    def num_hopping_parameters(self):
        return self.num_hoppings_per_layer * self._num_couplings * self._depth

    @property
    def num_strings_parameters(self):
        return self.num_strings_per_layer 

    @property
    def num_phase_parameters(self):
        return self.num_phases_per_layer * self._depth

    @property
    def hopper_correction(self) -> float:
        """ The correction for the norms of the hopping terms to make sure that all hopping strengths can be
        reached within the boundaries of (-np.pi, np.pi)"""
        if self._S == 0:
            return 1.
        else:
            return np.sqrt((self._S + 1) * self._S) * 2

    def _construct_hopping_operators(self, mode='qiskit'):
        """
        Constructs the Hamiltonian inspired hopping operators which are exponentiated when building the circuit and
        stores them in `self._hopping_terms` (for 'qiskit' mode) or `self._matrix_hopping_terms` (for 'qutip' mode).

        Args:
            mode (str): Must be one of ['qiskit', 'qutip']

        Returns:
            None
        """
        if mode == 'qiskit':
            self._hopping_terms = []  # will be a list of shape (nedges * ncouplings)
        elif mode == 'qutip':
            self._matrix_hopping_terms = []  # will be a list of shape (nedges * ncouplings)
        else:
            raise UserWarning("mode must be one of ['qiskit', 'qutip']")

        # 1. Iterate through all lattice edges
        for edge in self._lattice.edges:
            hopping_dim = edge[1]

            # 2. Iterate through all coupling matrices (mixmatrices) along the given dimension of the edge
            for mixmatrix in self._coupling_matrices[hopping_dim]:
                # Build up the hopping term along this edge with the specified coupling
                hopper_edge = hopping_like(edge, lattice=self._lattice, S=self._S, mixmat=mixmatrix, output=mode)
                # Simplify the operator if in qiskit mode.
                if mode == 'qiskit':
                    hopper_edge.chop()
                    self._hopping_terms.append(hopper_edge)
                elif mode == 'qutip':
                    self._matrix_hopping_terms.append(hopper_edge)

    def _construct_string_operators(self, mode='qiskit'):
        if mode == 'qiskit':
            self._string_terms = []  # will be a list of shape (nedges * ncouplings)
        elif mode == 'qutip':
            self._matrix_string_terms = []  # will be a list of shape (nedges * ncouplings)
        else:
            raise UserWarning("mode must be one of ['qiskit', 'qutip']")

        # 1. Iterate through all pairs of lattice edges
        # list_edge_pairs = get_list_edge_pairs(self._lattice)

        # Warning: Hardcoded version for the specific case of a 2 x 2 lattice
        #if self._lattice.nsites != 3:
        #    raise UserWarning("THE string term is only for 3 sites ")
        #edges_list = []
       
        #edges = [[0,0], [1,0]]
        mixmat = self._coupling_matrices[0][0]
        edges_list= []
        N = self._lattice.nsites
        for m in range(2, N):
            for i in range(N-m):
                v = np.arange(i,i + m )
                el= []
                for v_el in v:
                    el.append([v_el,0])
                edges_list.append(el)
        

        for i in range(len(edges_list)):
            #print(len(edges_list))
            string_edge = string_like(edges_list[i], lattice=self._lattice, S=self._S, mixmat=mixmat, output=mode)
            string_edge.chop()
            self._string_terms.append(string_edge)
      

        
        #mixmat = self._coupling_matrices[0][0] #così dovrebbe prendere solo una matrice 
        #string_edge1 = string_like(edges, lattice=self._lattice, S=self._S, mixmat=mixmat, output=mode)
        #string_edge1.chop()
        
        #self._string_terms.append(string_edge1)
    

    def construct_circuit(self, parameters, q=None, num_time_slices=1):
        """
        Construct the variational form, given its parameters.
        Args:
            parameters (numpy.ndarray): circuit parameters.
            coupling_matrices (list): list of np.ndarray for the coupling matrices
            q (QuantumRegister): Quantum Register for the circuit.
        Returns:
            QuantumCircuit: a quantum circuit with given `parameters`
        Raises:
            ValueError: the number of parameters is incorrect.
        """
        edges_list= []
        N = self._lattice.nsites
        for m in range(2, N):
            for i in range(N-m):
                v = np.arange(i,i + m )
                el= []
                for v_el in v:
                    el.append([v_el,0])
                edges_list.append(el)
        
        # 1 Parse inputs
        if len(parameters) != self._num_parameters:
            raise ValueError('The number of parameters has to be {}'.format(self._num_parameters))

        # 2 Set up the quantum register and the circuit
        if q is None:
            if self._initial_state is not None:
                circuit = self._initial_state.construct_circuit('circuit')
                q = circuit.qregs
            else:
                q = QuantumRegister(self._num_qubits, name='q')
                circuit = QuantumCircuit(q)
        else:
            circuit = self._initial_state.construct_circuit('circuit', q)
        all_qubits = circuit.qubits
        fermion_qubits = circuit.qubits[:self._lattice.nsites * 2]

        # 3 Build up the variational form with the parameters
        hoppings_per_layer = self.num_hoppings_per_layer
        #strings_per_layer = self.num_strings_per_layer
        couplings_per_hopping = self._num_couplings
        phases_per_layer = self.num_phases_per_layer
        #params_per_layer = couplings_per_hopping * hoppings_per_layer + phases_per_layer + strings_per_layer
        params_per_layer = couplings_per_hopping * hoppings_per_layer + phases_per_layer


        #print("\n",parameters, "\n")
        phase_params_last = parameters[-self._lattice.nsites*2:]
        #print("\n",phase_params_last, "\n")
        parameters = parameters[:len(parameters) - (self._lattice.nsites*2)]
        #print("\n", parameters, "\n")
        string_params = parameters[ (self._depth ) * params_per_layer :]

        # 4. Iterate over depth to construct the individual blocks
        for d in range(self._depth):
            
            # 4.1 For each depth-block add parametrized exponentials of hopping terms
            hopping_params = parameters[d * params_per_layer: (d + 1) * params_per_layer
                                                              - (phases_per_layer)]

            #print("hopping_params",hopping_params) 
            phase_params = parameters[(d + 1) * params_per_layer
                                                              - (phases_per_layer): (d + 1) * params_per_layer ]
            #print(phase_params)                                              
            
            
            #phase_params = parameters[(d + 1) * params_per_layer - phases_per_layer: (d + 1) * params_per_layer]
            #print("phase_params",phase_params)
            #print("np.arange(hoppings_per_layer)", np.arange(hoppings_per_layer),"len(self._hopping_terms)", len(self._hopping_terms), "hopping_params ",hopping_params )
            #for i, hopper, parameter in zip(np.arange(hoppings_per_layer), self._hopping_terms, hopping_params):
            var_terms_hop = [] 
            #var_terms_string = []
            for i, hopper, parameter in zip(np.arange(len(self._hopping_terms)), self._hopping_terms, hopping_params):
                #var_term = construct_trotter_step(hopper,parameter,
                                                #name='h_{}({})'.format(
                                                #      self._lattice.edges[i // couplings_per_hopping],
                                                #      parameter), num_time_slices= num_time_slices)
                var_terms_hop.append(construct_trotter_step(hopper,
                                                  parameter,
                                                  name='h_{}({})'.format(self._lattice.edges[i//couplings_per_hopping],
                                                                         parameter), num_time_slices= num_time_slices))
                #circuit.append(var_term, qargs=all_qubits)
            #for i, string_term, parameter in zip(np.arange(len(self._string_terms)),self._string_terms, string_params):
            #    var_terms_string.append(construct_trotter_step(string_term, parameter, name='s_{}({})'.format(i,
            #                                                             parameter),num_time_slices= num_time_slices))
            
            
            if self._depth_phase == -1:
                for var_term in var_terms_hop:
                    circuit.append(var_term, qargs=all_qubits)
                #for string_term_c in var_terms_string:
                #    circuit.append(string_term_c, qargs=all_qubits)
                for qbit, phase_param in zip(fermion_qubits, phase_params):
                            circuit.rz(phase_param, qbit)

            if self._depth_phase != -1:
                phases_per_layer_phase = int(phases_per_layer / self._depth_phase)
                #print(phases_per_layer_phase , int(phases_per_layer_phase ), phases_per_layer, self._depth_phase  )
                #occhio che in GT& é diverso len(varform._hopping_terms)hoppings_per_layer
                if self._depth_phase <= len(self._hopping_terms):
                    #forward
                    """
                    n_rest =  self._depth_phase
                    for d in range(self._depth_phase):
                        phase_params_per_d_phase = phase_params[d * phases_per_layer_phase : (d+1) * phases_per_layer_phase ]
                        circuit.append(var_terms[d], qargs=all_qubits)
                        for qbit, phase_param in zip(fermion_qubits, phase_params_per_d_phase):
                            circuit.rz(phase_param, qbit)
                    for var_term in var_terms[n_rest:]:
                        circuit.append(var_term, qargs=all_qubits)
                    """  
                    n_rest = len(self._hopping_terms) - self._depth_phase
                    #print(n_rest)
                    #print(len(self._hopping_terms) )
                    for var_term in var_terms_hop[:n_rest]:
                        circuit.append(var_term, qargs=all_qubits)

                    for d in range(self._depth_phase):
                        phase_params_per_d_phase = phase_params[d * phases_per_layer_phase : (d+1) * phases_per_layer_phase ]
                        circuit.append(var_terms_hop[n_rest+d], qargs=all_qubits)
                        
                        for qbit, phase_param in zip(fermion_qubits, phase_params_per_d_phase):
                            circuit.rz(phase_param, qbit)

                    
                    #for string_term_c in var_terms_string:
                    #    circuit.append(string_term_c, qargs=all_qubits)
                    

                if self._depth_phase > len(self._hopping_terms):
                    for i,var_term in enumerate(var_terms_hop):
                        phase_params_per_d_phase = phase_params[i * phases_per_layer_phase : (i+1) * phases_per_layer_phase ]
                        circuit.append(var_term, qargs=all_qubits)
                        for qbit, phase_param in zip(fermion_qubits, phase_params_per_d_phase):
                            circuit.rz(phase_param, qbit)
                        
                    for d in np.arange(len(self._hopping_terms), self._depth_phase):
                        phase_params_per_d_phase = phase_params[d * phases_per_layer_phase : (d+1) * phases_per_layer_phase ]
                        for qbit, phase_param in zip(fermion_qubits, phase_params_per_d_phase):
                            circuit.rz(phase_param, qbit)
                    #for string_term_c in var_terms_string:
                    #    circuit.append(string_term_c, qargs=all_qubits)

        for i,string_term, parameter in zip(np.arange(len(self._string_terms)),self._string_terms, string_params):
                var_term = construct_trotter_step(string_term, parameter, num_time_slices=num_time_slices, name='s_[{},{}]({})'.format(edges_list[i][0][0],(edges_list[i][-1][0]+1),
                                                                         parameter))
                circuit.append(var_term, qargs=all_qubits)

        for qbit, phase_param in zip(fermion_qubits, phase_params_last):
                            circuit.rz(phase_param, qbit)
        

        return circuit





class WilsonLGT6_double_string(VariationalForm):
    

    def __init__(self, lattice, S, rep=None, coupling_matrices=None, depth=1, initial_state=None, depth_phase=-1):
      
     
        super().__init__()

        # 1 Initialize the varform-circuit parameters
        self._num_qubits = lattice.nsites * 2 + lattice.nedges * int(np.ceil(np.log2(2 * S + 1)))
        self._depth = depth
        self._depth_phase = depth_phase

        # 2 Initialize the var-form lattice parameters
        self._lattice = lattice
        self._S = S
        self._initial_state = initial_state

        # 3 Set the parametrized couplings. Either `rep` or `coupling_matrices` must be given as argument.
        # xor these two arguments:
        assert (rep is not None) ^ (coupling_matrices is not None), \
            'Either `rep` or `coupling_matrices` must be given.'

        # Note: The attribute self._coupling_matrices will be a 4d - numpy array of shape
        # (ndim, ncouplings_per_dim, ncomponents, ncomponents), i.e.
        # self._coupling_matrices[2][3] returns the `3`rd coupling matrix for hopping along
        # lattice dimension `2` as a (ncomponents, ncomponents) numpy array.
        if rep is not None:
            gamma = [val for val in rep.values()]
            self._coupling_matrices = np.array([np.array([-1j * gamma[0] @ gamma[hopping_dim + 1]])
                                                for hopping_dim in range(self._lattice.ndim)])

        elif coupling_matrices is not None:
            self._coupling_matrices = atleast_4d(coupling_matrices)

        # 4 Parse the coupling_matrices:
        # Each dimension should have a list of coupling matrices:
        if not self._coupling_matrices.shape[0] == self._lattice.ndim:
            raise UserWarning('Error with coupling matrices. The given lattice has {} dimensions but only coupling'
                              'matrices were given for {} dimensions. Please check that `coupling_matrices` has the '
                              'shape (lattice.ndim, n_couplings_per_dim, ncomponents, ncomponents)'.format(
                self._lattice.ndim, self._coupling_matrices.shape[0]))
        # And each coupling matrix should be a square matrices of size (ncomp, ncomp). Currently only 2 component
        # spinors are supported.
        if not self._coupling_matrices.shape[2:] == (2, 2):  # TODO: Generalize for more components spinors
            raise UserWarning('Error with coupling matrices. Your coupling matrices couple {} component spinors,'
                              'but currently only 2 component spinors are supported.'.format(
                self._coupling_matrices.shape[2]))

        # 5 Extract the number of parametrized couplings for every dimension
        self._num_couplings = self._coupling_matrices.shape[1]

        # 8 Set up the hopping terms along the edges
        self._construct_hopping_operators()
        self._construct_string_operators()
        self._matrix_hopping_terms = []
        self._matrix_string_terms = []

        # 6 Count the total number of parameters of the variational form:
        # one parameter for every edge (along which hopp. occurs) and every parametrized coupling plus
        # one parameter for final z-rotation at the end of each layer

        if depth_phase == -1: 
            self._num_parameters = (lattice.nedges * self._num_couplings             # hopping terms
                                + lattice.nsites*2 + self.num_strings_per_layer ) * depth        #  occhio metto una rotazione per fermionqubit+ spinor component z-rotations + last side 
        elif depth_phase != -1: 
            self._num_parameters = (lattice.nedges * self._num_couplings  + lattice.nsites * 2  * depth_phase + self.num_strings_per_layer + lattice.nsites * 2  ) * depth 

          
        # 7 Set bounds for the parameters
        hopping_bounds = [(-np.pi * self.hopper_correction,
                           np.pi * self.hopper_correction)] * self.num_hoppings_per_layer

        string_bounds = [(-np.pi * self.hopper_correction**2,
                           np.pi * self.hopper_correction**2)] * self.num_strings_per_layer

        phase_bounds = [(-np.pi, np.pi)] * self.num_phases_per_layer
        self._bounds = operator_sum([hopping_bounds + string_bounds + phase_bounds] * self._depth)

    @property
    def num_hoppings_per_layer(self):
        return self._lattice.nedges

    @property
    def num_strings_per_layer(self):
        return len(self._string_terms)

    @property
    def num_phases_per_layer(self):
        if self._depth_phase == -1: 
            return (self._lattice.nsites * 2 ) 
        elif self._depth_phase != -1:
            return (self._lattice.nsites * 2 )  * self._depth_phase + self._lattice.nsites * 2 

    @property
    def num_hopping_parameters(self):
        return self.num_hoppings_per_layer * self._num_couplings * self._depth

    @property
    def num_strings_parameters(self):
        return self.num_strings_per_layer * self._depth

    @property
    def num_phase_parameters(self):
        return self.num_phases_per_layer * self._depth

    @property
    def hopper_correction(self) -> float:
        """ The correction for the norms of the hopping terms to make sure that all hopping strengths can be
        reached within the boundaries of (-np.pi, np.pi)"""
        if self._S == 0:
            return 1.
        else:
            return np.sqrt((self._S + 1) * self._S) * 2

    def _construct_hopping_operators(self, mode='qiskit'):
        """
        Constructs the Hamiltonian inspired hopping operators which are exponentiated when building the circuit and
        stores them in `self._hopping_terms` (for 'qiskit' mode) or `self._matrix_hopping_terms` (for 'qutip' mode).

        Args:
            mode (str): Must be one of ['qiskit', 'qutip']

        Returns:
            None
        """
        if mode == 'qiskit':
            self._hopping_terms = []  # will be a list of shape (nedges * ncouplings)
        elif mode == 'qutip':
            self._matrix_hopping_terms = []  # will be a list of shape (nedges * ncouplings)
        else:
            raise UserWarning("mode must be one of ['qiskit', 'qutip']")

        # 1. Iterate through all lattice edges
        for edge in self._lattice.edges:
            hopping_dim = edge[1]

            # 2. Iterate through all coupling matrices (mixmatrices) along the given dimension of the edge
            for mixmatrix in self._coupling_matrices[hopping_dim]:
                # Build up the hopping term along this edge with the specified coupling
                hopper_edge = hopping_like(edge, lattice=self._lattice, S=self._S, mixmat=mixmatrix, output=mode)
                # Simplify the operator if in qiskit mode.
                if mode == 'qiskit':
                    hopper_edge.chop()
                    self._hopping_terms.append(hopper_edge)
                elif mode == 'qutip':
                    self._matrix_hopping_terms.append(hopper_edge)

    def _construct_string_operators(self, mode='qiskit'):
        if mode == 'qiskit':
            self._string_terms = []  # will be a list of shape (nedges * ncouplings)
        elif mode == 'qutip':
            self._matrix_string_terms = []  # will be a list of shape (nedges * ncouplings)
        else:
            raise UserWarning("mode must be one of ['qiskit', 'qutip']")

        # 1. Iterate through all pairs of lattice edges
        # list_edge_pairs = get_list_edge_pairs(self._lattice)

        # Warning: Hardcoded version for the specific case of a 2 x 2 lattice
        #if self._lattice.nsites != 3:
        #    raise UserWarning("THE string term is only for 3 sites ")
        #edges_list = []
       
        #edges = [[0,0], [1,0]]
        mixmat = self._coupling_matrices[0][0]
        edges_list= []
        N = self._lattice.nsites
        for m in range(2, N):
            for i in range(N-m):
                v = np.arange(i,i + m )
                el= []
                for v_el in v:
                    el.append([v_el,0])
                edges_list.append(el)
        

        for i in range(len(edges_list)):
            #print(len(edges_list))
            string_edge = string_like(edges_list[i], lattice=self._lattice, S=self._S, mixmat=mixmat, output=mode)
            string_edge.chop()
            self._string_terms.append(string_edge)
      

        
        #mixmat = self._coupling_matrices[0][0] #così dovrebbe prendere solo una matrice 
        #string_edge1 = string_like(edges, lattice=self._lattice, S=self._S, mixmat=mixmat, output=mode)
        #string_edge1.chop()
        
        #self._string_terms.append(string_edge1)
    

    def construct_circuit(self, parameters, q=None, num_time_slices=1):
        """
        Construct the variational form, given its parameters.
        Args:
            parameters (numpy.ndarray): circuit parameters.
            coupling_matrices (list): list of np.ndarray for the coupling matrices
            q (QuantumRegister): Quantum Register for the circuit.
        Returns:
            QuantumCircuit: a quantum circuit with given `parameters`
        Raises:
            ValueError: the number of parameters is incorrect.
        """
        edges_list= []
        N = self._lattice.nsites
        for m in range(2, N):
            for i in range(N-m):
                v = np.arange(i,i + m )
                el= []
                for v_el in v:
                    el.append([v_el,0])
                edges_list.append(el)
        
        # 1 Parse inputs
        if len(parameters) != self._num_parameters:
            raise ValueError('The number of parameters has to be {}'.format(self._num_parameters))

        # 2 Set up the quantum register and the circuit
        if q is None:
            if self._initial_state is not None:
                circuit = self._initial_state.construct_circuit('circuit')
                q = circuit.qregs
            else:
                q = QuantumRegister(self._num_qubits, name='q')
                circuit = QuantumCircuit(q)
        else:
            circuit = self._initial_state.construct_circuit('circuit', q)
        all_qubits = circuit.qubits
        fermion_qubits = circuit.qubits[:self._lattice.nsites * 2]

        # 3 Build up the variational form with the parameters
        hoppings_per_layer = self.num_hoppings_per_layer
        strings_per_layer = self.num_strings_per_layer
        couplings_per_hopping = self._num_couplings
        phases_per_layer = self.num_phases_per_layer
        print
        params_per_layer = couplings_per_hopping * hoppings_per_layer + phases_per_layer + strings_per_layer

        #params_per_layer = couplings_per_hopping * hoppings_per_layer + phases_per_layer

        # 4. Iterate over depth to construct the individual blocks
        for d in range(self._depth):
            
            hopping_params = parameters[d * params_per_layer: (d + 1) * params_per_layer
                                                              - (phases_per_layer + strings_per_layer)]
            #print("hopping_params",hopping_params) 
            string_params = parameters[(d + 1) * params_per_layer - (phases_per_layer + strings_per_layer):
                                   (d + 1) * params_per_layer - phases_per_layer]
            #print("string_params",string_params) 
            phase_params = parameters[(d + 1) * params_per_layer - phases_per_layer: (d + 1) * params_per_layer]
            # 4.1 For each depth-block add parametrized exponentials of hopping terms
    
                                                    
            
            
            #phase_params = parameters[(d + 1) * params_per_layer - phases_per_layer: (d + 1) * params_per_layer]
            #print("phase_params",phase_params)
            #print("np.arange(hoppings_per_layer)", np.arange(hoppings_per_layer),"len(self._hopping_terms)", len(self._hopping_terms), "hopping_params ",hopping_params )
            #for i, hopper, parameter in zip(np.arange(hoppings_per_layer), self._hopping_terms, hopping_params):
            var_terms_hop = [] 
            var_terms_string = []
            for i, hopper, parameter in zip(np.arange(len(self._hopping_terms)), self._hopping_terms, hopping_params):
                #var_term = construct_trotter_step(hopper,parameter,
                                                #name='h_{}({})'.format(
                                                #      self._lattice.edges[i // couplings_per_hopping],
                                                #      parameter), num_time_slices= num_time_slices)
                var_terms_hop.append(construct_trotter_step(hopper,
                                                  parameter,
                                                  name='h_{}({})'.format(self._lattice.edges[i//couplings_per_hopping],
                                                                         parameter), num_time_slices= num_time_slices))
                #circuit.append(var_term, qargs=all_qubits)
            for i, string_term, parameter in zip(np.arange(len(self._string_terms)),self._string_terms, string_params):
                var_terms_string.append(construct_trotter_step(string_term, parameter, name='s_[{},{}]({})'.format(edges_list[i][0][0],(edges_list[i][-1][0]+1),parameter),num_time_slices= num_time_slices))
            

            if self._depth_phase == -1 :
                for var_term in var_terms_hop:
                    circuit.append(var_term, qargs=all_qubits)
                for string_term_c in var_terms_string:
                    circuit.append(string_term_c, qargs=all_qubits)
                for qbit, phase_param in zip(fermion_qubits, phase_params):
                            circuit.rz(phase_param, qbit)

            if self._depth_phase != -1:
                phases_per_layer_phase = self._lattice.nsites * 2 
                #print(phases_per_layer_phase , int(phases_per_layer_phase ), phases_per_layer, self._depth_phase  )
                #occhio che in GT& é diverso len(varform._hopping_terms)hoppings_per_layer
                if self._depth_phase <= len(self._hopping_terms):
                    #forward
                    """
                    n_rest =  self._depth_phase
                    for d in range(self._depth_phase):
                        phase_params_per_d_phase = phase_params[d * phases_per_layer_phase : (d+1) * phases_per_layer_phase ]
                        circuit.append(var_terms[d], qargs=all_qubits)
                        for qbit, phase_param in zip(fermion_qubits, phase_params_per_d_phase):
                            circuit.rz(phase_param, qbit)
                    for var_term in var_terms[n_rest:]:
                        circuit.append(var_term, qargs=all_qubits)
                    """  
                    n_rest = len(self._hopping_terms) - self._depth_phase 
                    #print(n_rest)
                    #print(len(self._hopping_terms) )
                    for var_term in var_terms_hop[:n_rest]:
                        circuit.append(var_term, qargs=all_qubits)

                    for d in range(self._depth_phase):
                        phase_params_per_d_phase = phase_params[d * phases_per_layer_phase : (d+1) * phases_per_layer_phase]

                        circuit.append(var_terms_hop[n_rest+d], qargs=all_qubits)
                        
                        for qbit, phase_param in zip(fermion_qubits, phase_params_per_d_phase):
                            circuit.rz(phase_param, qbit)

                    for string_term_c in var_terms_string:
                            circuit.append(string_term_c, qargs=all_qubits)
                            
                    phase_params_per_d_phase_last = phase_params[ len(phase_params) - self._lattice.nsites *2 :]
                    for qbit, phase_param in zip(fermion_qubits, phase_params_per_d_phase_last):
                            circuit.rz(phase_param, qbit)


                    
                    #for string_term_c in var_terms_string:
                    #    circuit.append(string_term_c, qargs=all_qubits)
                    

                if self._depth_phase > len(self._hopping_terms):
                    for i,var_term in enumerate(var_terms_hop):
                        phase_params_per_d_phase = phase_params[i * phases_per_layer_phase : (i+1) * phases_per_layer_phase ]
                        circuit.append(var_term, qargs=all_qubits)
                        for qbit, phase_param in zip(fermion_qubits, phase_params_per_d_phase):
                            circuit.rz(phase_param, qbit)
                        
                    for d in np.arange(len(self._hopping_terms), self._depth_phase):
                        phase_params_per_d_phase = phase_params[d * phases_per_layer_phase : (d+1) * phases_per_layer_phase ]
                        for qbit, phase_param in zip(fermion_qubits, phase_params_per_d_phase):
                            circuit.rz(phase_param, qbit)
                    #for string_term_c in var_terms_string:
                    #    circuit.append(string_term_c, qargs=all_qubits)

    

        
        

        return circuit


class WilsonLGT6_double_string_pair(VariationalForm):
    def __init__(self, lattice, S, rep=None, coupling_matrices=None, depth=1, initial_state=None, depth_phase=-1, depth_pair=1):
      
     
        super().__init__()

        # 1 Initialize the varform-circuit parameters
        self._num_qubits = lattice.nsites * 2 + lattice.nedges * int(np.ceil(np.log2(2 * S + 1)))
        self._depth = depth
        self._depth_phase = depth_phase
        self._depth_pair = depth_pair

        # 2 Initialize the var-form lattice parameters
        self._lattice = lattice
        self._S = S
        self._initial_state = initial_state

        # 3 Set the parametrized couplings. Either `rep` or `coupling_matrices` must be given as argument.
        # xor these two arguments:
        assert (rep is not None) ^ (coupling_matrices is not None), \
            'Either `rep` or `coupling_matrices` must be given.'

        # Note: The attribute self._coupling_matrices will be a 4d - numpy array of shape
        # (ndim, ncouplings_per_dim, ncomponents, ncomponents), i.e.
        # self._coupling_matrices[2][3] returns the `3`rd coupling matrix for hopping along
        # lattice dimension `2` as a (ncomponents, ncomponents) numpy array.
        if rep is not None:
            gamma = [val for val in rep.values()]
            self._coupling_matrices = np.array([np.array([-1j * gamma[0] @ gamma[hopping_dim + 1]])
                                                for hopping_dim in range(self._lattice.ndim)])

        elif coupling_matrices is not None:
            self._coupling_matrices = atleast_4d(coupling_matrices)

        # 4 Parse the coupling_matrices:
        # Each dimension should have a list of coupling matrices:
        if not self._coupling_matrices.shape[0] == self._lattice.ndim:
            raise UserWarning('Error with coupling matrices. The given lattice has {} dimensions but only coupling'
                              'matrices were given for {} dimensions. Please check that `coupling_matrices` has the '
                              'shape (lattice.ndim, n_couplings_per_dim, ncomponents, ncomponents)'.format(
                self._lattice.ndim, self._coupling_matrices.shape[0]))
        # And each coupling matrix should be a square matrices of size (ncomp, ncomp). Currently only 2 component
        # spinors are supported.
        if not self._coupling_matrices.shape[2:] == (2, 2):  # TODO: Generalize for more components spinors
            raise UserWarning('Error with coupling matrices. Your coupling matrices couple {} component spinors,'
                              'but currently only 2 component spinors are supported.'.format(
                self._coupling_matrices.shape[2]))

        # 5 Extract the number of parametrized couplings for every dimension
        self._num_couplings = self._coupling_matrices.shape[1]

        # 8 Set up the hopping terms along the edges
        self._construct_hopping_operators()
        self._construct_string_operators()
        self._construct_pair_operators()
        self._matrix_hopping_terms = []
        self._matrix_string_terms = []
        self._matrix_pair_terms = []

        # 6 Count the total number of parameters of the variational form:
        # one parameter for every edge (along which hopp. occurs) and every parametrized coupling plus
        # one parameter for final z-rotation at the end of each layer

        if depth_phase == -1: 
            self._num_parameters = (lattice.nedges * self._num_couplings             # hopping terms
                                + lattice.nsites*2 + self.num_strings_per_layer + lattice.nsites * self._depth_pair ) * depth        #  occhio metto una rotazione per fermionqubit+ spinor component z-rotations + last side 
        elif depth_phase != -1: 
            self._num_parameters = (lattice.nedges * self._num_couplings  + lattice.nsites * 2  * depth_phase + self.num_strings_per_layer + lattice.nsites * 2 + lattice.nsites * self._depth_pair ) * depth 

          
        # 7 Set bounds for the parameters
        hopping_bounds = [(-np.pi * self.hopper_correction,
                           np.pi * self.hopper_correction)] * self.num_hoppings_per_layer

        string_bounds = [(-np.pi * self.hopper_correction**2,
                           np.pi * self.hopper_correction**2)] * self.num_strings_per_layer

        phase_bounds = [(-np.pi, np.pi)] * self.num_phases_per_layer
        self._bounds = operator_sum([hopping_bounds + string_bounds + phase_bounds] * self._depth)

    @property
    def num_hoppings_per_layer(self):
        return self._lattice.nedges

    @property
    def num_strings_per_layer(self):
        return len(self._string_terms)

    @property
    def num_phases_per_layer(self):
        if self._depth_phase == -1: 
            return (self._lattice.nsites * 2 ) 
        elif self._depth_phase != -1:
            return (self._lattice.nsites * 2 )  * self._depth_phase + self._lattice.nsites * 2 
    
    @property
    def num_pair_per_layer(self):
        return self._lattice.nsites * self._depth_pair

    @property
    def num_hopping_parameters(self):
        return self.num_hoppings_per_layer * self._num_couplings * self._depth

    @property
    def num_strings_parameters(self):
        return self.num_strings_per_layer * self._depth

    @property
    def num_pair_parameters(self):
        return self.num_pair_per_layer * self._depth

    @property
    def num_phase_parameters(self):
        return self.num_phases_per_layer * self._depth

    @property
    def hopper_correction(self) -> float:
        """ The correction for the norms of the hopping terms to make sure that all hopping strengths can be
        reached within the boundaries of (-np.pi, np.pi)"""
        if self._S == 0:
            return 1.
        else:
            return np.sqrt((self._S + 1) * self._S) * 2

    def _construct_hopping_operators(self, mode='qiskit'):
        """
        Constructs the Hamiltonian inspired hopping operators which are exponentiated when building the circuit and
        stores them in `self._hopping_terms` (for 'qiskit' mode) or `self._matrix_hopping_terms` (for 'qutip' mode).

        Args:
            mode (str): Must be one of ['qiskit', 'qutip']

        Returns:
            None
        """
        if mode == 'qiskit':
            self._hopping_terms = []  # will be a list of shape (nedges * ncouplings)
        elif mode == 'qutip':
            self._matrix_hopping_terms = []  # will be a list of shape (nedges * ncouplings)
        else:
            raise UserWarning("mode must be one of ['qiskit', 'qutip']")

        # 1. Iterate through all lattice edges
        for edge in self._lattice.edges:
            hopping_dim = edge[1]

            # 2. Iterate through all coupling matrices (mixmatrices) along the given dimension of the edge
            for mixmatrix in self._coupling_matrices[hopping_dim]:
                # Build up the hopping term along this edge with the specified coupling
                hopper_edge = hopping_like(edge, lattice=self._lattice, S=self._S, mixmat=mixmatrix, output=mode)
                # Simplify the operator if in qiskit mode.
                if mode == 'qiskit':
                    hopper_edge.chop()
                    self._hopping_terms.append(hopper_edge)
                elif mode == 'qutip':
                    self._matrix_hopping_terms.append(hopper_edge)

    def _construct_string_operators(self, mode='qiskit'):
        if mode == 'qiskit':
            self._string_terms = []  # will be a list of shape (nedges * ncouplings)
        elif mode == 'qutip':
            self._matrix_string_terms = []  # will be a list of shape (nedges * ncouplings)
        else:
            raise UserWarning("mode must be one of ['qiskit', 'qutip']")

        # 1. Iterate through all pairs of lattice edges
        # list_edge_pairs = get_list_edge_pairs(self._lattice)

        # Warning: Hardcoded version for the specific case of a 2 x 2 lattice
        #if self._lattice.nsites != 3:
        #    raise UserWarning("THE string term is only for 3 sites ")
        #edges_list = []
       
        #edges = [[0,0], [1,0]]
        mixmat = self._coupling_matrices[0][0]
        edges_list= []
        N = self._lattice.nsites
        for m in range(2, N):
            for i in range(N-m):
                v = np.arange(i,i + m )
                el= []
                for v_el in v:
                    el.append([v_el,0])
                edges_list.append(el)
        

        for i in range(len(edges_list)):
            #print(len(edges_list))
            string_edge = string_like(edges_list[i], lattice=self._lattice, S=self._S, mixmat=mixmat, output=mode)
            string_edge.chop()
            self._string_terms.append(string_edge)
      

        
        #mixmat = self._coupling_matrices[0][0] #così dovrebbe prendere solo una matrice 
        #string_edge1 = string_like(edges, lattice=self._lattice, S=self._S, mixmat=mixmat, output=mode)
        #string_edge1.chop()
        
        #self._string_terms.append(string_edge1)

    def _construct_pair_operators(self, mode='qiskit'):
       
        if mode == 'qiskit':
            self._pair_terms = []  # will be a list of shape (nedges * ncouplings)
        elif mode == 'qutip':
            self._matrix_pair_terms = []  # will be a list of shape (nedges * ncouplings)
        else:
            raise UserWarning("mode must be one of ['qiskit', 'qutip']")

        for site in self._lattice.sites:
            pair_site = pair_like(site=site, lattice=self._lattice, S=self._S)
            if mode == 'qiskit':
                    pair_site.chop()
                    self._pair_terms.append(pair_site)
    

    def construct_circuit(self, parameters, q=None, num_time_slices=1):
        """
        Construct the variational form, given its parameters.
        Args:
            parameters (numpy.ndarray): circuit parameters.
            coupling_matrices (list): list of np.ndarray for the coupling matrices
            q (QuantumRegister): Quantum Register for the circuit.
        Returns:
            QuantumCircuit: a quantum circuit with given `parameters`
        Raises:
            ValueError: the number of parameters is incorrect.
        """
        edges_list= []
        N = self._lattice.nsites
        for m in range(2, N):
            for i in range(N-m):
                v = np.arange(i,i + m )
                el= []
                for v_el in v:
                    el.append([v_el,0])
                edges_list.append(el)
        
        # 1 Parse inputs
        if len(parameters) != self._num_parameters:
            raise ValueError('The number of parameters has to be {}'.format(self._num_parameters))

        # 2 Set up the quantum register and the circuit
        if q is None:
            if self._initial_state is not None:
                circuit = self._initial_state.construct_circuit('circuit')
                q = circuit.qregs
            else:
                q = QuantumRegister(self._num_qubits, name='q')
                circuit = QuantumCircuit(q)
        else:
            circuit = self._initial_state.construct_circuit('circuit', q)
        all_qubits = circuit.qubits
        fermion_qubits = circuit.qubits[:self._lattice.nsites * 2]

        # 3 Build up the variational form with the parameters
        hoppings_per_layer = self.num_hoppings_per_layer
        strings_per_layer = self.num_strings_per_layer
        couplings_per_hopping = self._num_couplings
        phases_per_layer = self.num_phases_per_layer
        pair_per_layer = self.num_pair_per_layer
        
        params_per_layer = couplings_per_hopping * hoppings_per_layer + phases_per_layer + strings_per_layer + pair_per_layer
        
        #params_per_layer = couplings_per_hopping * hoppings_per_layer + phases_per_layer

        # 4. Iterate over depth to construct the individual blocks
        for d in range(self._depth):
            
            hopping_params = parameters[d * params_per_layer: (d + 1) * params_per_layer
                                                              - (phases_per_layer + strings_per_layer+ pair_per_layer)]
            #print("hopping_params",hopping_params) 
            string_params = parameters[(d + 1) * params_per_layer - (phases_per_layer + strings_per_layer+ pair_per_layer):
                                   (d + 1) * params_per_layer - (phases_per_layer+pair_per_layer) ]
            #print("string_params",string_params) 
            phase_params = parameters[(d + 1) * params_per_layer - (phases_per_layer+pair_per_layer): (d + 1) * params_per_layer- pair_per_layer]
            # 4.1 For each depth-block add parametrized exponentials of hopping terms
            pair_params = parameters[(d + 1) * params_per_layer - pair_per_layer: (d + 1) * params_per_layer]
    
                                                    
            
            
            #phase_params = parameters[(d + 1) * params_per_layer - phases_per_layer: (d + 1) * params_per_layer]
            #print("phase_params",phase_params)
            #print("np.arange(hoppings_per_layer)", np.arange(hoppings_per_layer),"len(self._hopping_terms)", len(self._hopping_terms), "hopping_params ",hopping_params )
            #for i, hopper, parameter in zip(np.arange(hoppings_per_layer), self._hopping_terms, hopping_params):
            var_terms_hop = [] 
            var_terms_string = []
            var_terms_pair = []
            for i, hopper, parameter in zip(np.arange(len(self._hopping_terms)), self._hopping_terms, hopping_params):
                #var_term = construct_trotter_step(hopper,parameter,
                                                #name='h_{}({})'.format(
                                                #      self._lattice.edges[i // couplings_per_hopping],
                                                #      parameter), num_time_slices= num_time_slices)
                var_terms_hop.append(construct_trotter_step(hopper,
                                                  parameter,
                                                  name='h_{}({})'.format(self._lattice.edges[i//couplings_per_hopping],
                                                                         parameter), num_time_slices= num_time_slices))
                #circuit.append(var_term, qargs=all_qubits)
            for i, string_term, parameter in zip(np.arange(len(self._string_terms)),self._string_terms, string_params):
                var_terms_string.append(construct_trotter_step(string_term, parameter, name='s_[{},{}]({})'.format(edges_list[i][0][0],(edges_list[i][-1][0]+1),parameter),num_time_slices= num_time_slices))
            
            for d_pair in range(self._depth_pair):
                pair_params_sublayer = pair_params[d_pair * self._lattice.nsites :(d_pair+1) * self._lattice.nsites ]
                for i, pair_term, parameter in zip(np.arange(len(self._pair_terms)),self._pair_terms, pair_params_sublayer  ):
                    var_terms_pair.append(construct_trotter_step(pair_term, parameter, name='pair_[{}]({})'.format(i,parameter),num_time_slices= num_time_slices))

            if self._depth_phase == -1 :
                for var_term in var_terms_hop:
                    circuit.append(var_term, qargs=all_qubits)
                for string_term_c in var_terms_string:
                    circuit.append(string_term_c, qargs=all_qubits)

                for pair_term_c in var_terms_pair:
                    circuit.append(pair_term_c, qargs=all_qubits)

                for qbit, phase_param in zip(fermion_qubits, phase_params):
                            circuit.rz(phase_param, qbit)

            if self._depth_phase != -1:
                phases_per_layer_phase = self._lattice.nsites * 2 
                #print(phases_per_layer_phase , int(phases_per_layer_phase ), phases_per_layer, self._depth_phase  )
                #occhio che in GT& é diverso len(varform._hopping_terms)hoppings_per_layer
                if self._depth_phase <= len(self._hopping_terms):
                    #forward
                    """
                    n_rest =  self._depth_phase
                    for d in range(self._depth_phase):
                        phase_params_per_d_phase = phase_params[d * phases_per_layer_phase : (d+1) * phases_per_layer_phase ]
                        circuit.append(var_terms[d], qargs=all_qubits)
                        for qbit, phase_param in zip(fermion_qubits, phase_params_per_d_phase):
                            circuit.rz(phase_param, qbit)
                    for var_term in var_terms[n_rest:]:
                        circuit.append(var_term, qargs=all_qubits)
                    """  
                    n_rest = len(self._hopping_terms) - self._depth_phase 
                    #print(n_rest)
                    #print(len(self._hopping_terms) )
                    for var_term in var_terms_hop[:n_rest]:
                        circuit.append(var_term, qargs=all_qubits)

                    for d in range(self._depth_phase):
                        phase_params_per_d_phase = phase_params[d * phases_per_layer_phase : (d+1) * phases_per_layer_phase]

                        circuit.append(var_terms_hop[n_rest+d], qargs=all_qubits)
                        
                        for qbit, phase_param in zip(fermion_qubits, phase_params_per_d_phase):
                            circuit.rz(phase_param, qbit)

                    for string_term_c in var_terms_string:
                            circuit.append(string_term_c, qargs=all_qubits)

                    for pair_term_c in var_terms_pair:
                        circuit.append(pair_term_c, qargs=all_qubits)

                    phase_params_per_d_phase_last = phase_params[ len(phase_params) - self._lattice.nsites *2 :]
                    for qbit, phase_param in zip(fermion_qubits, phase_params_per_d_phase_last):
                            circuit.rz(phase_param, qbit)


                    
                    #for string_term_c in var_terms_string:
                    #    circuit.append(string_term_c, qargs=all_qubits)
                    

                if self._depth_phase > len(self._hopping_terms):
                    n_rest = self._depth_phase - len(self._hopping_terms) 
                    n_rest_string = n_rest - len(string_params)


                    for d in range(len(self._hopping_terms)):
                        phase_params_per_d_phase = phase_params[d * phases_per_layer_phase : (d+1) * phases_per_layer_phase]

                        circuit.append(var_terms_hop[d], qargs=all_qubits)
                        
                        for qbit, phase_param in zip(fermion_qubits, phase_params_per_d_phase):
                            circuit.rz(phase_param, qbit)

                    if n_rest_string <= 0 : 
                        for d in range(n_rest):
                            phase_params_per_d_phase = phase_params[(d+ len(self._hopping_terms)) * phases_per_layer_phase : (d+1 + len(self._hopping_terms)) * phases_per_layer_phase]
                            circuit.append(var_terms_string[d], qargs=all_qubits)

                            for qbit, phase_param in zip(fermion_qubits, phase_params_per_d_phase):
                                circuit.rz(phase_param, qbit)
                        for string_term_c in var_terms_string[n_rest:]:
                            circuit.append(string_term_c, qargs=all_qubits)

                        for pair_term_c in var_terms_pair:
                            circuit.append(pair_term_c, qargs=all_qubits)

                        phase_params_per_d_phase_last = phase_params[ len(phase_params) - self._lattice.nsites *2 :]
                        for qbit, phase_param in zip(fermion_qubits, phase_params_per_d_phase_last):
                            circuit.rz(phase_param, qbit)

                    if n_rest_string > 0 :  
                        for d in range(len(self._string_terms)):
                            phase_params_per_d_phase = phase_params[(d+ len(self._hopping_terms)) * phases_per_layer_phase : (d+1 + len(self._hopping_terms)) * phases_per_layer_phase]
                            circuit.append(var_terms_string[d], qargs=all_qubits)

                            for qbit, phase_param in zip(fermion_qubits, phase_params_per_d_phase):
                                circuit.rz(phase_param, qbit) 
                        if n_rest_string > len(self._pair_terms ) :
                            print(" Error , phase_depth has to be smaller ")   
                        if n_rest_string == len(self._pair_terms ):
                            
                            for d in range(len(self._pair_terms)):
                                phase_params_per_d_phase = phase_params[(d+ len(self._hopping_terms) + len(self._string_terms)) * phases_per_layer_phase : (d+1 +len(self._hopping_terms) + len(self._string_terms)) * phases_per_layer_phase]
                                circuit.append(var_terms_pair[d], qargs=all_qubits)
                                for qbit, phase_param in zip(fermion_qubits, phase_params_per_d_phase):
                                    circuit.rz(phase_param, qbit)
                            phase_params_per_d_phase_last = phase_params[ len(phase_params) - self._lattice.nsites *2 :]
                            for qbit, phase_param in zip(fermion_qubits, phase_params_per_d_phase_last):
                                circuit.rz(phase_param, qbit)

                        if n_rest_string < len(self._pair_terms ):
                            for d in range(n_rest_string):
                                phase_params_per_d_phase = phase_params[(d+ len(self._hopping_terms) + len(self._string_terms)) * phases_per_layer_phase : (d+1 +len(self._hopping_terms) + len(self._string_terms)) * phases_per_layer_phase]
                                circuit.append(var_terms_pair[d], qargs=all_qubits)
                                for qbit, phase_param in zip(fermion_qubits, phase_params_per_d_phase):
                                    circuit.rz(phase_param, qbit)

                            for pair_term_c in var_terms_pair[n_rest_string:]:
                                circuit.append(pair_term_c, qargs=all_qubits)
                            phase_params_per_d_phase_last = phase_params[ len(phase_params) - self._lattice.nsites *2 :]
                            for qbit, phase_param in zip(fermion_qubits, phase_params_per_d_phase_last):
                                circuit.rz(phase_param, qbit)
        return circuit
# flux p -> -> a 
class WilsonLGT6_double_string_pair_S_1_5(VariationalForm):
    def __init__(self, lattice, S, rep=None, coupling_matrices=None, depth=1, initial_state=None, depth_phase=-1, depth_pair=1):
      
     
        super().__init__()

        # 1 Initialize the varform-circuit parameters
        self._num_qubits = lattice.nsites * 2 + lattice.nedges * int(np.ceil(np.log2(2 * S + 1)))
        self._depth = depth
        self._depth_phase = depth_phase
        self._depth_pair = depth_pair

        # 2 Initialize the var-form lattice parameters
        self._lattice = lattice
        self._S = S
        self._initial_state = initial_state

        # 3 Set the parametrized couplings. Either `rep` or `coupling_matrices` must be given as argument.
        # xor these two arguments:
        assert (rep is not None) ^ (coupling_matrices is not None), \
            'Either `rep` or `coupling_matrices` must be given.'

        # Note: The attribute self._coupling_matrices will be a 4d - numpy array of shape
        # (ndim, ncouplings_per_dim, ncomponents, ncomponents), i.e.
        # self._coupling_matrices[2][3] returns the `3`rd coupling matrix for hopping along
        # lattice dimension `2` as a (ncomponents, ncomponents) numpy array.
        if rep is not None:
            gamma = [val for val in rep.values()]
            self._coupling_matrices = np.array([np.array([-1j * gamma[0] @ gamma[hopping_dim + 1]])
                                                for hopping_dim in range(self._lattice.ndim)])

        elif coupling_matrices is not None:
            self._coupling_matrices = atleast_4d(coupling_matrices)

        # 4 Parse the coupling_matrices:
        # Each dimension should have a list of coupling matrices:
        if not self._coupling_matrices.shape[0] == self._lattice.ndim:
            raise UserWarning('Error with coupling matrices. The given lattice has {} dimensions but only coupling'
                              'matrices were given for {} dimensions. Please check that `coupling_matrices` has the '
                              'shape (lattice.ndim, n_couplings_per_dim, ncomponents, ncomponents)'.format(
                self._lattice.ndim, self._coupling_matrices.shape[0]))
        # And each coupling matrix should be a square matrices of size (ncomp, ncomp). Currently only 2 component
        # spinors are supported.
        if not self._coupling_matrices.shape[2:] == (2, 2):  # TODO: Generalize for more components spinors
            raise UserWarning('Error with coupling matrices. Your coupling matrices couple {} component spinors,'
                              'but currently only 2 component spinors are supported.'.format(
                self._coupling_matrices.shape[2]))

        # 5 Extract the number of parametrized couplings for every dimension
        self._num_couplings = self._coupling_matrices.shape[1]

        # 8 Set up the hopping terms along the edges
        self._construct_hopping_operators()
        self._construct_string_operators()
        self._construct_pair_operators()
        self._matrix_hopping_terms = []
        self._matrix_string_terms = []
        self._matrix_pair_terms = []

        # 6 Count the total number of parameters of the variational form:
        # one parameter for every edge (along which hopp. occurs) and every parametrized coupling plus
        # one parameter for final z-rotation at the end of each layer

        if depth_phase == -1: 
            self._num_parameters = (lattice.nedges * self._num_couplings             # hopping terms
                                + lattice.nsites*2 + self.num_strings_per_layer + lattice.nsites * self._depth_pair ) * depth        #  occhio metto una rotazione per fermionqubit+ spinor component z-rotations + last side 
        elif depth_phase != -1: 
            self._num_parameters = (lattice.nedges * self._num_couplings  + lattice.nsites * 2  * depth_phase + self.num_strings_per_layer + lattice.nsites * 2 + lattice.nsites * self._depth_pair ) * depth 

          
        # 7 Set bounds for the parameters
        hopping_bounds = [(-np.pi * self.hopper_correction,
                           np.pi * self.hopper_correction)] * self.num_hoppings_per_layer

        string_bounds = [(-np.pi * self.hopper_correction**2,
                           np.pi * self.hopper_correction**2)] * self.num_strings_per_layer

        phase_bounds = [(-np.pi, np.pi)] * self.num_phases_per_layer
        self._bounds = operator_sum([hopping_bounds + string_bounds + phase_bounds] * self._depth)

    @property
    def num_hoppings_per_layer(self):
        return self._lattice.nedges

    @property
    def num_strings_per_layer(self):
        return len(self._string_terms)

    @property
    def num_phases_per_layer(self):
        if self._depth_phase == -1: 
            return (self._lattice.nsites * 2 ) 
        elif self._depth_phase != -1:
            return (self._lattice.nsites * 2 )  * self._depth_phase + self._lattice.nsites * 2 
    
    @property
    def num_pair_per_layer(self):
        return self._lattice.nsites * self._depth_pair

    @property
    def num_hopping_parameters(self):
        return self.num_hoppings_per_layer * self._num_couplings * self._depth

    @property
    def num_strings_parameters(self):
        return self.num_strings_per_layer * self._depth

    @property
    def num_pair_parameters(self):
        return self.num_pair_per_layer * self._depth

    @property
    def num_phase_parameters(self):
        return self.num_phases_per_layer * self._depth

    @property
    def hopper_correction(self) -> float:
        """ The correction for the norms of the hopping terms to make sure that all hopping strengths can be
        reached within the boundaries of (-np.pi, np.pi)"""
        if self._S == 0:
            return 1.
        else:
            return np.sqrt((self._S + 1) * self._S) * 2

    def _construct_hopping_operators(self, mode='qiskit'):
        """
        Constructs the Hamiltonian inspired hopping operators which are exponentiated when building the circuit and
        stores them in `self._hopping_terms` (for 'qiskit' mode) or `self._matrix_hopping_terms` (for 'qutip' mode).

        Args:
            mode (str): Must be one of ['qiskit', 'qutip']

        Returns:
            None
        """
        if mode == 'qiskit':
            self._hopping_terms = []  # will be a list of shape (nedges * ncouplings)
        elif mode == 'qutip':
            self._matrix_hopping_terms = []  # will be a list of shape (nedges * ncouplings)
        else:
            raise UserWarning("mode must be one of ['qiskit', 'qutip']")

        # 1. Iterate through all lattice edges
        for edge in self._lattice.edges:
            hopping_dim = edge[1]

            # 2. Iterate through all coupling matrices (mixmatrices) along the given dimension of the edge
            for mixmatrix in self._coupling_matrices[hopping_dim]:
                # Build up the hopping term along this edge with the specified coupling
                hopper_edge = hopping_like(edge, lattice=self._lattice, S=self._S, mixmat=mixmatrix, output=mode)
                # Simplify the operator if in qiskit mode.
                if mode == 'qiskit':
                    hopper_edge.chop()
                    self._hopping_terms.append(hopper_edge)
                elif mode == 'qutip':
                    self._matrix_hopping_terms.append(hopper_edge)

    def _construct_string_operators(self, mode='qiskit'):
        if mode == 'qiskit':
            self._string_terms = []  # will be a list of shape (nedges * ncouplings)
        elif mode == 'qutip':
            self._matrix_string_terms = []  # will be a list of shape (nedges * ncouplings)
        else:
            raise UserWarning("mode must be one of ['qiskit', 'qutip']")

        # 1. Iterate through all pairs of lattice edges
        # list_edge_pairs = get_list_edge_pairs(self._lattice)

        # Warning: Hardcoded version for the specific case of a 2 x 2 lattice
        #if self._lattice.nsites != 3:
        #    raise UserWarning("THE string term is only for 3 sites ")
        #edges_list = []
       
        #edges = [[0,0], [1,0]]
        mixmat = self._coupling_matrices[0][0]
        edges_list= []
        N = self._lattice.nsites
        for m in range(2, N):
            for i in range(N-m):
                v = np.arange(i,i + m )
                el= []
                for v_el in v:
                    el.append([v_el,0])
                edges_list.append(el)
        
        #mixmat per creare a - . - p 
        for i in range(len(edges_list)):
            #print(len(edges_list))
            string_edge = string_like(edges_list[i], lattice=self._lattice, S=self._S, mixmat=mixmat, output=mode)
            string_edge.chop()
            self._string_terms.append(string_edge)
        
        #mixmat per creare p -> -> . -> -> a 
        for i in range(len(edges_list)):
            #print(len(edges_list))
            string_edge = string_like(edges_list[i], lattice=self._lattice, S=self._S, mixmat= np.array([[0,1],[0,0]]) , output=mode)
            string_edge.chop()
            self._string_terms.append(string_edge)
      

        
        #mixmat = self._coupling_matrices[0][0] #così dovrebbe prendere solo una matrice 
        #string_edge1 = string_like(edges, lattice=self._lattice, S=self._S, mixmat=mixmat, output=mode)
        #string_edge1.chop()
        
        #self._string_terms.append(string_edge1)

    def _construct_pair_operators(self, mode='qiskit'):
       
        if mode == 'qiskit':
            self._pair_terms = []  # will be a list of shape (nedges * ncouplings)
        elif mode == 'qutip':
            self._matrix_pair_terms = []  # will be a list of shape (nedges * ncouplings)
        else:
            raise UserWarning("mode must be one of ['qiskit', 'qutip']")

        for site in self._lattice.sites:
            pair_site = pair_like(site=site, lattice=self._lattice, S=self._S)
            if mode == 'qiskit':
                    pair_site.chop()
                    self._pair_terms.append(pair_site)
    

    def construct_circuit(self, parameters, q=None, num_time_slices=1):
        """
        Construct the variational form, given its parameters.
        Args:
            parameters (numpy.ndarray): circuit parameters.
            coupling_matrices (list): list of np.ndarray for the coupling matrices
            q (QuantumRegister): Quantum Register for the circuit.
        Returns:
            QuantumCircuit: a quantum circuit with given `parameters`
        Raises:
            ValueError: the number of parameters is incorrect.
        """
        edges_list= []
        N = self._lattice.nsites
        for m in range(2, N):
            for i in range(N-m):
                v = np.arange(i,i + m )
                el= []
                for v_el in v:
                    el.append([v_el,0])
                edges_list.append(el)
        
        # 1 Parse inputs
        if len(parameters) != self._num_parameters:
            raise ValueError('The number of parameters has to be {}'.format(self._num_parameters))

        # 2 Set up the quantum register and the circuit
        if q is None:
            if self._initial_state is not None:
                circuit = self._initial_state.construct_circuit('circuit')
                q = circuit.qregs
            else:
                q = QuantumRegister(self._num_qubits, name='q')
                circuit = QuantumCircuit(q)
        else:
            circuit = self._initial_state.construct_circuit('circuit', q)
        all_qubits = circuit.qubits
        fermion_qubits = circuit.qubits[:self._lattice.nsites * 2]

        # 3 Build up the variational form with the parameters
        hoppings_per_layer = self.num_hoppings_per_layer
        strings_per_layer = self.num_strings_per_layer
        couplings_per_hopping = self._num_couplings
        phases_per_layer = self.num_phases_per_layer
        pair_per_layer = self.num_pair_per_layer
        
        params_per_layer = couplings_per_hopping * hoppings_per_layer + phases_per_layer + strings_per_layer + pair_per_layer
        
        #params_per_layer = couplings_per_hopping * hoppings_per_layer + phases_per_layer

        # 4. Iterate over depth to construct the individual blocks
        for d in range(self._depth):
            
            hopping_params = parameters[d * params_per_layer: (d + 1) * params_per_layer
                                                              - (phases_per_layer + strings_per_layer+ pair_per_layer)]
            #print("hopping_params",hopping_params) 
            string_params = parameters[(d + 1) * params_per_layer - (phases_per_layer + strings_per_layer+ pair_per_layer):
                                   (d + 1) * params_per_layer - (phases_per_layer+pair_per_layer) ]
            #print("string_params",string_params) 
            phase_params = parameters[(d + 1) * params_per_layer - (phases_per_layer+pair_per_layer): (d + 1) * params_per_layer- pair_per_layer]
            # 4.1 For each depth-block add parametrized exponentials of hopping terms
            pair_params = parameters[(d + 1) * params_per_layer - pair_per_layer: (d + 1) * params_per_layer]
    
                                                    
            
            
            #phase_params = parameters[(d + 1) * params_per_layer - phases_per_layer: (d + 1) * params_per_layer]
            #print("phase_params",phase_params)
            #print("np.arange(hoppings_per_layer)", np.arange(hoppings_per_layer),"len(self._hopping_terms)", len(self._hopping_terms), "hopping_params ",hopping_params )
            #for i, hopper, parameter in zip(np.arange(hoppings_per_layer), self._hopping_terms, hopping_params):
            var_terms_hop = [] 
            var_terms_string = []
            var_terms_pair = []
            for i, hopper, parameter in zip(np.arange(len(self._hopping_terms)), self._hopping_terms, hopping_params):
                #var_term = construct_trotter_step(hopper,parameter,
                                                #name='h_{}({})'.format(
                                                #      self._lattice.edges[i // couplings_per_hopping],
                                                #      parameter), num_time_slices= num_time_slices)
                var_terms_hop.append(construct_trotter_step(hopper,
                                                  parameter,
                                                  name='h_{}({})'.format(self._lattice.edges[i//couplings_per_hopping],
                                                                         parameter), num_time_slices= num_time_slices))
                #circuit.append(var_term, qargs=all_qubits)
            for i, string_term, parameter in zip(np.arange(len(self._string_terms)),self._string_terms, string_params):
                # c'é un problema negli string term ( non esiste edge [2], dovrei mettere mod )
                # var_terms_string.append(construct_trotter_step(string_term, parameter, name='s_[{},{}]({})'.format(edges_list[i][0][0],(edges_list[i][-1][0]+1),parameter),num_time_slices= num_time_slices))
                var_terms_string.append(construct_trotter_step(string_term, parameter, name='s_[{}]({})'.format(i,parameter),num_time_slices= num_time_slices))
            
            for d_pair in range(self._depth_pair):
                pair_params_sublayer = pair_params[d_pair * self._lattice.nsites :(d_pair+1) * self._lattice.nsites ]
                for i, pair_term, parameter in zip(np.arange(len(self._pair_terms)),self._pair_terms, pair_params_sublayer  ):
                    var_terms_pair.append(construct_trotter_step(pair_term, parameter, name='pair_[{}]({})'.format(i,parameter),num_time_slices= num_time_slices))

            if self._depth_phase == -1 :
                for var_term in var_terms_hop:
                    circuit.append(var_term, qargs=all_qubits)
                for string_term_c in var_terms_string:
                    circuit.append(string_term_c, qargs=all_qubits)

                for pair_term_c in var_terms_pair:
                    circuit.append(pair_term_c, qargs=all_qubits)

                for qbit, phase_param in zip(fermion_qubits, phase_params):
                            circuit.rz(phase_param, qbit)

            if self._depth_phase != -1:
                phases_per_layer_phase = self._lattice.nsites * 2 
                #print(phases_per_layer_phase , int(phases_per_layer_phase ), phases_per_layer, self._depth_phase  )
                #occhio che in GT& é diverso len(varform._hopping_terms)hoppings_per_layer
                if self._depth_phase <= len(self._hopping_terms):
                    #forward
                    """
                    n_rest =  self._depth_phase
                    for d in range(self._depth_phase):
                        phase_params_per_d_phase = phase_params[d * phases_per_layer_phase : (d+1) * phases_per_layer_phase ]
                        circuit.append(var_terms[d], qargs=all_qubits)
                        for qbit, phase_param in zip(fermion_qubits, phase_params_per_d_phase):
                            circuit.rz(phase_param, qbit)
                    for var_term in var_terms[n_rest:]:
                        circuit.append(var_term, qargs=all_qubits)
                    """  
                    n_rest = len(self._hopping_terms) - self._depth_phase 
                    #print(n_rest)
                    #print(len(self._hopping_terms) )
                    for var_term in var_terms_hop[:n_rest]:
                        circuit.append(var_term, qargs=all_qubits)

                    for d in range(self._depth_phase):
                        phase_params_per_d_phase = phase_params[d * phases_per_layer_phase : (d+1) * phases_per_layer_phase]

                        circuit.append(var_terms_hop[n_rest+d], qargs=all_qubits)
                        
                        for qbit, phase_param in zip(fermion_qubits, phase_params_per_d_phase):
                            circuit.rz(phase_param, qbit)

                    for string_term_c in var_terms_string:
                            circuit.append(string_term_c, qargs=all_qubits)

                    for pair_term_c in var_terms_pair:
                        circuit.append(pair_term_c, qargs=all_qubits)

                    phase_params_per_d_phase_last = phase_params[ len(phase_params) - self._lattice.nsites *2 :]
                    for qbit, phase_param in zip(fermion_qubits, phase_params_per_d_phase_last):
                            circuit.rz(phase_param, qbit)


                    
                    #for string_term_c in var_terms_string:
                    #    circuit.append(string_term_c, qargs=all_qubits)
                    

                if self._depth_phase > len(self._hopping_terms):
                    n_rest = self._depth_phase - len(self._hopping_terms) 
                    n_rest_string = n_rest - len(string_params)


                    for d in range(len(self._hopping_terms)):
                        phase_params_per_d_phase = phase_params[d * phases_per_layer_phase : (d+1) * phases_per_layer_phase]

                        circuit.append(var_terms_hop[d], qargs=all_qubits)
                        
                        for qbit, phase_param in zip(fermion_qubits, phase_params_per_d_phase):
                            circuit.rz(phase_param, qbit)

                    if n_rest_string <= 0 : 
                        for d in range(n_rest):
                            phase_params_per_d_phase = phase_params[(d+ len(self._hopping_terms)) * phases_per_layer_phase : (d+1 + len(self._hopping_terms)) * phases_per_layer_phase]
                            circuit.append(var_terms_string[d], qargs=all_qubits)

                            for qbit, phase_param in zip(fermion_qubits, phase_params_per_d_phase):
                                circuit.rz(phase_param, qbit)
                        for string_term_c in var_terms_string[n_rest:]:
                            circuit.append(string_term_c, qargs=all_qubits)

                        for pair_term_c in var_terms_pair:
                            circuit.append(pair_term_c, qargs=all_qubits)

                        phase_params_per_d_phase_last = phase_params[ len(phase_params) - self._lattice.nsites *2 :]
                        for qbit, phase_param in zip(fermion_qubits, phase_params_per_d_phase_last):
                            circuit.rz(phase_param, qbit)

                    if n_rest_string > 0 :  
                        for d in range(len(self._string_terms)):
                            phase_params_per_d_phase = phase_params[(d+ len(self._hopping_terms)) * phases_per_layer_phase : (d+1 + len(self._hopping_terms)) * phases_per_layer_phase]
                            circuit.append(var_terms_string[d], qargs=all_qubits)

                            for qbit, phase_param in zip(fermion_qubits, phase_params_per_d_phase):
                                circuit.rz(phase_param, qbit) 
                        if n_rest_string > len(self._pair_terms ) :
                            print(" Error , phase_depth has to be smaller ")   
                        if n_rest_string == len(self._pair_terms ):
                            
                            for d in range(len(self._pair_terms)):
                                phase_params_per_d_phase = phase_params[(d+ len(self._hopping_terms) + len(self._string_terms)) * phases_per_layer_phase : (d+1 +len(self._hopping_terms) + len(self._string_terms)) * phases_per_layer_phase]
                                circuit.append(var_terms_pair[d], qargs=all_qubits)
                                for qbit, phase_param in zip(fermion_qubits, phase_params_per_d_phase):
                                    circuit.rz(phase_param, qbit)
                            phase_params_per_d_phase_last = phase_params[ len(phase_params) - self._lattice.nsites *2 :]
                            for qbit, phase_param in zip(fermion_qubits, phase_params_per_d_phase_last):
                                circuit.rz(phase_param, qbit)

                        if n_rest_string < len(self._pair_terms ):
                            for d in range(n_rest_string):
                                phase_params_per_d_phase = phase_params[(d+ len(self._hopping_terms) + len(self._string_terms)) * phases_per_layer_phase : (d+1 +len(self._hopping_terms) + len(self._string_terms)) * phases_per_layer_phase]
                                circuit.append(var_terms_pair[d], qargs=all_qubits)
                                for qbit, phase_param in zip(fermion_qubits, phase_params_per_d_phase):
                                    circuit.rz(phase_param, qbit)

                            for pair_term_c in var_terms_pair[n_rest_string:]:
                                circuit.append(pair_term_c, qargs=all_qubits)
                            phase_params_per_d_phase_last = phase_params[ len(phase_params) - self._lattice.nsites *2 :]
                            for qbit, phase_param in zip(fermion_qubits, phase_params_per_d_phase_last):
                                circuit.rz(phase_param, qbit)
        return circuit

class WilsonLGT4_new_pair(VariationalForm):

    def __init__(self, lattice, S, rep=None, coupling_matrices=None, depth=1, initial_state=None, mix_bool = False , depth_phase=-1):
        
        super().__init__()

        # 1 Initialize the varform-circuit parameters
        self._num_qubits = lattice.nsites * 2 + lattice.nedges * int(np.ceil(np.log2(2 * S + 1)))
        self._depth = depth

        # 2 Initialize the var-form lattice parameters
        self._lattice = lattice
        self._S = S
        self._initial_state = initial_state
        self._mix_bool = mix_bool
        self._depth_phase = depth_phase

        # 3 Set the parametrized couplings. Either `rep` or `coupling_matrices` must be given as argument.
        # xor these two arguments:
        assert (rep is not None) ^ (coupling_matrices is not None), \
            'Either `rep` or `coupling_matrices` must be given.'

        # Note: The attribute self._coupling_matrices will be a 4d - numpy array of shape
        # (ndim, ncouplings_per_dim, ncomponents, ncomponents), i.e.
        # self._coupling_matrices[2][3] returns the `3`rd coupling matrix for hopping along
        # lattice dimension `2` as a (ncomponents, ncomponents) numpy array.
        if rep is not None:
            gamma = [val for val in rep.values()]
            self._coupling_matrices = np.array([np.array([-1j * gamma[0] @ gamma[hopping_dim + 1]])
                                                for hopping_dim in range(self._lattice.ndim)])

        elif coupling_matrices is not None:
            self._coupling_matrices = atleast_4d(coupling_matrices)

        # 4 Parse the coupling_matrices:
        # Each dimension should have a list of coupling matrices:
        if not self._coupling_matrices.shape[0] == self._lattice.ndim:
            raise UserWarning('Error with coupling matrices. The given lattice has {} dimensions but only coupling'
                              'matrices were given for {} dimensions. Please check that `coupling_matrices` has the '
                              'shape (lattice.ndim, n_couplings_per_dim, ncomponents, ncomponents)'.format(
                self._lattice.ndim, self._coupling_matrices.shape[0]))
        # And each coupling matrix should be a square matrices of size (ncomp, ncomp). Currently only 2 component
        # spinors are supported.
        if not self._coupling_matrices.shape[2:] == (2, 2):  # TODO: Generalize for more components spinors
            raise UserWarning('Error with coupling matrices. Your coupling matrices couple {} component spinors,'
                              'but currently only 2 component spinors are supported.'.format(
                self._coupling_matrices.shape[2]))

        # 5 Extract the number of parametrized couplings for every dimension
        self._num_couplings = self._coupling_matrices.shape[1]

        # 8 Set up the hopping terms along the edges
        self._construct_hopping_operators()
        self._construct_string_operators()
        self._construct_pair_operators()
        self._matrix_hopping_terms = []
        self._matrix_string_terms = []
        self._matrix_pair_terms = []

        

        # 6 Count the total number of parameters of the variational form:
        # one parameter for every edge (along which hopp. occurs) and every parametrized coupling plus
        # one parameter for final z-rotation at the end of each layer
        if self._depth_phase == -1:
            self._num_parameters = (lattice.nedges * self._num_couplings  # hopping terms
                                + self.num_strings_per_layer  # string terms
                                + lattice.nsites * 2 + self.num_pair_per_layer ) * depth   # + aggiungiamo 2 per ogni qubit 
        elif self._depth_phase != -1:
            self._num_parameters = (lattice.nedges * self._num_couplings  # hopping terms
                                + self.num_strings_per_layer  # string terms
                                + lattice.nsites * 2 + lattice.nsites * 2 * depth_phase + self.num_pair_per_layer ) * depth 


        # 7 Set bounds for the parameters
        hopping_bounds = [(-np.pi * self.hopper_correction,
                           np.pi * self.hopper_correction)] * self.num_hoppings_per_layer

        string_bounds = [(-np.pi * self.hopper_correction**2,
                           np.pi * self.hopper_correction**2)] * self.num_strings_per_layer

        phase_bounds = [(-np.pi, np.pi)] * self.num_phases_per_layer
        self._bounds = operator_sum([hopping_bounds + string_bounds + phase_bounds] * self._depth)

    @property
    def num_hoppings_per_layer(self):
        return self._lattice.nedges

    @property
    def num_strings_per_layer(self):
        return len(self._string_terms)

    @property
    def num_phases_per_layer(self):
        if self._depth_phase != -1:
            return self._lattice.nsites * 2 + self._lattice.nsites * 2 * self._depth_phase
        elif self._depth_phase == -1:
            return self._lattice.nsites * 2

    @property
    def num_pair_per_layer(self):
        return self._lattice.nsites 
             
    @property
    def num_hopping_parameters(self):
        return self.num_hoppings_per_layer * self._num_couplings * self._depth

    @property
    def num_strings_parameters(self):
        return self.num_strings_per_layer * self._depth

    @property
    def num_phase_parameters(self):
        return self.num_phases_per_layer * self._depth

    @property
    def hopper_correction(self) -> float:
        """ The correction for the norms of the hopping terms to make sure that all hopping strengths can be
        reached within the boundaries of (-np.pi, np.pi)"""
        if self._S == 0:
            return 1.
        else:
            return np.sqrt((self._S + 1) * self._S) * 2

    def _construct_hopping_operators(self, mode='qiskit'):
        """
        Constructs the Hamiltonian inspired hopping operators which are exponentiated when building the circuit and
        stores them in `self._hopping_terms` (for 'qiskit' mode) or `self._matrix_hopping_terms` (for 'qutip' mode).

        Args:
            mode (str): Must be one of ['qiskit', 'qutip']

        Returns:
            None
        """
        if mode == 'qiskit':
            self._hopping_terms = []  # will be a list of shape (nedges * ncouplings)
        elif mode == 'qutip':
            self._matrix_hopping_terms = []  # will be a list of shape (nedges * ncouplings)
        else:
            raise UserWarning("mode must be one of ['qiskit', 'qutip']")

        # 1. Iterate through all lattice edges
        for edge in self._lattice.edges:
            hopping_dim = edge[1]

            # 2. Iterate through all coupling matrices (mixmatrices) along the given dimension of the edge
            for mixmatrix in self._coupling_matrices[hopping_dim]:
                # Build up the hopping term along this edge with the specified coupling
                hopper_edge = hopping_like(edge, lattice=self._lattice, S=self._S, mixmat=mixmatrix, output=mode)
                # Simplify the operator if in qiskit mode.
                if mode == 'qiskit':
                    hopper_edge.chop()
                    self._hopping_terms.append(hopper_edge)
                elif mode == 'qutip':
                    self._matrix_hopping_terms.append(hopper_edge)

    def _construct_string_operators(self, mode='qiskit'):
        if mode == 'qiskit':
            self._string_terms = []  # will be a list of shape (nedges * ncouplings)
        elif mode == 'qutip':
            self._matrix_string_terms = []  # will be a list of shape (nedges * ncouplings)
        else:
            raise UserWarning("mode must be one of ['qiskit', 'qutip']")

        # 1. Iterate through all pairs of lattice edges
        # list_edge_pairs = get_list_edge_pairs(self._lattice)

        # Warning: Hardcoded version for the specific case of a 2 x 2 lattice
        edge1 = np.ndarray(2, dtype='int64')
        edge1[0] = 0
        edge1[1] = 0
        edge2 = np.ndarray(2, dtype='int64')
        edge2[0] = 1
        edge2[1] = 1
        edge3 = np.ndarray(2, dtype='int64')
        edge3[0] = 0
        edge3[1] = 1
        edge4 = np.ndarray(2, dtype='int64')
        edge4[0] = 2
        edge4[1] = 0
        list_edge_pairs = [[edge1, edge2], [edge3, edge4]]
        """
        for edge_pair in list_edge_pairs:

            # 2. Iterate through all coupling matrices (mixmatrices) along the given dimension of the edge
            for mixmatrix in self._coupling_matrices[0]:  # TODO: generalize for arbitrary coupling matrices
                # Build up the hopping term along this edge with the specified coupling
                string_edge = string_like(edge_pair, lattice=self._lattice, S=self._S, mixmat=mixmatrix, output=mode)
                # Simplify the operator if in qiskit mode
                if mode == 'qiskit':
                    string_edge.chop()
                    self._string_terms.append(string_edge)
                elif mode == 'qutip':
                    self._matrix_string_terms.append(string_edge)
        """
        mixmat = self._coupling_matrices[0][0] #così dovrebbe prendere solo una matrice 
        string_edge1 = string_like(list_edge_pairs[0], lattice=self._lattice, S=self._S, mixmat=mixmat, output=mode)
        string_edge1.chop()
        string_edge2 = string_like(list_edge_pairs[1], lattice=self._lattice, S=self._S, mixmat=mixmat, output=mode)
        string_edge2.chop()


        #mixstring1 = string_edge1.multiply(string_edge2)
        #for i,mix in enumerate(mixstring1.paulis):
        #    if np.imag(mix[0]) != 0 :
        #        mixstring1.paulis[i][0] *= 1j
        #mixstring1.chop()        

        self._string_terms.append(string_edge1)
        self._string_terms.append(string_edge2)

        if not self._mix_bool:
            print("\nNon c'é mixed term\n")
        elif self._mix_bool: 
            print("\nMettiamo mixed term\n")
            mixstring1 = string_edge1.multiply(string_edge2)
            for i,mix in enumerate(mixstring1.paulis):
               if np.imag(mix[0]) != 0 :
                mixstring1.paulis[i][0] *= 1j
            mixstring1.chop()  
            self._string_terms.append(mixstring1)

    def _construct_pair_operators(self, mode='qiskit'):
       
        if mode == 'qiskit':
            self._pair_terms = []  # will be a list of shape (nedges * ncouplings)
        elif mode == 'qutip':
            self._matrix_pair_terms = []  # will be a list of shape (nedges * ncouplings)
        else:
            raise UserWarning("mode must be one of ['qiskit', 'qutip']")

        for site in self._lattice.sites:
            pair_site = pair_like(site=site, lattice=self._lattice, S=self._S)
            if mode == 'qiskit':
                    pair_site.chop()
                    self._pair_terms.append(pair_site)

    def construct_circuit(self, parameters, q=None, num_time_slices=1):
        """
        Construct the variational form, given its parameters.
        Args:
            parameters (numpy.ndarray): circuit parameters.
            coupling_matrices (list): list of np.ndarray for the coupling matrices
            q (QuantumRegister): Quantum Register for the circuit.
        Returns:
            QuantumCircuit: a quantum circuit with given `parameters`
        Raises:
            ValueError: the number of parameters is incorrect.
        """
        # 1 Parse inputs
        if len(parameters) != self._num_parameters:
            raise ValueError('The number of parameters has to be {}'.format(self._num_parameters))

        # 2 Set up the quantum register and the circuit
        if q is None:
            if self._initial_state is not None:
                circuit = self._initial_state.construct_circuit('circuit')
                q = circuit.qregs
            else:
                q = QuantumRegister(self._num_qubits, name='q')
                circuit = QuantumCircuit(q)
        else:
            circuit = self._initial_state.construct_circuit('circuit', q)
        all_qubits = circuit.qubits
        fermion_qubits = circuit.qubits[:self._lattice.nsites * 2]

        

        # 3 Build up the variational form with the parameters
        hoppings_per_layer = self.num_hoppings_per_layer
        strings_per_layer = self.num_strings_per_layer
        couplings_per_hopping = self._num_couplings
        phases_per_layer = self.num_phases_per_layer
        pair_per_layer = self.num_pair_per_layer
        params_per_layer = couplings_per_hopping * hoppings_per_layer + phases_per_layer + strings_per_layer + pair_per_layer

        # 4. Iterate over depth to construct the individual blocks
        for d in range(self._depth):
            # 4.1 For each depth-block add parametrized exponentials of hopping terms
            hopping_params = parameters[d * params_per_layer: (d + 1) * params_per_layer
                                                              - (phases_per_layer + strings_per_layer+ pair_per_layer)]
            #print("hopping_params",hopping_params) 
            string_params = parameters[(d + 1) * params_per_layer - (phases_per_layer + strings_per_layer + pair_per_layer):
                                   (d + 1) * params_per_layer - (phases_per_layer+pair_per_layer)]
            #print("string_params",string_params) 
            phase_params = parameters[(d + 1) * params_per_layer - (phases_per_layer+pair_per_layer): (d + 1) * params_per_layer- pair_per_layer]
            #print("phase_params",phase_params)
            pair_params = parameters[(d + 1) * params_per_layer - pair_per_layer: (d + 1) * params_per_layer]
    
                                                    
            
        
          
            
            #print("np.arange(hoppings_per_layer)", np.arange(hoppings_per_layer),"len(self._hopping_terms)", len(self._hopping_terms), "hopping_params ",hopping_params )
            #for i, hopper, parameter in zip(np.arange(hoppings_per_layer), self._hopping_terms, hopping_params):
            if self._depth_phase == -1: 
                for i, hopper, parameter in zip(np.arange(len(self._hopping_terms)), self._hopping_terms, hopping_params):
                    var_term = construct_trotter_step(hopper,
                                                  parameter,
                                                  name='h_{}({})'.format(
                                                      self._lattice.edges[i // couplings_per_hopping],
                                                      parameter), num_time_slices= num_time_slices)
                    circuit.append(var_term, qargs=all_qubits)
                for string_term, parameter in zip(self._string_terms, string_params):
                    var_term = construct_trotter_step(string_term, parameter, num_time_slices= num_time_slices)
                    circuit.append(var_term, qargs=all_qubits)

                for i, pair_term, parameter in zip(np.arange(len(self._pair_terms)),self._pair_terms, pair_params  ):
                    var_term =  construct_trotter_step(pair_term, parameter, name='pair_[{}]({})'.format(i,parameter),num_time_slices= num_time_slices)
                    circuit.append(var_term, qargs=all_qubits)
                for qbit, phase_param in zip(fermion_qubits, phase_params):
                    circuit.u1(phase_param, qbit)


            elif self._depth_phase != -1: 
                #phase_params_last = phase_params[ -self._lattice.nsites * 2 :]
                #phase_params = phase_params[ :self._lattice.nsites * 2 ]
                var_terms_circ = []
                for i, hopper, parameter in zip(np.arange(len(self._hopping_terms)), self._hopping_terms, hopping_params):
                    var_terms_circ.append(construct_trotter_step(hopper,
                                                  parameter,
                                                  name='h_{}({})'.format(
                                                      self._lattice.edges[i // couplings_per_hopping],
                                                      parameter), num_time_slices= num_time_slices))
                string_terms_circ = []
                for string_term, parameter in zip(self._string_terms, string_params):
                    string_terms_circ.append(construct_trotter_step(string_term, parameter, num_time_slices= num_time_slices))
                
                var_terms_pair = []
                for i, pair_term, parameter in zip(np.arange(len(self._pair_terms)),self._pair_terms, pair_params  ):
                    var_terms_pair.append(construct_trotter_step(pair_term, parameter, name='pair_[{}]({})'.format(i,parameter),num_time_slices= num_time_slices))

                phases_per_layer_phase = self._lattice.nsites * 2 
                #print(phases_per_layer_phase , int(phases_per_layer_phase ), phases_per_layer, self._depth_phase  )
                #occhio che in GT& é diverso len(varform._hopping_terms)hoppings_per_layer
                if self._depth_phase <= len(self._hopping_terms):
                    #forward
                    n_rest = len(self._hopping_terms) - self._depth_phase
                    #print(n_rest)
                    #print(len(self._hopping_terms) )
                    for var_term in var_terms_circ[:n_rest]:
                        circuit.append(var_term, qargs=all_qubits)

                    for d in range(self._depth_phase):
                        phase_params_per_d_phase = phase_params[d * phases_per_layer_phase : (d+1) * phases_per_layer_phase ]
                        
                        circuit.append(var_terms_circ[n_rest+d], qargs=all_qubits)
                        
                        for qbit, phase_param in zip(fermion_qubits, phase_params_per_d_phase):
                            circuit.rz(phase_param, qbit)
                    
                    for var_term in string_terms_circ:
                        circuit.append(var_term, qargs=all_qubits)

                    for pair_term_c in var_terms_pair:
                        circuit.append(pair_term_c, qargs=all_qubits)

                    for qbit, phase_param in zip(fermion_qubits, phase_params[(self._depth_phase) * self._lattice.nsites * 2  :]):
                        circuit.u1(phase_param, qbit)
                    

        return circuit

class WilsonLGT6_double_string_pair_N_1(VariationalForm):
    def __init__(self, lattice, S, rep=None, coupling_matrices=None, depth=1, initial_state=None, depth_phase=-1, depth_pair=1, depth_hop=1):
      
     
        super().__init__()

        # 1 Initialize the varform-circuit parameters
        self._num_qubits = lattice.nsites * 2 + lattice.nedges * int(np.ceil(np.log2(2 * S + 1)))
        self._depth = depth
        self._depth_phase = depth_phase
        self._depth_pair = depth_pair

        # 2 Initialize the var-form lattice parameters
        self._lattice = lattice
        self._S = S
        self._initial_state = initial_state

        # 3 Set the parametrized couplings. Either `rep` or `coupling_matrices` must be given as argument.
        # xor these two arguments:
        assert (rep is not None) ^ (coupling_matrices is not None), \
            'Either `rep` or `coupling_matrices` must be given.'

        # Note: The attribute self._coupling_matrices will be a 4d - numpy array of shape
        # (ndim, ncouplings_per_dim, ncomponents, ncomponents), i.e.
        # self._coupling_matrices[2][3] returns the `3`rd coupling matrix for hopping along
        # lattice dimension `2` as a (ncomponents, ncomponents) numpy array.
        if rep is not None:
            gamma = [val for val in rep.values()]
            self._coupling_matrices = np.array([np.array([-1j * gamma[0] @ gamma[hopping_dim + 1]])
                                                for hopping_dim in range(self._lattice.ndim)])

        elif coupling_matrices is not None:
            self._coupling_matrices = atleast_4d(coupling_matrices)

        # 4 Parse the coupling_matrices:
        # Each dimension should have a list of coupling matrices:
        if not self._coupling_matrices.shape[0] == self._lattice.ndim:
            raise UserWarning('Error with coupling matrices. The given lattice has {} dimensions but only coupling'
                              'matrices were given for {} dimensions. Please check that `coupling_matrices` has the '
                              'shape (lattice.ndim, n_couplings_per_dim, ncomponents, ncomponents)'.format(
                self._lattice.ndim, self._coupling_matrices.shape[0]))
        # And each coupling matrix should be a square matrices of size (ncomp, ncomp). Currently only 2 component
        # spinors are supported.
        if not self._coupling_matrices.shape[2:] == (2, 2):  # TODO: Generalize for more components spinors
            raise UserWarning('Error with coupling matrices. Your coupling matrices couple {} component spinors,'
                              'but currently only 2 component spinors are supported.'.format(
                self._coupling_matrices.shape[2]))

        # 5 Extract the number of parametrized couplings for every dimension
        self._num_couplings = self._coupling_matrices.shape[1]

        # 8 Set up the hopping terms along the edges
        self._construct_hopping_operators()
        self._construct_string_operators()
        self._construct_pair_operators()
        self._matrix_hopping_terms = []
        self._matrix_string_terms = []
        self._matrix_pair_terms = []

        # 6 Count the total number of parameters of the variational form:
        # one parameter for every edge (along which hopp. occurs) and every parametrized coupling plus
        # one parameter for final z-rotation at the end of each layer

        if depth_phase == -1: 
            self._num_parameters = (lattice.nedges * self._num_couplings             # hopping terms
                                + lattice.nsites*2 + self.num_strings_per_layer + self.num_pair_per_layer ) * depth        #  occhio metto una rotazione per fermionqubit+ spinor component z-rotations + last side 
        elif depth_phase != -1: 
            self._num_parameters = (lattice.nedges * self._num_couplings  + lattice.nsites * 2  * depth_phase + self.num_strings_per_layer + lattice.nsites * 2 + self.num_pair_per_layer) * depth 

          
        # 7 Set bounds for the parameters
        hopping_bounds = [(-np.pi * self.hopper_correction,
                           np.pi * self.hopper_correction)] * self.num_hoppings_per_layer

        string_bounds = [(-np.pi * self.hopper_correction**2,
                           np.pi * self.hopper_correction**2)] * self.num_strings_per_layer

        phase_bounds = [(-np.pi, np.pi)] * self.num_phases_per_layer
        self._bounds = operator_sum([hopping_bounds + string_bounds + phase_bounds] * self._depth)

    @property
    def num_hoppings_per_layer(self):
        return self._lattice.nedges 

    @property
    def num_strings_per_layer(self):
        return len(self._string_terms)

    @property
    def num_phases_per_layer(self):
        if self._depth_phase == -1: 
            return (self._lattice.nsites * 2 ) 
        elif self._depth_phase != -1:
            return (self._lattice.nsites * 2 )  * self._depth_phase + self._lattice.nsites * 2 
    
    @property
    def num_pair_per_layer(self):
        #self._lattice.nsites * self._depth_pair
        return len(self._pair_terms) 

    @property
    def num_hopping_parameters(self):
        return self.num_hoppings_per_layer * self._num_couplings * self._depth

    @property
    def num_strings_parameters(self):
        return self.num_strings_per_layer * self._depth

    @property
    def num_pair_parameters(self):
        return self.num_pair_per_layer * self._depth
        # return self.num_pair_per_layer 

    @property
    def num_phase_parameters(self):
        return self.num_phases_per_layer * self._depth

    @property
    def hopper_correction(self) -> float:
        """ The correction for the norms of the hopping terms to make sure that all hopping strengths can be
        reached within the boundaries of (-np.pi, np.pi)"""
        if self._S == 0:
            return 1.
        else:
            return np.sqrt((self._S + 1) * self._S) * 2

    def _construct_hopping_operators(self, mode='qiskit'):
        """
        Constructs the Hamiltonian inspired hopping operators which are exponentiated when building the circuit and
        stores them in `self._hopping_terms` (for 'qiskit' mode) or `self._matrix_hopping_terms` (for 'qutip' mode).

        Args:
            mode (str): Must be one of ['qiskit', 'qutip']

        Returns:
            None
        """
        if mode == 'qiskit':
            self._hopping_terms = []  # will be a list of shape (nedges * ncouplings)
        elif mode == 'qutip':
            self._matrix_hopping_terms = []  # will be a list of shape (nedges * ncouplings)
        else:
            raise UserWarning("mode must be one of ['qiskit', 'qutip']")

        # 1. Iterate through all lattice edges
        for edge in self._lattice.edges:
            hopping_dim = edge[1]

            # 2. Iterate through all coupling matrices (mixmatrices) along the given dimension of the edge
            for mixmatrix in self._coupling_matrices[hopping_dim]:
                # Build up the hopping term along this edge with the specified coupling
                hopper_edge = hopping_like(edge, lattice=self._lattice, S=self._S, mixmat=mixmatrix, output=mode)
                # Simplify the operator if in qiskit mode.
                if mode == 'qiskit':
                    hopper_edge.chop()
                    self._hopping_terms.append(hopper_edge)
                elif mode == 'qutip':
                    self._matrix_hopping_terms.append(hopper_edge)

    def _construct_string_operators(self, mode='qiskit'):
        if mode == 'qiskit':
            self._string_terms = []  # will be a list of shape (nedges * ncouplings)
        elif mode == 'qutip':
            self._matrix_string_terms = []  # will be a list of shape (nedges * ncouplings)
        else:
            raise UserWarning("mode must be one of ['qiskit', 'qutip']")

        # 1. Iterate through all pairs of lattice edges
        # list_edge_pairs = get_list_edge_pairs(self._lattice)

        # Warning: Hardcoded version for the specific case of a 2 x 2 lattice
        #if self._lattice.nsites != 3:
        #    raise UserWarning("THE string term is only for 3 sites ")
        #edges_list = []
       
        #edges = [[0,0], [1,0]]
        mixmat = self._coupling_matrices[0][0]
        edges_list= []
        N = self._lattice.nsites
        for m in range(2, N):
            for i in range(N-m):
                v = np.arange(i,i + m )
                el= []
                for v_el in v:
                    el.append([v_el,0])
                edges_list.append(el)
        

        for i in range(len(edges_list)):
            #print(len(edges_list))
            string_edge = string_like(edges_list[i], lattice=self._lattice, S=self._S, mixmat=mixmat, output=mode)
            string_edge.chop()
            self._string_terms.append(string_edge)
      

        
        #mixmat = self._coupling_matrices[0][0] #così dovrebbe prendere solo una matrice 
        #string_edge1 = string_like(edges, lattice=self._lattice, S=self._S, mixmat=mixmat, output=mode)
        #string_edge1.chop()
        
        #self._string_terms.append(string_edge1)

    def _construct_pair_operators(self, mode='qiskit'):
       
        if mode == 'qiskit':
            self._pair_terms = []  # will be a list of shape (nedges * ncouplings)
        elif mode == 'qutip':
            self._matrix_pair_terms = []  # will be a list of shape (nedges * ncouplings)
        else:
            raise UserWarning("mode must be one of ['qiskit', 'qutip']")

        for site in self._lattice.sites:
            pair_site = pair_like(site=site, lattice=self._lattice, S=self._S) 
            if mode == 'qiskit':
                    pair_site.chop()
                    self._pair_terms.append(pair_site)

        mix_term = pair_like(site=self._lattice.sites[0], lattice=self._lattice, S=self._S) * pair_like(site=self._lattice.sites[1], lattice=self._lattice, S=self._S)
        if mode == 'qiskit':
                mix_term.chop()
                self._pair_terms.append(mix_term)
    

    def construct_circuit(self, parameters, q=None, num_time_slices=1):
        """
        Construct the variational form, given its parameters.
        Args:
            parameters (numpy.ndarray): circuit parameters.
            coupling_matrices (list): list of np.ndarray for the coupling matrices
            q (QuantumRegister): Quantum Register for the circuit.
        Returns:
            QuantumCircuit: a quantum circuit with given `parameters`
        Raises:
            ValueError: the number of parameters is incorrect.
        """
        edges_list= []
        N = self._lattice.nsites
        for m in range(2, N):
            for i in range(N-m):
                v = np.arange(i,i + m )
                el= []
                for v_el in v:
                    el.append([v_el,0])
                edges_list.append(el)
        
        # 1 Parse inputs
        if len(parameters) != self._num_parameters:
            raise ValueError('The number of parameters has to be {}'.format(self._num_parameters))

        # 2 Set up the quantum register and the circuit
        if q is None:
            if self._initial_state is not None:
                circuit = self._initial_state.construct_circuit('circuit')
                q = circuit.qregs
            else:
                q = QuantumRegister(self._num_qubits, name='q')
                circuit = QuantumCircuit(q)
        else:
            circuit = self._initial_state.construct_circuit('circuit', q)
        all_qubits = circuit.qubits
        fermion_qubits = circuit.qubits[:self._lattice.nsites * 2]

        # 3 Build up the variational form with the parameters
        hoppings_per_layer = self.num_hoppings_per_layer
        strings_per_layer = self.num_strings_per_layer
        couplings_per_hopping = self._num_couplings
        phases_per_layer = self.num_phases_per_layer
        pair_per_layer = self.num_pair_per_layer
        params_per_layer = couplings_per_hopping * hoppings_per_layer + phases_per_layer + strings_per_layer + pair_per_layer

        #params_per_layer = couplings_per_hopping * hoppings_per_layer + phases_per_layer

        # 4. Iterate over depth to construct the individual blocks
        for d in range(self._depth):
            
            hopping_params = parameters[d * params_per_layer: (d + 1) * params_per_layer
                                                              - (phases_per_layer + strings_per_layer+ pair_per_layer)]
            #print("hopping_params",hopping_params) 
            string_params = parameters[(d + 1) * params_per_layer - (phases_per_layer + strings_per_layer+ pair_per_layer):
                                   (d + 1) * params_per_layer - (phases_per_layer+pair_per_layer) ]
            #print("string_params",string_params) 
            phase_params = parameters[(d + 1) * params_per_layer - (phases_per_layer+pair_per_layer): (d + 1) * params_per_layer- pair_per_layer]
            # 4.1 For each depth-block add parametrized exponentials of hopping terms
            pair_params = parameters[(d + 1) * params_per_layer - pair_per_layer: (d + 1) * params_per_layer]
    
                                                    
            
            
            #phase_params = parameters[(d + 1) * params_per_layer - phases_per_layer: (d + 1) * params_per_layer]
            #print("phase_params",phase_params)
            #print("np.arange(hoppings_per_layer)", np.arange(hoppings_per_layer),"len(self._hopping_terms)", len(self._hopping_terms), "hopping_params ",hopping_params )
            #for i, hopper, parameter in zip(np.arange(hoppings_per_layer), self._hopping_terms, hopping_params):
            var_terms_hop = [] 
            var_terms_string = []
            var_terms_pair = []
            for i, hopper, parameter in zip(np.arange(len(self._hopping_terms)), self._hopping_terms, hopping_params):
                #var_term = construct_trotter_step(hopper,parameter,
                                                #name='h_{}({})'.format(
                                                #      self._lattice.edges[i // couplings_per_hopping],
                                                #      parameter), num_time_slices= num_time_slices)
                var_terms_hop.append(construct_trotter_step(hopper,
                                                  parameter,
                                                  name='h_{}({})'.format(self._lattice.edges[i//couplings_per_hopping],
                                                                         parameter), num_time_slices= num_time_slices))
                #circuit.append(var_term, qargs=all_qubits)
            for i, string_term, parameter in zip(np.arange(len(self._string_terms)),self._string_terms, string_params):
                var_terms_string.append(construct_trotter_step(string_term, parameter, name='s_[{},{}]({})'.format(edges_list[i][0][0],(edges_list[i][-1][0]+1),parameter),num_time_slices= num_time_slices))
            
            for d_pair in range(self._depth_pair):
                pair_params_sublayer = pair_params[d_pair * pair_per_layer :(d_pair+1) * pair_per_layer ]
                for i, pair_term, parameter in zip(np.arange(len(self._pair_terms)),self._pair_terms, pair_params_sublayer  ):
                    var_terms_pair.append(construct_trotter_step(pair_term, parameter, name='pair_[{}]({})'.format(i,parameter),num_time_slices= num_time_slices))

            if self._depth_phase == -1 :
                for var_term in var_terms_hop:
                    circuit.append(var_term, qargs=all_qubits)
                for string_term_c in var_terms_string:
                    circuit.append(string_term_c, qargs=all_qubits)

                for pair_term_c in var_terms_pair:
                    circuit.append(pair_term_c, qargs=all_qubits)

                for qbit, phase_param in zip(fermion_qubits, phase_params):
                            circuit.rz(phase_param, qbit)

            if self._depth_phase != -1:
                phases_per_layer_phase = self._lattice.nsites * 2 
                #print(phases_per_layer_phase , int(phases_per_layer_phase ), phases_per_layer, self._depth_phase  )
                #occhio che in GT& é diverso len(varform._hopping_terms)hoppings_per_layer
                if self._depth_phase <= len(self._hopping_terms):
                    #forward
                    """
                    n_rest =  self._depth_phase
                    for d in range(self._depth_phase):
                        phase_params_per_d_phase = phase_params[d * phases_per_layer_phase : (d+1) * phases_per_layer_phase ]
                        circuit.append(var_terms[d], qargs=all_qubits)
                        for qbit, phase_param in zip(fermion_qubits, phase_params_per_d_phase):
                            circuit.rz(phase_param, qbit)
                    for var_term in var_terms[n_rest:]:
                        circuit.append(var_term, qargs=all_qubits)
                    """  
                    n_rest = len(self._hopping_terms) - self._depth_phase 
                    #print(n_rest)
                    #print(len(self._hopping_terms) )
                    for var_term in var_terms_hop[:n_rest]:
                        circuit.append(var_term, qargs=all_qubits)

                    for d in range(self._depth_phase):
                        phase_params_per_d_phase = phase_params[d * phases_per_layer_phase : (d+1) * phases_per_layer_phase]

                        circuit.append(var_terms_hop[n_rest+d], qargs=all_qubits)
                        
                        for qbit, phase_param in zip(fermion_qubits, phase_params_per_d_phase):
                            circuit.rz(phase_param, qbit)

                    for string_term_c in var_terms_string:
                            circuit.append(string_term_c, qargs=all_qubits)

                    for pair_term_c in var_terms_pair:
                        circuit.append(pair_term_c, qargs=all_qubits)

                    phase_params_per_d_phase_last = phase_params[ len(phase_params) - self._lattice.nsites *2 :]
                    for qbit, phase_param in zip(fermion_qubits, phase_params_per_d_phase_last):
                            circuit.rz(phase_param, qbit)

                    

                if self._depth_phase > len(self._hopping_terms):
                    n_rest = self._depth_phase - len(self._hopping_terms) 
                    n_rest_string = n_rest - len(string_params)


                    for d in range(len(self._hopping_terms)):
                        phase_params_per_d_phase = phase_params[d * phases_per_layer_phase : (d+1) * phases_per_layer_phase]

                        circuit.append(var_terms_hop[d], qargs=all_qubits)
                        
                        for qbit, phase_param in zip(fermion_qubits, phase_params_per_d_phase):
                            circuit.rz(phase_param, qbit)

                    if n_rest_string <= 0 : 
                        for d in range(n_rest):
                            phase_params_per_d_phase = phase_params[(d+ len(self._hopping_terms)) * phases_per_layer_phase : (d+1 + len(self._hopping_terms)) * phases_per_layer_phase]
                            circuit.append(var_terms_string[d], qargs=all_qubits)

                            for qbit, phase_param in zip(fermion_qubits, phase_params_per_d_phase):
                                circuit.rz(phase_param, qbit)
                        for string_term_c in var_terms_string[n_rest:]:
                            circuit.append(string_term_c, qargs=all_qubits)

                        for pair_term_c in var_terms_pair:
                            circuit.append(pair_term_c, qargs=all_qubits)

                        phase_params_per_d_phase_last = phase_params[ len(phase_params) - self._lattice.nsites *2 :]
                        for qbit, phase_param in zip(fermion_qubits, phase_params_per_d_phase_last):
                            circuit.rz(phase_param, qbit)

                    if n_rest_string > 0 :  
                        for d in range(len(self._string_terms)):
                            phase_params_per_d_phase = phase_params[(d+ len(self._hopping_terms)) * phases_per_layer_phase : (d+1 + len(self._hopping_terms)) * phases_per_layer_phase]
                            circuit.append(var_terms_string[d], qargs=all_qubits)

                            for qbit, phase_param in zip(fermion_qubits, phase_params_per_d_phase):
                                circuit.rz(phase_param, qbit) 
                        if n_rest_string > len(self._pair_terms ) :
                            print(" Error , phase_depth has to be smaller ")   
                        if n_rest_string == len(self._pair_terms ):
                            
                            for d in range(len(self._pair_terms)):
                                phase_params_per_d_phase = phase_params[(d+ len(self._hopping_terms) + len(self._string_terms)) * phases_per_layer_phase : (d+1 +len(self._hopping_terms) + len(self._string_terms)) * phases_per_layer_phase]
                                circuit.append(var_terms_pair[d], qargs=all_qubits)
                                for qbit, phase_param in zip(fermion_qubits, phase_params_per_d_phase):
                                    circuit.rz(phase_param, qbit)
                            phase_params_per_d_phase_last = phase_params[ len(phase_params) - self._lattice.nsites *2 :]
                            for qbit, phase_param in zip(fermion_qubits, phase_params_per_d_phase_last):
                                circuit.rz(phase_param, qbit)

                        if n_rest_string < len(self._pair_terms ):
                            for d in range(n_rest_string):
                                phase_params_per_d_phase = phase_params[(d+ len(self._hopping_terms) + len(self._string_terms)) * phases_per_layer_phase : (d+1 +len(self._hopping_terms) + len(self._string_terms)) * phases_per_layer_phase]
                                circuit.append(var_terms_pair[d], qargs=all_qubits)
                                for qbit, phase_param in zip(fermion_qubits, phase_params_per_d_phase):
                                    circuit.rz(phase_param, qbit)

                            for pair_term_c in var_terms_pair[n_rest_string:]:
                                circuit.append(pair_term_c, qargs=all_qubits)
                            phase_params_per_d_phase_last = phase_params[ len(phase_params) - self._lattice.nsites *2 :]
                            for qbit, phase_param in zip(fermion_qubits, phase_params_per_d_phase_last):
                                circuit.rz(phase_param, qbit)
                            
        return circuit