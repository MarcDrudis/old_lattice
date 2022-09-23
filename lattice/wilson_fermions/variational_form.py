from qiskit.aqua.components.variational_forms import VariationalForm
from qiskit import QuantumRegister, QuantumCircuit
import numpy as np
import qutip as qt

from lattice.operators.qiskit_aqua_operator_utils import operator_sum
from lattice.wilson_fermions.basic_operators import psi, psidag, U, Udag, standard_basis, fermion_id, link_id
from lattice import Lattice
from lattice.wilson_fermions.qiskit_utils import construct_trotter_step

__all__ = ['WilsonLGT']


################################################################################################
# 1. Set up functionality to construct the hopping terms form the lattice
################################################################################################

def hopping_term(edge, lattice, S, rep, hopp_coeff=1., output='qiskit'):
    """
    Generates the hopping term along the given `edge` in `lattice` scaled by the `hopp_coeff`.
    """

    # 1. Extract the gama matrices from the representation of the Clifford algebra
    gamma = [val for val in rep.values()]

    # 2. Extract the site and direction of the edge along which hopping occurs:
    site = lattice.site_vector(edge[0])
    #ricorda ceh edge é definito tipo [0,1], con sito e direzione 
    hopping_dim = edge[1]

    # 3. Generate the hopping term as a sum over the fermionic (spinor) components
    summands = []
    # 3.1 Check for boundary sites
    # Treat open and closed boundary conditions (skip term if edge is at boundary)
    is_at_boundary = lattice.is_boundary_along(site, hopping_dim, direction='positive')
    if is_at_boundary:
        # print('encountered boundary at {}, {}'.format(site, j))
        raise UserWarning('The given `site` and `hopping_dim` combination goes outside the lattice.')

    # 3.2Get the edge along which hopping takes place and the next site
    next_site = lattice.project(site + standard_basis(hopping_dim, lattice.ndim))

    # 3.3 Construct the product of matrices -1j * gamma0 * gammaj for that direction
    gamma_mix = -1j * gamma[0] @ gamma[hopping_dim + 1]
    # print('gamma_mix_{}:\n'.format(j+1), gamma_mix_j.full())

    # 3.4 Sum over all spinor components:
    for alpha in range(2):
        for beta in range(2):
            # Skip cases with zero coefficients
            if gamma_mix[alpha, beta] == 0:
                # print('skipped ab {}, {}'.format(alpha, beta))
                continue

            # Generate the fermionic hopping terms
            bwd_hopp = (hopp_coeff
                        * psidag(site, alpha, lattice)
                        * gamma_mix[alpha, beta]
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
    return operator_sum(summands)


################################################################################################
# 2. Set up the variational form that is based on the lattice hopping terms to conserve
#    the Gauss law (gauge-invariance)
################################################################################################

class WilsonLGT(VariationalForm):
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

    def __init__(self, lattice, S, rep, depth=1, initial_state=None):
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
        ##### Nota ceh l'ho toooolto !!
        #####self.validate(locals())

        super().__init__()
        self._num_qubits = lattice.nsites * 2 + lattice.nedges * int(np.ceil(np.log2(2 * S + 1)))
        self._depth = depth
        self._lattice = lattice
        self._S = S
        self._rep = rep
        self._initial_state = initial_state
        # one parameter for every edge (along which hopp. occurs)
        self._num_parameters = (lattice.nedges + (self.num_qubits+1)//2)* depth
        self._bounds = [(-np.pi, np.pi)] * self._num_parameters #TODO: Adapt

        # Set up the hopping terms along the edges
        self._construct_hopping_operators()
        self._matrix_hopping_operators = []

    @property
    def hopper_correction(self):
        """ The correction for the norms of the hopping terms to make sure that all hopping strengths can be
        reached within the boundaries of (-np.pi, np.pi)"""
        return np.sqrt((self._S + 1) * self._S) * 2

    def _construct_hopping_operators(self):
        self._hopping_terms = []
        for edge in self._lattice.edges:
            hopper_edge = hopping_term(edge, lattice=self._lattice, S=self._S, rep=self._rep, output='qiskit')
            hopper_edge.chop()
            self._hopping_terms.append(hopper_edge)

    def _construct_parametrized_evolution_circuits(self):
        # TODO
        pass

    def construct_circuit(self, parameters, q=None):
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

        # 2 Set up the quantum register
        if q is None:
            q = QuantumRegister(self._num_qubits, name='q')
        all_qubits = [qbit for qbit in q]

        # 3 Set up and initialize the circuit
        if self._initial_state is not None:
            circuit = self._initial_state.construct_circuit('circuit', q)
        else:
            circuit = QuantumCircuit(q)

        # 4 Build up the variational form with the parameters
        hoppings_per_layer = len(self._hopping_terms)
        phases_per_layer = self.num_qubits
        params_per_layer = hoppings_per_layer + phases_per_layer

        # 4.1 Iterate over depth
        for d in range(self._depth):
            # 4.2 For each depth-block add parametrized exponentials of hopping terms
            hopping_params = parameters[d * params_per_layer: d * params_per_layer + hoppings_per_layer]
            phase_params = parameters[d * params_per_layer + hoppings_per_layer: (d + 1) * params_per_layer]

            circuit.barrier(q)
            for edge, hopper, parameter in zip(self._lattice.edges,
                                                     self._hopping_terms,
                                                     hopping_params):
                   ##nota che var_term é come trotter però per osni   
                   # hopper é l'hamiltonian hopper term ad un certo edge,
                   # Invece del tempo mette parameter*self.hopper_correction
                   # Hopper_correction é solo una normalizzazion factor, parameters lo prende 
                   # da hopping_params, i parameters si danno dentro                                
                var_term = construct_trotter_step(hopper,
                                                  parameter * self.hopper_correction,
                                                  name='h_{}({})'.format(edge, parameter))
                circuit.append(var_term, qargs=all_qubits)

            for qbit, phase_param in zip(all_qubits[::2], phase_params):
                circuit.u1(phase_param, qbit)

        return circuit

    def _eval_matrix_varform(self, parameters, init_state):
        # 1. If hopping operators are not yet built, build them
        if self._matrix_hopping_operators == []:
            self._matrix_hopping_operators = \
                [hopping_term(edge, self._lattice, self._S, self._rep, output='qutip')
                 for edge in self._lattice.edges]

        # 2. Define a function for sinqle qubit z rotations
        dim_S = int(2 * self._S + 1)

        def zterm(pos, comp, lattice, dim_S):
            assert 0 <= pos <= lattice.nsites - 1, 'pos out of bounds for lattice of size {}'.format(lattice.nsites)
            ops = [qt.identity(2)] * (2 * pos + comp) \
                  + [qt.sigmaz()] \
                  + [qt.identity(2)] * (2 * lattice.nsites - (2 * pos + comp) - 1) \
                  + [qt.identity(dim_S)] * lattice.nedges
            return qt.tensor(ops[::-1])

        # 3 Build up the variational form with the parameters
        hoppings_per_layer = self._lattice.nedges
        phases_per_layer = self.num_qubits
        params_per_layer = hoppings_per_layer + phases_per_layer

        # 3.1 Initialize the var_form matrix
        var_form = init_state.copy()

        # 4.1 Iterate over depth
        for d in range(self._depth):
            # 4.2 For each depth-block add parametrized exponentials of hopping terms
            hopping_params = parameters[d * params_per_layer: d * params_per_layer + hoppings_per_layer]
            phase_params = parameters[d * params_per_layer + hoppings_per_layer: (d + 1) * params_per_layer]

            for hopper, parameter in zip(self._matrix_hopping_operators,
                                               hopping_params):

                var_form = (1j * parameter * self.hopper_correction * hopper).expm() \
                             * var_form

            # 4.3 At the end add single qubit rotations for each site
            for pos, param in zip(np.arange(self._lattice.nsites), phase_params):
                # perform z-rotations on every 2nd qubit
                var_form = (1j * param * zterm(pos, 1, self._lattice, dim_S)).expm() \
                            * var_form

        return var_form
