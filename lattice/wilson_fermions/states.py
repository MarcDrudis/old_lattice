from .observables import *
from ..operators.qiskit_aqua_operator_utils import operator_sum
from ..operators.spin_operators import embed_state
import qutip as qt
import matplotlib.pyplot as plt
from qiskit.aqua.components.initial_states import InitialState
from qiskit import QuantumRegister, QuantumCircuit, execute, Aer
import scipy

################################################################################################
# 1. Set up plotting capabilities for states
################################################################################################


# Functionality for plotting a state.
def mass_charge_distribution(state, lattice, rep=dirac, S=0):
    """
    Measures the expectation values of site masses and site charges across the lattice for a given state
    and plots them to the console.
    Note: currently only supports qutip states.

    Args:
        state (qutip.Qobj): The state of the system as a qutip.Qobj
        lattice (Lattice): The lattice object representing the lattice system on which the state is given
        rep (dict): The representation of the Clifford algebra to be used
        S (float): The spin truncation value of the Quantum Link Model. Must be a non-negative integer or half-integer

    Returns:
        mass (np.ndarray): A numpy array containing the expectation values of the mass operator `site_mass` for
            the given `state` on each site of the given `lattice`.
        charge (np.ndarray): A numpy array containing the expectation values of the charge operator `site_charge` for
            the given `state` on each site of the given `lattice`.
    """
    mx = lambda x: site_mass(x, lattice, rep, S=S).to_qubit_operator(output='qutip') + 1
    qx = lambda x: site_charge(x, lattice, S=S).to_qubit_operator(output='qutip') - 1

    mass   = [qt.expect(mx(site), state) for site in lattice.sites]
    charge = [qt.expect(qx(site), state) for site in lattice.sites]

    # plt.plot(init_mass, 'o--', label='initial state')
    plt.plot(mass, 'o--', label='mass', alpha=0.8)
    plt.plot(charge, 'o--', label='charge', alpha=0.8)
    plt.hlines(1, -1, lattice.nsites, linestyle='dashed', color='grey', alpha=0.5, label='fermi sea level')
    plt.hlines(0, -1, lattice.nsites, color='grey', alpha=0.5)
    plt.legend(loc='lower right');
    plt.title('Mass-Charge distribution')
    plt.xlabel('Site index')
    plt.xticks(np.arange(0, lattice.nsites))
    plt.xlim(-0.5, lattice.nsites - 0.5)
    plt.ylabel('Expectation');
    plt.ylim(-2.1, 2.1);

    return mass, charge

################################################################################################
# 2. Set up state building blocks (for the dirac representation)
# TODO: Generalize to other representations
################################################################################################


occ = qt.basis(2,1)
unocc = qt.basis(2,0)
# praticamente qt.basis(2,1) rappresenta ket(1) mentre qt.basis(1,0)=ket(0)
#Attenzione sono scritti nella physics convention !!!!!! La particle sta a sinistra (nel primo qubit
# nel physics convention )
particle = [unocc, unocc]
vacuum   = [occ, unocc]
antiparticle = [occ, occ]
twoparticle = [unocc, occ]


def construct_dirac_state(state_string, output='qutip'):
    """Construct a state of the dirac field on the lattice in dirac representation.

    The string is mapped as follows:
        '.'  ---> vacuum
        'p'  ---> particle
        'a'  ---> antiparticle
        'b'  ---> both (particle & antipartcle)

    E.g. 'p..a' would correspond to the state:  particle @ vacuum @ vacuum @ antiparticle
    (if dirac representation is used)
    """

    # 1. parse
    for char in state_string:
        assert char in '.pab', "Characters of state_string must be one of ['.', 'p', 'a', 'b']"

    mapping = {
        'p': particle,
        '.': vacuum,
        'a': antiparticle,
        'b': twoparticle
    }

    # 2. construct the state
    # Construct the list of e.g. [ particle, vacuum, ... ]
    state_list = list(map(mapping.__getitem__, state_string))

    # Sum them together, i.e. [particle, vacuum, ... ] ---> particle + vacuum + ...
    state_list = operator_sum(state_list)
#operator sum fa passare state_list=[[occ,unocc],[occ,unocc]] ---> state_list=[occ,unocc,occ,unocc]
#prima state_list[0][0]=occ, adesso state_list[0]=occ
    # Tensor together the product - flipping [::-1] needed because of qiskit ordering of tensor product
    # se avessimo fatto questo lavoro prima di operator sum andava bene 
    final_state = qt.tensor(state_list[::-1])

    if output == 'qutip':
        return final_state
    elif output == 'vector':
        return final_state.full().flatten()
    else:
        raise UserWarning("Output must be one of ['qutip', 'vector'], not '{}'".format(output))

################################################################################################
# 3. Set up basic constructors to initialize fermionic registers in `Dirac` representation
#    eigenstates
# TODO: Generalize to other representations
################################################################################################

class DiracState(InitialState):
    """An initializer for the bare vacuum state of the 1D free Schwinger Model in Wilson Form"""

    CONFIGURATION = {
        'name': 'bare_vacuum',
        'description': 'Bare vacuum state for the 1D Schwinger Model in Wilson formalism',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'bare_vacuum_state_schema',
            'type': 'object',
            'properties': {
            },
            'additionalProperties': False
        }
    }

    def __init__(self, state_string, ms_list=[], S=0.):
        """Constructor.

        Args:
            lattice (Lattice): The lattice object

            state_string (str): The string specifying the state. Must consist of ['p', 'a', 'b', '.'].

        """
        super().__init__()
        self._num_sites = len(state_string)
        self._num_edges = len(ms_list)  # assuming 2 component wilson fermions
        self._state_string = state_string
        self._ms_list = ms_list
        self._S = S

    def _construct_fermionic_subcircuit(self, mode, register=None):
        """
            Construct the statevector of desired initial fermion state.

            Args:
            mode (string): `vector` or `circuit`. The `vector` mode produces the vector.
                            While the `circuit` constructs the quantum circuit corresponding that
                            vector.
            register (QuantumRegister): register for circuit construction.

            Returns:
            QuantumCircuit or numpy.ndarray: statevector.

            Raises:
            ValueError: when mode is not 'vector' or 'circuit'.
        """
        if mode == 'vector':
            return construct_dirac_state(state_string=self._state_string).full()
        elif mode == 'qutip':
            return construct_dirac_state(state_string=self._state_string)
        elif mode == 'circuit':

            for char in self._state_string:
                assert char in '.pab', "Characters of state_string must be one of ['.', 'p', 'a', 'b']"
#nella convenzione fisica!! infatti a sinistra c'é 1 per il vacuum cioé qunado
#la particle non c'é 
            mapping = {
                'p': '00',
                '.': '10',
                'a': '11',
                'b': '01'
            }

            # 2. construct the state
            # Construct the list of e.g. [ particle, vacuum, ... ]
            binary_state_list = list(map(mapping.__getitem__, self._state_string))
            binary_state_list = operator_sum(binary_state_list)

            if register is None:
                register = QuantumRegister(self._num_sites*2, name='fermionic')
            quantum_circuit = QuantumCircuit(register)
            for n in range(self._num_sites*2):
                if binary_state_list[n] == '1':
                    quantum_circuit.x(register[n])
            return quantum_circuit
        else:
            raise ValueError('Mode should be either "vector" or "circuit"')

    def _construct_spin_subcircuit(self, mode, register=None):
        """
            Construct the statevector of desired initial spin state.

            Args:
            mode (string): `vector` or `circuit`. The `vector` mode produces the vector.
                            While the `circuit` constructs the quantum circuit corresponding that
                            vector.
            register (QuantumRegister): register for circuit construction.

            Returns:
            QuantumCircuit or numpy.ndarray: statevector.

            Raises:
            ValueError: when mode is not 'vector' or 'circuit'.
        """
        return embedded_spin_register_state(self._ms_list, self._S, output=mode)

    def construct_circuit(self, mode, register=None):
        """
            Construct the statevector of desired initial fermion state.

            Args:
            mode (string): `vector` or `circuit`. The `vector` mode produces the vector.
                            While the `circuit` constructs the quantum circuit corresponding that
                            vector.
            register (QuantumRegister): register for circuit construction.

            Returns:
            QuantumCircuit or numpy.ndarray: statevector.

            Raises:
            ValueError: when mode is not 'vector' or 'circuit'.
        """
        if mode == 'vector':
            # TODO
            if self._S == 0:
                assert len(self._ms_list) == 0, 'Nonzero `ms_list` but `S = 0`.'
                return self._construct_fermionic_subcircuit(mode='qutip').full()
            else:
                return qt.tensor([self._construct_fermionic_subcircuit(mode='qutip'),
                                  self._construct_spin_subcircuit(mode='qutip')][::-1]).full()

        elif mode == 'qutip':
            if self._S == 0:
                assert len(self._ms_list) == 0, 'Nonzero `ms_list` but `S = 0`.'
                return self._construct_fermionic_subcircuit(mode='qutip')
            else:
                return qt.tensor([self._construct_fermionic_subcircuit(mode='qutip'),
                                  self._construct_spin_subcircuit(mode='qutip')][::-1])

        elif mode == 'circuit':

            circuit = self._construct_fermionic_subcircuit(mode='circuit')
            if self._S == 0:
                assert len(self._ms_list) == 0, 'Nonzero `ms_list` but `S = 0`.'
            else:
                ### attenzione se si fa += non viene aggiornato circuit.qubits
                # cioé il numero del registe 
                #old:circuit += self._construct_spin_subcircuit(mode='circuit')
                circuit =self._construct_fermionic_subcircuit(mode='circuit') + self._construct_spin_subcircuit(mode='circuit')

            if register is not None:
                instruction = circuit.to_instruction()
                instruction.name = 'state_init'
                all_qubits = [qbit for qbit in register]
                circuit = QuantumCircuit(register)
                circuit.append(instruction, qargs=all_qubits)

            return circuit

#ha solo fermions!!!! 
class BareDiracVacuum(DiracState):
    """An initializer for the bare vacuum state of the 1D Schwinger Model in Wilson Form"""

    CONFIGURATION = {
        'name': 'bare_vacuum',
        'description': 'Bare vacuum state for the 1D Schwinger Model in Wilson formalism',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'bare_vacuum_state_schema',
            'type': 'object',
            'properties': {
            },
            'additionalProperties': False
        }
    }

    def __init__(self, num_sites):
        """Constructor.

        Args:
            num_sites (int): number of lattice sites
        """
        super().__init__(state_string='.'*num_sites)


################################################################################################
# 4. Set up basic building blocks for initializing spin registers for
#    building up the link states
################################################################################################


def spin_state(ms, S, output='qutip'):
    """
    Constructs the state |S, ms> for a spin S system in qutip.

    Args:
        ms (float): The 'magnetic' quantum number of the spin state. Must be an integer or half
            integer number in the closed interval [-S, -S+1, ... , S-1, S].

        S (float): The 'spin' quantum number S. Must be a non-negative integer or half-integer.

        output (str): The output format. Must be one of ['qutip', 'vector', 'index']

    Returns:
        qutip.Qobj or np.ndarray or int:
            The corresponding state |S, ms> as a qutip state-vector or numpy array.
            The ordering is as follows:
            [0, 0, ..., 0, 1] = |S, -S>
            [1, 0, 0, ..., 0] == |S, S>
    """
    S = float(S)
    ms = float(ms)

    # 1. Parse input values of S and ms for valid quantum numbers.
    if (scipy.fix(2 * S) != 2 * S) or (S < 0):
        raise TypeError('S must be a non-negative integer or half-integer')

    # Conditions for ms:
    ms_half_or_int = (scipy.fix(2 * ms) == 2 * ms)
    ms_in_range = (-S <= ms <= S)
    ms_and_S_compatible = (S.is_integer() == ms.is_integer())
    # Throw error if any of the conditions is not True
    if not ms_half_or_int or not ms_in_range or not ms_and_S_compatible:
        raise TypeError("ms value '{}' invalid for S={}. `ms` must be an integer or half integer number in the "
                        "closed interval [-S, -S+1, ... , S-1, S].".format(ms, S))

    # 2. Construct and return the qutip state
    dim_S = int(2 * S + 1)
    state_index = ms + S
    if output == 'qutip':
        return qt.basis(dim_S, int(2 * S - state_index))
    elif output == 'vector':
        state_vector = np.zeros(dim_S, dtype=complex)
        state_vector[int(2 * S - state_index)] = 1.
        return state_vector
    elif output == 'index':
        return int(2 * S - state_index)
        # instead, for spin embed_location='lower': return int(state_index)
    else:
        raise UserWarning("Output must be one of ['qutip', 'vector'], not '{}'".format(output))


def spin_register_state(ms_list, S):
    """
    Constructs the state corresponding to (S, ms1) @ (S, ms2) @ ... @ (S, msN) on a spin register
    of length N in qutip.
    The ordering of the tensor product is given by the corresponding ordering in qiskit. I.e. the
    reversed ordering of the qutip tensor product.

    Args:
        ms_list (list[float]): A list of the 'magnetic' quantum numbers of the spin states in the
            spin register. Each element must be be an integer or half integer number in the closed
            interval [-S, -S+1, ... , S-1, S].
        S (float): The 'spin' quantum number S. Must be a non-negative integer or half-integer.

    Returns:
        qutip.Qobj:
            The corresponding state (S, ms1) @ (S, ms2) @ ... @ (S, msN)  as a qutip state-vector.
            The ordering of the tensor product is given by the qiskit ordering.
            I.e. [LSB, ... , MSB]
    """

    return qt.tensor([spin_state(ms, S) for ms in ms_list][::-1])  # [::-1] b.c. of qiskit ordering


def embedded_spin_state(ms, S, nqubits=None, output='vector', qreg=None):
    """
    Constructs and embeds the spin state |S, ms> for a spin S system in a statevector for `nqubits` qubits.

    Args:
        ms (float): The 'magnetic' quantum number of the spin state. Must be an integer or half
            integer number in the closed interval [-S, -S+1, ... , S-1, S].

        S (float): The 'spin' quantum number S. Must be a non-negative integer or half-integer.

        nqubits (int or None): The number of qubits used to represent the state. I.e. the spin state vector will be
            embedded into a vector of dimension 2^nqubits.
            If `None` the number of qubits required is inferred automatically.

        qreg (qiskit.QuantumRegister or list): The quantum register on which the circuit should be built.
            Only required for output == 'circuit'.

        output (str): The desired output. Must be one of ['vector', 'qutip', 'circuit']

    Returns:
        qutip.Qobj or np.ndarray:
            The corresponding state |S, ms> as a qutip state-vector or numpy array.
            The ordering is as follows:
            [0, 0, ..., 0, 1] = |S, -S>
            [1, 0, 0, ..., 0] == |S, S>
    """

    # If number of qubits is given, embed state in that number of qubits. Else, infer the
    # least possible number of qubits required.
    if nqubits is None:
        nqubits = int(np.ceil(np.log2(2 * S + 1)))

    if output == 'vector':
        spin_state_vector = spin_state(ms, S, output='vector')
        embedded_state_vector = embed_state(spin_state_vector, nqubits=nqubits)
        return embedded_state_vector

    elif output == 'qutip':
        spin_state_vector = spin_state(ms, S, output='vector')
        embedded_state_vector = embed_state(spin_state_vector, nqubits=nqubits)
        return qt.Qobj(embedded_state_vector, dims=[[2]*nqubits, [1]*nqubits])

    elif output == 'circuit':
        if qreg is None:
            qreg = QuantumRegister(nqubits)
        assert isinstance(qreg, QuantumRegister), 'Need to provide a valid QuantumRegister for `circuit` output.'

        state_index = spin_state(ms, S, output='index')
        binary_repr = np.binary_repr(state_index, width=nqubits)  # ordering: MSB ... LSB
        circuit = QuantumCircuit(qreg)

        # flip binary_repr because in qiskit first (topmost) qubit is the LSB
        for binary, qubit in zip(binary_repr[::-1], qreg):
            # encode from 'bottom up', i.e. embedded spin state [1.0, 0.0] goes to qiskit state
            # [0.0, 1.0] to match with my qutip implementation.
            if binary == '1':  # instead, for spin embed_location='lower': if binary == '0'
                circuit.x(qubit)
        return circuit

    else:
        raise UserWarning("Output must be one of ['qutip', 'vector'], not '{}'".format(output))


def embedded_spin_register_state(ms_list, S, nqubits=None, output='vector'):
    """
    Constructs the state corresponding to (S, ms1) @ (S, ms2) @ ... @ (S, msN) on a spin register
    of length `len(ms_list)` and embeds it into qubits.
    The ordering of the tensor product is given by the corresponding ordering in qiskit. I.e. the
    reversed ordering of the qutip tensor product.

    Args:
        ms_list (list[float]): A list of the 'magnetic' quantum numbers of the spin states in the
            spin register. Each element must be be an integer or half integer number in the closed
            interval [-S, -S+1, ... , S-1, S].
            The ordering of `ms_list` to the register is [LSB, ... , MSB] (qiskit ordering).

        S (float): The 'spin' quantum number S. Must be a non-negative integer or half-integer.

        nqubits (int or None): The number of qubits into which a single spin in the register (i.e. (S, ms) )
            is to be embedded.

        output (str): Must be one of ['vector', 'circuit', 'qutip']

    Returns:
        np.ndarray or qutip.Qobj or qiskit.QuantumCircuit:
            The corresponding state (S, ms1) @ (S, ms2) @ ... @ (S, msN)  as a state-vector in
            the given output format.
            The ordering of the tensor product is given by the qiskit ordering.
            I.e. [LSB, ... , MSB]
    """
    if nqubits is None:
        nqubits = int(np.ceil(np.log2(2 * S + 1)))

    if output == 'vector':
        return qt.tensor([embedded_spin_state(ms, S, nqubits, output='qutip') for ms in ms_list][::-1]).full().flatten()

    elif output == 'qutip':
        return qt.tensor([embedded_spin_state(ms, S, nqubits, output='qutip') for ms in ms_list][::-1])
        # [::-1] b.c. of qiskit ordering

    elif output =='circuit':
        # initialize an empty list to save the construction circuits of all individual spin states
        individual_spins = []

        # go through the list of spins and construct each state individually
        for i, ms in enumerate(ms_list):
            # set up and name an appropriate register
            spin_i = QuantumRegister(nqubits, name='spin({:.1f}){}'.format(S, i))
            # set up the state preparation circuit
            circ_i = embedded_spin_state(ms, S, nqubits, output='circuit', qreg=spin_i)
            individual_spins.append(circ_i)
        # Add together the qiskit circuits for the individual spins in the spin register
        return operator_sum(individual_spins)

    else:
        raise UserWarning("Output must be one of ['circuit', 'qutip', 'vector'], not '{}'".format(output))


################################################################################################
# 5. Useful extra functions (e.g. set up a gauge invariant state in 1D)
################################################################################################


def gauge_inv_state_1d(state_string, lhs, S):
    """
    Constructs a gauge invariant (fermion register, spin register) state on a 1D lattice in
    qutip in Dirac representation.


    Args:
        state_string (str): A string consisting of characters [p, ., b, a] only.
            The string is mapped as follows:
            '.'  ---> vacuum
            'p'  ---> particle
            'a'  ---> antiparticle
            'b'  ---> both (particle & antipartcle)

            E.g. 'p..a' would correspond to the state:  particle @ vacuum @ vacuum @ antiparticle
            (if dirac representation is used)

        lhs (float): The spin value on the periodic edge in 1D

    Returns:
        qutip.Qobj
    """
    link_config = np.zeros(len(state_string))
    link_config[-1] = lhs

    delta_flux = {
        'p': 1,
        'a': -1,
        '.': 0,
        'b': 0
    }

    for i, symbol in enumerate(state_string):
        if i == len(state_string) - 1:
            assert link_config[i - 1] + delta_flux[symbol] == lhs, 'cannot be gauss inv. under pbc'
        link_config[i] = link_config[i - 1] + delta_flux[symbol]

    return qt.tensor(spin_register_state(link_config, S), construct_dirac_state(state_string))
        #again [::-1] order b.c. of qiskit ordering

################################################################################################
# 6. Functionality to decompose states into configurations
################################################################################################


def int_to_config_dirac(encoding_int, fermi_len, spin_len, S, theta, spin_encoding):
    """
    Turns an integer (which corresponds to a computational basis state of the register) into a
    configuration of a Fermion-Spin model where the fermionic register has length `fermi_len * 2`
    (with two components per site, `fermi_len = number of sites`) and
    the spin register has length `spin_len` and consists of spin `S` systems.
    Note: This function assumes the following ordering of registers: (fermionic register, spin register).
    I.e. the LSB is a fermionic-register element and the MSB a spinS-register element

    WARNING: This decomposition is only meaningful, if using the dirac representation where
    gamma0 is diagonal.

    WARNING: For the logarithmic encoding of the spin system, this decomposition works only if the spin state
    is embedded in the UPPER part of the spin statevector.

    Args:
        encoding_int (int):
        fermi_len (int):  The length of the fermionic register.
        spin_len (int): The length of the spin-register
        S (float): The spin value `S` of the spin-register. Must be a non-negative integer or half-integer
        theta (float): The vacuum-angle `theta` corresponding to a uniform electric background field.
        spin_encoding (str): Specify the encoding of the spin state. Must be one of ['lin_encoding','log_encoding'].

    Returns:
        fermi_config (list): The fermionic configuration in terms of 'p', '.', 'b', 'a' corresponding to
            particle, vacuum, both, antiparticle
        spin_config (list): The spin configuration in terms of the ms spin value
    """
    dimfermi = 4
    # `dimfermi**fermi_len` corresponds to the length of the statevector representing the fermionic d.o.f.
    if spin_encoding == 'lin_encoding':
        dimS = int(2 * S + 1)
    elif spin_encoding == 'log_encoding':
        dimS = 1 << int(np.ceil(np.log2(2 * S + 1)))  # corresponds to 2 ** num_qubits
    else:
        raise UserWarning("`spin_encoding` must be one of ['log_encoding', 'lin_encoding'].")

    # determine the fermion states:
    fermi_part = encoding_int % (1 << 2 * fermi_len)  # encoding_int mod 4**fermi_len (= dimfermi**fermi_len)
    fermi_coeffs = np.zeros(fermi_len, dtype=int)
    for i in range(fermi_len):
        fermi_coeffs[i] = fermi_part % dimfermi
        fermi_part /= dimfermi

    # determine the spins:
    spin_part = encoding_int >> 2 * fermi_len  # encoding_int / 4 ** fermi_len (= dimfermi**fermi_len)

    spin_coeffs = np.zeros(spin_len, dtype=int)
    for i in range(spin_len):
        spin_coeffs[i] = spin_part % dimS
        spin_part /= dimS

    # map to configuration:
    fermi_mapping = {0: 'p', 1: '.', 2: 'b', 3: 'a'}
    spin_mapping = lambda m: S - m + theta

    fermi_config = "".join(map(fermi_mapping.__getitem__, fermi_coeffs))
    spin_config = list(map(spin_mapping, spin_coeffs))

    return fermi_config, spin_config


def state_decompose_dirac(state, fermi_len, spin_len, S, theta=0.,
                          spin_encoding='log_encoding', thres=0.05, output='other'):
    """
    Decomposes a general qutip state into the computational basis states with probability > `thres`
    and returns them as transformed to configurations with the respective occurrence probabilities.

    WARNING: This decomposition is only meaningful, if using the dirac representation where
    gamma0 is diagonal.

    WARNING: For the logarithmic encoding of the spin system, this decomposition works only if the spin state
    is embedded in the UPPER part of the spin statevector.

    Args:
        state (qutip.Qobj or numpy.ndarray): A qutip state of a (fermion, spin)-register for a Wilson Fermion model
        fermi_len (int):  The length of the fermionic register.
        spin_len (int): The length of the spin-register
        S (float): The spin value `S` of the spin-register. Must be a non-negative integer or half-integer
        theta (float): The vacuum-angle `theta` corresponding to a uniform electric background field.
        spin_encoding (str): Specify the encoding of the spin state. Must be one of ['lin_encoding','log_encoding'].
        thres (float): The threshold. All state components with amplitude squared > thres will be returned.
        output (str): The desired output type

    Returns:

    """
    if type(state) is type(qt.Qobj()):
        state_vector = state.full()
    elif type(state) is type(np.array(0)):
        state_vector = state
    elif type(state) is type(QuantumCircuit()):
        state_vector = execute(state,backend=Aer.get_backend("statevector_simulator"), shots=1).result().get_statevector()
    else:
        raise UserWarning("Type of `state` must be one ['qt.Qobj', 'np.array', 'QunantumCircuit'], not {}".format(type(state)))
    # Calcualate the probabilities = absolute value squared of wavefunc coefficients
    config_prob = np.square(np.abs(state_vector.flatten()))

    # Extract relevant state components and their probabilities
    relevant_components = np.where(config_prob > thres)[0]
    relevant_prob = config_prob[relevant_components]

    # Sort according to descending probability
    sorting = np.argsort(relevant_prob)[::-1]
    relevant_prob = relevant_prob[sorting]
    relevant_components = relevant_components[sorting]

    # Transform the state components to configurations
    to_config = lambda i: int_to_config_dirac(encoding_int=i,
                                              fermi_len=fermi_len,
                                              spin_len=spin_len,
                                              S=S,
                                              theta=theta,
                                              spin_encoding=spin_encoding)
    relevant_configs = list(map(to_config, relevant_components))

    if output == 'dict':
        return {config[0]: prob for config, prob in zip(relevant_configs, relevant_prob)}

    return relevant_configs, relevant_prob
