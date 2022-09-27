from .observables import *
from qiskit.quantum_info import Pauli
from qiskit.aqua.operators import WeightedPauliOperator


def gauge_operator(site, lattice, params, boundary_cond=None, output='qutip'):
    """
    Implements the local gauge operator Gx on the given `site`  on `lattice` with given parameters `params`.

    Args:
        site (np.ndarray): The coordinates of the lattice site for which the gauge operator Gx should be constructed.
        lattice (Lattice): The lattice object
        params (dict): The dictionary of parameters.
            Must contain
             S (float): the Spin truncation value of the Quantum Link Model
             e (float): the gauge coupling parameter
            Optional
             theta(float): Topological theta-term corresponding to a constant electric background field.
                    Default value of theta is 0, if not provided.
        boundary_cond (dict): A dictonary specifying the boundary conditions on the lattice in
            the format:
                (site_index, direction) : boundary_link_value.
            For example the entry `(3,0) : 0.5` would mean that the link in direction `0` at lattice site with
            index `3` is constrained to have the flux value `0.5`.
        output (str): The desired output type, must be one of ['qiskit','qutip','matrix','spmatrix']

    Returns:
        qiskit.aqua.operators.WeightedPauliOperator or qutip.Qobj
    """
    # Extract parameters
    e = params['e']
    S = params['S']
    #perché s+1? 
    allowed_bc_values = np.arange(-S, S+1)
    # Extract non-zero theta angle if existent. Otherwise set to 0.
    if 'theta' in params.keys():
        theta = params['theta']
    else:
        theta = 0.

    # Initialize a charge offset value (-e b.c. of Dirac Sea interpretation)
    charge_offset = -e
    
    # Initialize the final return operator
    gaugeop = (site_charge(site, lattice, e) @ link_id(S, lattice)).to_qubit_operator(output=output)

    ############################################
    # Iterate over the lattice dimensions
    for i in range(lattice.ndim):
        # Check if site is on positive side boundary across the given dimension
        #cioé se in quel site in quella direction c'é un boundary 
        if lattice.is_boundary_along(site, dim=i, direction='positive'):
            # If along positive direction, require positive direction B.C.
            #ricorda bond_cond(1,0)=0.5
    
            boundary_flux = boundary_cond[lattice.site_index(site), i]
            assert boundary_flux in allowed_bc_values, "Boundary condition ({}, {}) = {} is not compatible" \
                                                       "with spin value S={}.".format(lattice.site_index(site),
                                                                                      i,
                                                                                      boundary_flux,
                                                                                      S)
            charge_offset -= e * (boundary_flux + theta)
            print(e * (boundary_flux + theta))

            rhs_edge_i = None
        else:
            
            # Else, initialize the operator
            #initialise the edge ([site_index, direction])
            rhs_edge_i = np.array([lattice.site_index(site), i])
            
            # Set up the outflux operator
            E_rhs_i = E(rhs_edge_i, S, lattice, e=e, theta=theta)
 
#Nota che con l'aumento del site_index può essere che si trovi sul lato sinistro della riga sopra 
        # Check if site is on negative side boundary across the given dimension
        if lattice.is_boundary_along(site, dim=i, direction='negative'):
            
            # If along negative direction, require negative direction B.C. Else, initialize the operator
            #cioé se siamo sul bordo a sinistra, allora prendo la bc
            # prende il valora di Elhs= bc (0,0)
            boundary_flux = boundary_cond[lattice.site_index(site), i]
            
            assert boundary_flux in allowed_bc_values, "Boundary condition ({}, {}) = {} is not compatible" \
                                                       "with spin value S={}.".format(lattice.site_index(site),
                                                                                      i,
                                                                                      boundary_flux,
                                                                                      S)
            charge_offset += e * (boundary_flux + theta)
            print(e * (boundary_flux + theta))
            
            lhs_edge_i = None
        else:
            
            # Else, initialize the operator
            prev_site = np.copy(site)
            prev_site[i] -= 1
            # project to treat the periodic case: (non-periodic is treated with boundary above)
            prev_site = lattice.project(prev_site)
            lhs_edge_i = np.array([lattice.site_index(prev_site), i])
            # Set up the influx operator:
            #peché deve essere il campo ceh va nella stessa direzione i, ma per il sito precedetnte x-i
            #vedi eq. 2.68 Simon 
            E_lhs_i = E(lhs_edge_i, S, lattice, e=e, theta=theta)

        # Calculate the delta flux operator through site x along dimension i
        if lhs_edge_i is None:
            
            delta_i = -E_rhs_i
            
        elif rhs_edge_i is None:
            delta_i = E_lhs_i
        else:
            delta_i = E_lhs_i - E_rhs_i

        # Add the flux operator to the charge:
        gaugeop += (fermion_id(lattice) @ delta_i).to_qubit_operator(output=output)
    ###############################################
#
    # Correct for boundary conditions with the charge offset:
    #ma se non si é al B.c perché chargeoffet non é = 0? 
    
    charge_offset_operator = (charge_offset * fermion_id(lattice)) @ link_id(S, lattice)
    return gaugeop + charge_offset_operator.to_qubit_operator(output=output)


def gauss_law_regularizer(lattice, params, lam=1., boundary_cond=None, output='qutip'):
    """
        This function contructs the Gauss law regularizer (enforces only gauge invariant states)

        Args:
            lattice (Lattice): The lattice object
            params (dict): The dictionary of parameters. Must contain
                 S (float): the Spin truncation value of the Quantum Link Model
                 e (float): the gauge coupling parameter
            lam (float): This parameter regulates the strength of the gauge regularizing term
            boundary_cond (dict): A dictonary specifying the boundary conditions on the lattice in
                the format:
                    (site_index, direction) : boundary_link_value.
                For example the entry `(3,0) : 0.5` would mean that the link in direction `0` at lattice site with
                index `3` is constrained to have the flux value `0.5`.
            output (str): The desired output type, must be one of ['qiskit','qutip','matrix','spmatrix']

        Returns:
            qiskit.aqua.operators.WeightedPauliOperator or qutip.Qobj
        """

    # Check if boundary conditions are present:
    if lattice.boundary_cond is not 'periodic' and boundary_cond is None:
        raise UserWarning('Please provide boundary conditions for a finite lattice. Provide them in '
                          'dictionary format with the convention: '
                          '\nkey:    (site_index, dimension)'
                          '\nvalue:  boundary_condition_value')

    summands = []

    # Iterate over sites and add the gauge regulator terms per site
    for site in lattice.sites:
        Gx = gauge_operator(site, lattice, params, boundary_cond, output)

        ##### QUICK FIX ### TODO: IMPLEMENT * FOR AQUA OPERATORS
        if output == 'qiskit':
            lam_identity = WeightedPauliOperator(paulis=[[lam, Pauli.from_label('I' * Gx.num_qubits)]])
            summands.append(lam_identity * Gx * Gx)
        else:
            summands.append(lam * Gx * Gx)

    return operator_sum(summands)
