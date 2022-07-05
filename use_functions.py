import numpy as np
import scipy.sparse as sp
import scipy.linalg as LA
from seeq.models.lattice import Regular3DLattice

def Landau_energy(l, q): 
    """
    Returns the energy of the l-th Landau level for a Harper-Hofstadter lattice with
    a magnetic flux per plaquette of the form 1/N. The implemented relation is obtained
    perturbatively in 1/N up to second order.
    """
    return -4 + (4*np.pi/q)*(l+0.5) - 0.5*(np.pi/q)**2 * (2*l**2+2*l+1)

def Harper_matrix(flux, nu, Lx, PBC_X=False):
    """
    Create matrix defining the Harper equations with periodic boundary conditions
    along Y.
    
    Inputs:
    
    flux ................................... Magnetic flux (float)
    nu ..................................... Potential phase (to be regarded as k_y) (float)
    PBC_X .................................. Indicates if there are periodic boundary conditions around 
                                             the 1D Harper chain (Boolean)
                                            
    Outputs:
    
    H ...................................... Harper matrix
    """
    # Initialize and set matrix entries
    H = np.zeros((Lx,Lx))
    for i in range(0, Lx):
        H[i, i] = 2*np.cos((2*np.pi*(i)*flux)-nu)
        H[i,(i-1)] = 1.
        if (i+1)<Lx: H[i,(i+1)] = 1.
            
    # Now, add hopping for periodic boundary conditions
    if PBC_X:
        H[-1,0], H[0, -1] = 1., 1.
        
    # ................................................................
    # CAVEAT: If using periodic boundary conditions, it is necessary that L is a multiple
    #         of q. Otherwise, the wavefunction would acquire a net phase after applying a 
    #         traslation along the whole chain, inconsistent with the periodicity of the
    #         boundary conditions
    # ................................................................


    return H


class HarperHofstadter(Regular3DLattice):
    """Lattice for the Harper-Hofstadter 2D model with nearest-neighbor hoppings.
    
    Parameters
    ----------
    Lx, Ly    -- Lattice length (number of vertices). Ly defaults to Lx
    Jx, Jy    -- hopping amplitudes (Jy defaults to Jx, Jx defaults to 1)
    flux      -- Magnetic flux per plaquette in units of flux quantum
    ω         -- Eigenfrequency of local modes
    r0        -- Location of the corner (defaults to (0,0,0))
    """

    def __init__(self, Lx, Ly=None, J=1.0, flux=0, ω=0, PBC=False, PBC_X=False,**kwdargs):
        
        if Ly is None:
            Ly = Lx
            
        # Boundary conditions
        def bc(coordinate):
            if PBC: return coordinate % Ly  # Periodic along Y
            else: return coordinate         # Open
            
        # Boundary conditions
        def bc_X(coordinate):
            if PBC_X: return coordinate % Lx  # Periodic along Y
            else: return coordinate         # Open

        def hopping2d(X, Y, Z):
            
            hoppings = [(ω, (X,Y,Z)), 
                            (-J, (bc_X(X+1),bc(Y),Z)), 
                            (-J, (bc_X(X-1),bc(Y),Z)), 
                            (-J*np.exp(-1j*2*np.pi*flux*bc_X(X)), (X,bc(Y+1),Z)), 
                            (-J*np.exp(1j*2*np.pi*flux*bc_X(X)), (X,bc(Y-1),Z))]
                
            return hoppings

        super(HarperHofstadter, self).__init__([Lx,Ly,1], hopping2d, **kwdargs)
        self.J = J
        self.ω = ω
        
def localization_color(vector):
    weight = np.linspace(-1, 1, len(vector))
    return np.sum(weight*vector**2)

def get_edge_dispersion(q, num_edge, threshold=0.8, N=200, side=1, L=50, fix_L=False, fit=False):
    """
    Retrieves the chiral modes with positive group velocity, localized
    at the left edge of a Harper-Hofstadter lattice with cylinder topology.
    Assumes a magnetic flux per plaquette and returns 'num_edge' edge mode
    dispersion relations.
    
    Inputs:
    
    N: Inverse of the artificial magnetic flux per plaquette (int)
    num_edge: Number of edge modes to get (int)
    threshold: Lower limit of state localization to be considered edge mode (float)
    N: Size of the discretized Brillouin Zone (int)
    side: +1 for positive group velocity (right edge), -1 for negative group
          velocity (left edge)
    
    Outputs:
    
    edge_dispersion
    """
    
    # Initialize dictionary: keys correspond to label, values correspond
    # to momentum values and energy spectrum
    edge_dispersion = {str(n): [[],[],[]] for n in range(num_edge)}
    
    # Declare momentum values and set lattice size
    ks = np.linspace(-np.pi, np.pi, N)
    if fix_L==False: Lx = 2*q + q//2
    else: Lx=L
    
    for k in ks:
      
        # Diagonalize, set counter to zero
        eigvals, eigvecs = np.linalg.eigh(Harper_matrix(1/q, k, Lx))
        counter = 0
        
        for i, eigval in enumerate(eigvals):
            
            # Append data regarding left edge-localized states
            if side*localization_color(eigvecs[:,i])>threshold and eigval<0:  
                edge_dispersion[str(counter)][0].append(k)
                edge_dispersion[str(counter)][1].append(eigval)
                edge_dispersion[str(counter)][2].append(np.abs(eigvecs[0,i]))
                
                # Update counter and set loop break
                counter += 1 
            if counter == num_edge: break
    
    if fit:
        for l in range(num_edge):    
        
            # Retrieve l'th edge mode dispersion relation
            ks, omega, _ = edge_dispersion[str(l)]
        
            # Convert to arrays and eliminate plateaus
            ks, omega = np.array(ks), np.array(omega)
            indices = (np.abs(np.gradient(omega))>1e-2)
            edge_dispersion[str(l)][0], edge_dispersion[str(l)][1] = ks[indices], omega[indices]
    
    
    return edge_dispersion


def get_group_velocities(edge_dispersion, E, n=1):
    """
    Computes the group velocity of all edge states contained
    in input dictionary 'edge_dispersion' that contain an 
    eigenvalue E, which is the energy at which velocities are
    evaluated.
    
    Inputs:
    
    edge_dispersion: Dispersion relations of edge modes (Dictionary)
    E: Energy at which group velocities are evaluated (float)
    
    Outputs:
    
    vgs: Group velocities of edge modes with energy E (list)
    """
    # Get total number of edge modes in dictionary and initialize output list
    num_edge = len(edge_dispersion)
    vgs = []
    edge_weights = []
    
    for l in range(num_edge):
        
        # Retrieve dispersion relation of l'th modes
        ks, omegas, eigvecs = edge_dispersion[str(l)]
        omegas = np.array(omegas)
        
        # If there is an eigenvalue E, compute group velocity
        if np.amin((omegas-E)**2)<1e-2:
            index = np.argmin((omegas-E)**2)
            #vgs.append(np.diff(omegas, n=n)[index]/((ks[1]-ks[0])**n))
            vgs.append(np.gradient(omegas)[index]/(ks[1]-ks[0]))
            edge_weights.append(eigvecs[index])
            
    return vgs, edge_weights

def add_emitters(H, gs, efreq=0):
    """
    Couples a series of emitters to a photonic lattice, and returns
    the resulting hamiltonian, assuming constant coupling constant and
    detuning among the set of emitters.
    
    Inputs:
    
    H .................... Hamiltonian of the photonic lattice (sparse matrix dxd)
    gs ................... Local couplings (complex array of dim (d, N) where d is the 
                           Hilbert space dimension and N is the numer )
    efreq ................ Frequency of the quantum emitter(s) (float)
    
    Outputs:
    
    H_e .................. System hamiltonian: Lattice + QEs + Interaction 
                           (sparse matrix (d+N)x(d+N))
    """
    # We start by declaring the number of emitters and its Hamiltonian
    N = gs.shape[1]
    H_emitters = sp.diags(efreq * np.ones((N)), dtype=complex)
    
    """
    We are seeking for a matrix representation of the hamiltonian in the single-excitation
    subspace. In basis {photon excitations, emitter excitations}, this matrix reads:
    
    (  H_lattice     gs   )
    (                     )
    (      gs       efreq )
    
    We will first conform the rows stacking its components horizontally, and then build
    the hamiltonian stacking the rows vertically
    """
    top = sp.hstack([H, gs])
    bottom = sp.hstack([gs.conjugate().transpose(), H_emitters])
    H_e = sp.vstack([top, bottom])
    
    return H_e

def total_hamiltonian(efreq, L, q, g=0.05, PBC=True, defects=[], Ly=None):
    """
    Gets the Hamiltonian describing the whole system of bath 
    (Harper-Hofstadter lattice) and a single quantum emitter
    of energy efreq and coupled to the lattice with coupling 
    constant g
    
    Inputs:
    
    efreq: Transition frequency of the quantum emitter (float)
    L: Determines the size LxL of the lattice (int)
    q: Inverse magnetic flux per-plaquette (int)
    g: Light-matter coupling constant (float)
    
    Outputs:
    
    H: Total Hamiltonian (sparse matrix of dim (L^2+1)x(L^2+1)
    """
    G = HarperHofstadter(L, Ly=Ly, flux=1/q, PBC=PBC).hamiltonian()
    
    if len(defects)>0:
        G = G.todense()
        for defect in defects:
            G[defect, :] = 0
            G[:, defect] = 0
        G = sp.csr_matrix(G)    
        
    gs = np.zeros((G.shape[0], 1))
    site = L//2
    gs[site, 0] = g 
    H = add_emitters(G, gs, efreq)
    
    return H

def dynamics_left_edge(H, T, full_lattice=False):
    """
    Computes the time-evolved population at time T of the local modes at
    sites located at the left edge of the lattice.
    
    Inputs:
    
    H: Total Hamiltonian (bath + emitter) (sparse matrix)
    T: Duration of the evolution (float)
    full_lattice: Boolean variable. If False (default) returns only left
                  left edge populations. If True, returns all lattice
                  populations
    
    Outputs:
    
    left_edge_pops: Population of evolved local modes at left edge (array)
    """
    # Prepare initial state: |vac>|e>
    psi = np.zeros((H.shape[0], 1))
    psi[-1, 0] = 1
    psi = sp.csr_matrix(psi)
    
    # Evolve the state and get emitter population
    psi_new = (sp.linalg.expm_multiply(-1j*H*T, psi)).todense()
    C_e = np.abs(psi_new[-1,0])**2
    
    # Get lattice size
    L = int(np.sqrt(H.shape[0] - 1))
    
    if full_lattice: 
        # Retrieve all lattice sites
        all_pops = np.array(np.abs(psi_new[0:-1, 0]))**2
        all_pops = np.reshape(all_pops, (L,L), order='F')
        return all_pops, C_e
    else:
        # Retrieve the population in left edge sites
        left_edge_pops = np.reshape(np.array(np.abs(psi_new[0:L, 0]))**2, (L))
        
        return left_edge_pops, C_e
    
def dynamics_left_edge_coherences(H, T, full_lattice=False, project=False):
    """
    Computes the time-evolved coherences at time T of the local modes at
    sites located at the left edge of the lattice.
    
    Inputs:
    
    H: Total Hamiltonian (bath + emitter) (sparse matrix)
    T: Duration of the evolution (float)
    full_lattice: Boolean variable. If False (default) returns only left
                  left edge populations. If True, returns all lattice
                  populations
    
    Outputs:
    
    left_edge_pops: Population of evolved local modes at left edge (array)
    """
    # Prepare initial state: |vac>|e>
    psi = np.zeros((H.shape[0], 1), dtype=complex)
    psi[-1, 0] = 1
    psi = sp.csr_matrix(psi)
    
    # Evolve the state and get emitter population
    psi_new = (sp.linalg.expm_multiply(-1j*H*T, psi)).todense()
    C_e = np.abs(psi_new[-1,0])**2
    
    # Get lattice size
    L = int(np.sqrt(H.shape[0] - 1))
    
    if full_lattice: 
        # Retrieve all lattice sites
        all_coherences = np.array(psi_new[0:-1, 0], dtype=complex)
        all_coherences = np.reshape(all_coherences, (L,L), order='F')
        return all_coherences, C_e
    else:
        if project:
            # Retrieve all lattice sites
            all_coherences = np.array(psi_new[0:-1, 0], dtype=complex)
            all_coherences = np.reshape(all_coherences, (L,L), order='F')
            projected = np.sum(all_coherences, axis=1)
            
            return projected, C_e
            
        else:
            # Retrieve the population in left edge sites
            left_edge_coherences = np.reshape(np.array(psi_new[0:L, 0], dtype=complex), (L))
        
            return left_edge_coherences, C_e
        
def dynamics_left_edge_reps(H, T, n):
    
    times = np.linspace(0, T, n)
    L = int(np.sqrt(H.shape[0] - 1))
    populations = np.zeros((n,L))
    
    for (i,t) in enumerate(times):
        populations[i,:] = dynamics_left_edge(H, t)[0]
        
    return np.flipud(populations)    
        
def Fourier_transform_matrix(C, N):
    """
    Matrix used to get the effective Hamiltonian in real space.
    
    Inputs:
    
    C: Number of effective waveguide modes i.e. Chern number
    N: Number of effective waveguide sites
    
    Outputs:
    
    U: Basis change matrix (dim (C*N+1, C*N+1))
    """
    # Declare real and momentum space discretizations
    js = np.arange(0,N)
    ks = np.linspace(-np.pi, np.pi, N)
    
    # Initalize basis change matrix
    U = np.zeros((C*N+1,C*N+1), dtype=complex)
    
    # Build single-mode Fourier transform
    U_block = np.zeros((N,N), dtype=complex)
    for (i, k) in enumerate(ks):
            U_block[i, :] = (1/np.sqrt(N))*np.exp(-1j*k*js)
    
    # Take direct sum over modes to define U
    for l in range(C):
        U[N*l:N*l+N, N*l:N*l+N] = U_block
        
    # Finally, add the emitter degree of freedom    
    U[-1,-1] = 1    
    
    return U

def get_DoS(energy, function=None, args=None, min_E=None, max_E=None, n=1000):
    """
    Compute the density of stats (DoS) for a given energies previously computed by other functions. This method assumes
    an approximation for the Dirac delta. If non approximation is provided, then and a square Heaviside function is
    used to weight the energies.
    
    Inputs:
    
    energy ...................... Array of computed energies
    function .................... Optional, approximates the Dirac delta to weight energies
    args ........................ Input values for 'function'
    min_E, max_E ................ Minimum/maximum values where to compute the DoS
    n ........................... Number of energies at which evaluate the DoS
    
    Outputs:
    
    energy_vector ............... Vector of equally spaced energies from min_E to max_E
    DoS ......................... Density of states (array)
    """

    # If no function is provided
    if function is None:
        # Define an two sides Heaviside step with total width of delta_x
        def function(x, delta_x):
            return np.sum((x > - delta_x / 2) * (x < delta_x / 2))

    # If the extreme values for the energy are not given, it is set to the extreme values of the given energy multiplied
    # by certain factor to observe the decrease of the DoS
    range_E = np.max(energy) - np.min(energy)
    scale_factor = 10 / 100
    if min_E is None:
        min_E = np.min(energy) - range_E * scale_factor
    if max_E is None:
        max_E = np.max(energy) + range_E * scale_factor

    # Initialize the array to save the DoS and the energies at which evaluate it
    DoS = np.zeros(n)
    energy_vector = np.linspace(min_E, max_E, n, endpoint=True)

    # Iterate every energy
    for i in range(n):
        x = (energy_vector[i] - energy)
        # Count the energies weighted by a given approximated Dirac Delta
        DoS[i] = function(x, *args)

    return energy_vector, DoS

def normal(x, s):
    return np.sum(1/(np.sqrt(2*np.pi)*s)*np.exp(-(x/s)**2))

def get_LDoS(energy, eigenvectors, pos=0, function=None, args=None, min_E=None, max_E=None, n=1000):
    """
    Compute the local density of stats (LDoS) for a given energies previously computed by other functions, and
    their corresponding eigenvectors. The local density of states, evaluated at a position 'pos', weights each
    Dirac delta term in the density of states by the amplitude of each eigenstate in such position.
    
    Respect of the previous function:
    
    Extra Inputs:
    
    eigenvectors ........... Eigenvectors of the system (dxd matrix)
    pos .................... Index denoting position where to evalueate the LDoS (int).
    """
    
    # Now, we define the function so it returns an array with lenght = len(energy)
    if function is None:
        # Define an two sides Heaviside step with total width of delta_x
        def function(x, delta_x):
            return (x > - delta_x / 2) * (x < delta_x / 2)

    # Now, we compute the local weight in array-form
    eigenvectors = np.array(eigenvectors)
    local_weight = np.abs(eigenvectors[pos,:])**2
    
    # We now implement the analogous of 'function' in the DoS function,
    # this time providing the local weight
    
    def weighted_sum(x, local_weight, *args):
         return np.sum(local_weight * function(x, *args))
   
    # If the extreme values for the energy are not given, it is set to the extreme values of the given energy multiplied
    # by certain factor to observe the decrease of the DoS
    range_E = np.max(energy) - np.min(energy)
    scale_factor = 10 / 100
    if min_E is None:
        min_E = np.min(energy) - range_E * scale_factor
    if max_E is None:
        max_E = np.max(energy) + range_E * scale_factor

    # Initialize the array to save the LDoS and the energies at which evaluate it
    LDoS = np.zeros(n)
    energy_vector = np.linspace(min_E, max_E, n, endpoint=True)

    # Iterate every energy
    for i in range(n):
        x = (energy_vector[i] - energy)
        # Count the energies weighted by a given approximated Dirac Delta
        LDoS[i] = weighted_sum(x, local_weight, *args)

    return energy_vector, LDoS

def get_relative_decay_rate(g, q, num_edge, N=100, l=0, threshold=0.65, E_min=None, E_max=None, n=80):
    
    edge_dispersion = get_edge_dispersion(q, num_edge, threshold=threshold, side=-1, fix_L=True, N=n)
    
    if E_min==None and E_max==None:
        E_min = Landau_energy(l, q)
        E_max = Landau_energy(num_edge-1,q)+0.5*(Landau_energy(num_edge,q)-Landau_energy(num_edge-1,q))
        
    energies = np.linspace(E_min, E_max, N)
    
    relative_decay_rate = []
    total_decay = []
    
    for E in energies:
        
        vgs, edge_weights = get_group_velocities(edge_dispersion, E)
        vgs, edge_weights = np.abs(np.array(vgs)), np.array(edge_weights)
        Gamma_0 = (g*edge_weights[l])**2/vgs[l]
        Gamma = np.sum((g*edge_weights)**2/vgs)
        relative_decay_rate.append(Gamma_0/Gamma)
        total_decay.append(Gamma)
        
    return np.array(relative_decay_rate), np.array(total_decay), np.array(energies)        

def normal_LDoS(x, s):
    return 1/(np.sqrt(2*np.pi)*s)*np.exp(-(x/s)**2)