import numpy as np

def flow(Etot = 0, H = 12.636, rho = 2.953, M = 131.293, q = 2, f = 10, Edep = .66581):
    '''
    Calculate the required flow rate due to energy deposition from electron beam pulse

    Parameters (optional)
    --------------------
    Etot : float
        Total energy deposited (J)
    H : float
        Heat of vaporization (kJ / mol)
    p : float
        Density of target (g / cm^3)
    M : float
        molecular weight of target (g / mol)
    q : float
        Charge of the electron beam (nC)
    f : float
        Frequency of the electron beam (Hz)
    Edep : float
        Energy deposited in the target per electron (GeV)
    '''

    # Conversion factors
    e = 1.602e-10 # nC
    GeV_to_J = 1.602e-10 # GeV to J
    kJ = 1e3 # kJ to J

    n = q / e # number of electrons per beam pulse
    HoV = H / M * rho * kJ # J / cm^3
    if Etot:
        E = Etot # J
    else:
        E = n * Edep * GeV_to_J # J

        T = 15.796 # cm
        spot = 12.73 * 1e-1 # cm
        V = spot ** 2 * np.pi * T # cm^3
        m = V * rho # g
        # m = 116620 # g
        
        print(f'FACET-II Power Dep : {.29 * E * f} W')
        print(f'FACET-II Energy Dep Density : {.29 * E / m} J/g')

    Q = E * f / HoV # cm^3 / s

    # print(f'HoV : {HoV} J/cm3\nE : {E} J\n')
    print(f'Flow rate : {Q} cm^3/s\n')

    return Q

def altFlow(f = 10, r = 10, T = 15.796):
    '''
    Calculate flow rate required to move rectangular prism in a given time
    '''
    mm = 1e-1 # mm to cm

    V = (2 * r * mm) ** 2 * T

    print(f'Volume : {V} cm^3')
    print(f'Flow rate : {V * f * 1e-3} L/s')

def ILC(t, nBunch, rho = 2.953, T = 15.796, spot = 12.73):
    '''
    Calculate energy deposition stuff for ILC

    Parameters
    ----------
    t : float
        Time of the pulse (µs) 
    nBunch : int
        Number of bunches
    '''

    # Conversion factors
    e = 1.602e-10 # nC
    GeV_to_J = 1.602e-10 # GeV to J
    kJ = 1e3 # kJ to J
    microsecond = 1e-6 # s to µs
    mm = 1e-1 # mm to cm

    PropDep = .29

    spot *= mm # mm to cm
    t *= microsecond # µs to s

    bunchEnergy = 6 * GeV_to_J * 2e10 # J
    # print(f'Bunch Energy : {bunchEnergy} J')
    Etot = nBunch * bunchEnergy # J
    Power = Etot / t # W

    # m = 116620 # g
    V = spot ** 2 * np.pi * T # cm^3
    m = V * rho # g
    # print(m)


    
    # print(f'ILC Q : {e * 2e10} nC')
    print(f'ILC Power Dep : {PropDep * Power * 1e-6} MW')
    print(f'ILC Energy Dep Density : {PropDep * Etot / m} J/g')
    print(f'ILC Energy Dep : {PropDep * Etot} J')

    return PropDep * Etot


def force(r = 10, h = 1., rho = 2.953, g = 9.81):
    '''
    Calculate the force on a disk of radius r

    Parameters (optional)
    --------------------
    r : float
        Radius of disk (mm)
    h : float
        Height of chamber (m)
    rho : float
        Density of target (g / cm^3)
    g : float
        Acceleration due to gravity (m / s^2)
    '''
    r *= 1e-3 # mm to m
    rho *= 1e3 # g / cm^3 to kg / m^3

    A = np.pi * r ** 2 # m^2
    p =  h * rho * g # kg / m s^2 = Pa
    F = p * A # N
    
    F_A = 400 * 1e6 # Pa
    T = .866 * 2 * r * np.sqrt(p / F_A)

    print(f'Area : {A * 1e6} mm^2')
    print(f'Pressure : {p * 1e-3} kPa')
    print(f'Force : {F} N')
    print(f'Thickness : {T * 1e6} µm')

    

    return p, F, T

if __name__ == '__main__':
    print('-------------------')
    print('Liquid Xenon Target')
    print('-------------------')
    E = ILC(1, 132)
    flow(E, f = 300)
    flow()
    print('---------------')
    print('Tungsten Target')
    print('---------------')
    E = ILC(1, 132, rho = 20, T = 1.4, spot = 4)
    flow(E, f = 300)
    E = ILC(63, 2640)
    flow(E)

    altFlow()
    altFlow(f = 300)