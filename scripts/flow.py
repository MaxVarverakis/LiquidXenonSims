import numpy as np

# def flow(Etot = 0, H = 12.636, rho = 2.953, M = 131.293, q = 2, f = 10, Edep = .66581):
def flow(Etot = 0, H = 12.636, rho = 2.953, M = 131.293, q = 2, f = 300, Edep = 2.3, T = 4.5, spot = 6):
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
    T : float
        Thickness of the target in Radiation Lengths
    spot : float
        Spot size of e+ shower (mm)
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

        T *= 2.872 # Rad lengths to cm
        spot *= 1e-1 # cm
        V = spot ** 2 * np.pi * T # cm^3
        m = V * rho # g
        # m = 116620 # g
        # print(f'Mass : {m} g')
        
        # print(f'FACET-II Power Dep : {E * f} W')
        print(f'FACET-II Energy Dep Density : {E / m} J/g')

    Q = E * f / HoV # cm^3 / s

    # print(f'HoV : {HoV} J/cm3')
    # print(f'E : {E} J')
    print(f'Flow rate : {Q} cm^3/s')
    # \n----------------------------------------\n----------------------------------------\n\

    return Q

def altFlow(f = 10, r = 10, T = 4.5):
    '''
    Calculate flow rate required to move rectangular prism in a given time
    '''
    mm = 1e-1 # mm to cm

    T *= 2.872 # Rad lengths to cm
    V = (2 * r * mm) ** 2 * T

    print(f'Volume : {V} cm^3')
    print(f'Flow rate : {V * f * 1e-3} L/s')

def ILC(t, nBunch, rho = 2.953, T = 4.5, spot = 6, PropDep = .18):
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
    # e = 1.602e-10 # nC
    GeV_to_J = 1.602e-10 # GeV to J
    kJ = 1e3 # kJ to J
    # microsecond = 1e-6 # µs to s
    mm = 1e-1 # mm to cm

    spot *= mm # mm to cm
    # t *= microsecond # µs to s

    bunchEnergy = 3 * GeV_to_J * 7.8e9 # J
    # print(f'Bunch Energy : {bunchEnergy} J')
    Etot = nBunch * bunchEnergy # J
    # Power = Etot / t # W
    Power = Etot / t
    # Power = 300 * Etot # 300 Hz frequency times Etot gives total Power

    # m = 116620 # g
    T *= 2.872 # Rad lengths to cm
    V = spot ** 2 * np.pi * T # cm^3
    # print(V)
    m = V * rho # g
    print(f'Mass : {m} g')


    
    # print(f'ILC Q : {e * 2e10} nC')
    # print(f'Power Dep : {PropDep * Power * 1e-3} kW')
    print(f'Energy Dep Density : {PropDep * Etot / m} J/g')
    # print(f'Energy Dep : {PropDep * Etot} J\n')

    return PropDep * Etot

def force(r = 10., h = 1., rho = 2.953, g = 9.81):
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
    offset = 3e5 # Pa

    A = np.pi * r ** 2 # m^2
    p =  h * rho * g # kg / m s^2 = Pa
    p += offset # Pa
    F = p * A # N
    
    F_A = 400 * 1e6 # Pa
    T = .866 * 2 * r * np.sqrt(p / F_A)

    print(f'Area : {A * 1e6} mm^2')
    print(f'Pressure : {p * 1e-3} kPa')
    print(f'Force : {F} N')
    print(f'Thickness : {T * 1e6} µm')

    

    return p, F, T

def be(Edep_per_electron = 6.76, T = .05, n = 1.25, bunches = 1, f = 10):
    '''
    Calculate the temperature change of Be windows due to energy deposition
    
    Entrance : 0.15 MeV / electron
    Exit : 6.76 MeV / electron

    Parameters
    ----------
    E : float
        Drive beam energy (GeV)
    Edep_per_electron : float
        Energy deposited per electron (MeV)
    T : float
        Thickness of Be window (cm)
            (496.701 µm minimum)
    n : float
        Number of e- per bunch (10^10)
    bunches : int
        Number of bunches per pulse (or bunch train)
    f : float
        Beam rep. rate (Hz)
    '''
    # Eprop = 42e-4 # 6 MeV dep in window / 10 GeV beam energy from simulation
    # Eprop = (Edep_per_electron * 1e-3) / 10
    # print(f'Eprop : {Eprop}')
    r = 1 # cm
    # T *= 1e-4 # µm to cm
    density = 1.848 # g / cm^3
    specificHeat = 1.82 # J / g K
    n *= 1e10

    V = np.pi * r ** 2 * T # cm^3
    m = V * density # g
    # print(m)

    # E *= 1.602e-13 # MeV to J
    Edep = Edep_per_electron * 1.602e-13 * n * bunches # J

    # print(f'Mass : {m} g')
    print(f'Energy/train : {Edep:.3f} J')
    print(f'Delta T/train : {Edep / (m * specificHeat):.3f} K')
    print(f'Delta T : {f * Edep / (m * specificHeat):.3f} K/s\n')

if __name__ == '__main__':
    # print('-------------------')
    # print('Liquid Xenon Target')
    # print('-------------------')
    # E = ILC(1, 132)
    # flow(E, f = 300)
    # flow()
    # force(r = 50.)
    # print('---------------')
    # print('Tungsten Target')
    # print('---------------')
    # print('')
    # E = ILC(1, 132)
    # flow(E, f = 300)
    
    # print('')
    # E = ILC(2e5, 66)
    # flow(E)

    # print('C3')
    # E = ILC(8.33e3, 133)
    # flow(E, f = 120)
    
    # print('FACET-II')
    # flow()

    # altFlow()
    # altFlow(f = 300)
    # print('-----------')
    # print('Entrance Window')
    # print('-----------')
    # be(E = .1)
    # be(E = .1, bunches = 132, t = 1e-6)
    # print('-----------')
    # print('Exit Window')
    # print('-----------')

    be()
    be(n = 2.5, bunches = 1312, f = 5)
    be(n = 0.78, bunches = 133, f = 120)
    be(n = 1.125, bunches = 1000, f = 11.1)