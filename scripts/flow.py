import numpy as np

def VolMass(rho = 2.953, T = 5.5, spot = .6):
    T *= 2.872 # Rad lengths to cm
    V = spot ** 2 * np.pi * T # cm^3
    m = V * rho # g

    return V, m

# def flow(Etot = 0, H = 12.636, rho = 2.953, M = 131.293, q = 2, f = 10, Edep = .66581):
def flow(Etot = 0, H = 12.636, rho = 2.953, M = 131.293, q = 2, f = 10, PropDep = .27, T = 5.5, spot = 6):
    '''
    Calculate the required flow rate due to energy deposition from electron beam pulse

    Parameters (optional)
    --------------------
    Etot : float
        Total energy deposited (J)
    H : float
        Heat of vaporization (kJ / mol)
    rho : float
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
    J = 1e3 # kJ to J

    HoV = H / M * rho * J # J / cm^3

    if Etot:
        E = Etot # J
    else:
        n = q / e # number of electrons per beam pulse
        E = n * PropDep * 10 * GeV_to_J # J

        T *= 2.872 # Rad lengths to cm
        spot *= 1e-1 # cm
        V = spot ** 2 * np.pi * T # cm^3
        m = V * rho # g
        # m = 116620 # g
        # print(f'Mass : {m} g')
        
        # print(f'FACET-II Power Dep : {E * f} W')
        print(f'Energy Dep Density : {E / m} J/g')

    Q = E * f / HoV # cm^3 / s

    print(f'HoV : {HoV} J/cm3')
    print(f'E : {E} J')
    print(f'Flow rate : {Q} cm^3/s\n')
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

def bunchFlow(f, T):
    '''
    f = frequency (Hz)
    T = thickness (Rad Lengths)
    '''
    v, _ = VolMass(T = T)
    print(f'Flow rate : {v * f} cm^3/s\n')

def bunchEnergy(nBunch, rho = 2.953, T = 4.5, spot = 6, n = 2.5e10, PropDep = .22):
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
    # kJ = 1e3 # kJ to J
    # microsecond = 1e-6 # µs to s
    mm = 1e-1 # mm to cm

    spot *= mm # mm to cm
    # t *= microsecond # µs to s
    
    Edep = 3 * PropDep * n * GeV_to_J # J

    # bunchEnergy = 3 * GeV_to_J * 7.8e9 # J
    # print(f'Bunch Energy : {bunchEnergy} J')
    Etot = nBunch * Edep # J
    # Power = Etot / t # W
    # Power = Etot * t
    # Power = 300 * Etot # 300 Hz frequency times Etot gives total Power

    # m = 116620 # g
    T *= 2.872 # Rad lengths to cm
    V = spot ** 2 * np.pi * T # cm^3
    # print(V)
    m = V * rho # g
    # print(f'Mass : {m} g')


    
    # print(f'ILC Q : {e * 2e10} nC')
    # print(f'Power Dep : {PropDep * Power * 1e-3} kW')
    print(f'Energy Dep Density per Train : {Etot / m} J/g')
    print(f'Energy Dep per Train: {Etot} J\n')

    return Etot

def trains_to_vaporize(T, Edep_per_train, HoV = 284.2):
    v, _ = VolMass(T = T)
    E_to_vap = v * HoV
    trains_to_vap = E_to_vap / Edep_per_train
    print(f'Number of trains to vaporize : {trains_to_vap}')

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

def be(Edep_per_electron = 9.21, T = .05, n = 1.25, bunches = 1, f = 10):
    '''
    Calculate the temperature change of Be windows due to energy deposition
    
    Exit : 9.21 MeV / electron at 10 GeV
    Exit : 3.04 MeV / electron at 3 GeV

    Parameters
    ----------
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
    print('#########################\n')
    print(f'Energy/train : {Edep:.3f} J')
    print(f'Delta T/train : {Edep / (m * specificHeat):.3f} K')
    print(f'Delta T : {f * Edep / (m * specificHeat):.3f} K/s\n')
    # print('#########################\n')

if __name__ == '__main__':
    # print('-------------------')
    # print('Liquid Xenon Target')
    # print('-------------------')
    # E = bunchEnergy(1, 132)
    # flow(E, f = 300)
    # flow()
    # force(r = 50.)
    # print('---------------')
    # print('Tungsten Target')
    # print('---------------')
    # print('')
    # E = bunchEnergy(1, 132)
    # flow(E, f = 300)
    
    # print('')
    # E = bunchEnergy(2e5, 66)
    # flow(E)

    # print('C3')
    # E = bunchEnergy(8.33e3, 133)
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


    # FACET-II
    # flow(PropDep = .02, T = 2)
    
    # ILC

    # bunchEnergy(1312, T = 2.5, PropDep = .08)
    # bunchFlow(5, 2.5)

    # C3
    # bunchEnergy(133, T = 2.5, PropDep = .08, n = .78e10)
    # trains_to_vaporize(T = 2.5, Edep_per_train = 39.886)
    # bunchFlow(120 / 33, 2.5)

    # print(VolMass())
    # flow()
    be(Edep_per_electron = 2.66)
    be(2.14, n = 2.5, bunches = 1312, f = 5)
    be(2.14, n = 0.78, bunches = 133, f = 120)
    
    # be(3.04, n = 2.5, bunches = 1312, f = 5)
    # be(3.04, n = 0.78, bunches = 133, f = 120)
    
    # be(3.04, n = 1.125, bunches = 1000, f = 11.1)