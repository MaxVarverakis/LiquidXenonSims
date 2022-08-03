import numpy as np

def flow(H = 12.636, rho = 2.953, M = 131.293, q = 2, f = 10, Edep = .66581):
    '''
    Calculate the required flow rate due to energy deposition from electron beam pulse

    Parameters
    ----------
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
    E = n * Edep * GeV_to_J # J

    Q = E * f / HoV # cm^3 / s

    print(f'HoV : {HoV} J/cm3\nE : {E} J\n')
    print(f'Flow rate : {Q} cm^3/s\n')

    return Q

def force(r = 10, h = 1., rho = 2.953, g = 9.81):
    '''
    Calculate the force on a disk of radius r

    Parameters
    ----------
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
    flow()
    force()