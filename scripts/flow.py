import numpy as np

def flow(H = 12.636, p = 2.953, M = 131.293, q = 2, f = 10, Edep = .66581):
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
    HoV = H / M * p * kJ # J / cm^3
    E = n * Edep * GeV_to_J # J

    Q = E * f / HoV # cm^3 / s

    print(f'HoV : {HoV} J/cm3\n  E : {E} J')
    print(f'Flow rate : {Q}')

    return Q

if __name__ == '__main__':
    flow()