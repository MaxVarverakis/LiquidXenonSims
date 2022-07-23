# import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ef = pd.read_csv('build/out0_nt_e-.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]'])
pf = pd.read_csv('build/out0_nt_e+.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]'])
# os.remove('build/out0_nt_Data.csv')

# print(df)

eData = ef.to_numpy()
pData = pf.to_numpy()
eE = eData[:, 0]
eW = eData[:, 1]
eA = eData[:, 2]

pE = pData[:, 0]
pW = pData[:, 1]
pA = pData[:, 2]

# print(ef)

def collect(x, y, b, scale = 1000):
    amount = np.zeros((len(b) - 1, 2))
    for i,val in enumerate(x):
        for idx in range(len(b) - 2):
            if val >= b[idx] and val <= b[idx + 1]:
                amount[idx][0] += y[i] ** 2
                amount[idx][1] += 1
    return [scale * np.sqrt(val / count)  if count != 0 else None for val,count in amount]

bins = np.linspace(0, 1e4, 51)

(neE, _, _) = plt.hist(eE, bins = bins)
(npE, _, _) = plt.hist(pE, bins = bins)
plt.close()

# for i in range(len(bins) - 1):
#     print(i, bins[i], bins[i + 1], width)

width = (bins[1] - bins[0]) / 2
energy = [bins[i] - width for i in range(1, len(bins))]
# print(energy)

# print(npE)

plt.subplot(1, 3, 1)
factor = (bins[1] - bins[0]) * 1e6
# factor = (len(bins) - 1) * (bins[1] - bins[0]) * 1e6

# plt.plot(energy, neE / factor, label = 'Electron')
# plt.plot(energy, npE / factor, label = 'Positron')
plt.semilogx(energy, neE / factor, label = 'Electron', c = 'r')
plt.semilogx(energy, npE / factor, label = 'Positron', c = 'b')
plt.xlabel('Energy [MeV]')
plt.ylabel('Energy Spectrum [/MeV]')
plt.grid(True, which = 'both', ls = ':')
plt.legend()
# plt.show()

plt.subplot(1, 3, 2)
eArms = collect(eE, eA, bins)
pArms = collect(pE, pA, bins)

# plt.scatter(eE, eA * 1000, label = 'Electron')
# plt.scatter(pE, pA * 1000, label = 'Positron')

plt.loglog(energy, eArms, label = 'Electron', c = 'r')
plt.loglog(energy, pArms, label = 'Positron', c = 'b')
plt.ylabel('RMS Angle [mrad]')
plt.xlabel('Energy [MeV]')
plt.grid(True, which = 'both', ls = ':')
plt.legend()
# plt.show()

plt.subplot(1, 3, 3)
eWrms = collect(eE, eW, bins)
pWrms = collect(pE, pW, bins)

# plt.scatter(eE, eA * 1000, label = 'Electron')
# plt.scatter(pE, pA * 1000, label = 'Positron')

plt.loglog(energy, eWrms, label = 'Electron', c = 'r')
plt.loglog(energy, pWrms, label = 'Positron', c = 'b')
plt.ylabel('RMS Spot Size [$\mu m$]')
plt.xlabel('Energy [MeV]')
plt.grid(True, which = 'both', ls = ':')
plt.legend()

plt.show()