# import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Cut all e+ over 1mm spot size
# Beryllium on either side of Xe => min thickness of Be needed to suppord vac on one side and liquid density on other?
# ask doug in slack
# vacuum window
LXe = 2.8720
LTa = 0.4094
L_RL = np.arange(.25, 4.25, .25)
iterations = 1e5
E0 = 1e4 # MeV
w = 5 # mm

mode = 'Xe'
cutoff = True

def plots(xlbl, ttlStr, ts2 = ''):
    # os.remove('build/out0_nt_e+.csv')
    # count = 0
    # if filterSpotSize:
    #     for df in [df0, df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15]:
    #         df = df[df[:, 1] <= .1]

    pYield = [len(df[:, 0]) / iterations for df in [df0, df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15]]
    # print(pYield)
    # print(L_RL)

    EDep_RMS = [np.sqrt(np.mean(edep ** 2)) / E0 for edep in [edep0, edep1, edep2, edep3, edep4, edep5, edep6, edep7, edep8, 
                                                        edep9, edep10, edep11, edep12, edep13, edep14, edep15]]
    E_RMS = [np.sqrt(np.mean(df[:, 0] ** 2)) for df in [df0, df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15]]
    TW_RMS = [np.sqrt(np.mean(df[:, 1] ** 2)) for df in [df0, df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15]]
    A_RMS = [np.sqrt(np.mean(df[:, 2] ** 2)) for df in [df0, df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15]]
    # E_RMS, TW_RMS, A_RMS = [np.sqrt(np.mean(np.square(df), axis = 0)) for df in [df0, df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15]]
    # print(E_RMS_Test, E_RMS)

    # print(TW_RMS, A_RMS)
    # TW_RMS = np.multiply(TW_RMS, 1e3)
    # A_RMS = np.multiply(A_RMS, 1e3)
    # print(TW_RMS, A_RMS)

    idx = np.argmax(pYield)
    data = [df0, df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15]
    data = data[idx]
    # data = data[0]
    E_filter = 1e2

    # TW_RMS = [TW_RMS[idx] if val > E_filter else None for idx,val in enumerate(E_RMS)]
    # A_RMS = [A_RMS[idx] if val > E_filter else None for idx,val in enumerate(E_RMS)]

    E = data[:, 0]
    TW = data[:, 1]
    A = data[:, 2]
    E = E[E > E_filter]
    TW = [TW[idx] if val > E_filter else None for idx,val in enumerate(E)]
    A = [A[idx] if val > E_filter else None for idx,val in enumerate(E)]

    # print(E)

    plt.scatter(L_RL, EDep_RMS, label = '__nolegend__')
    plt.plot(L_RL[idx], EDep_RMS[idx], 'ro', label = f'Fractional Energy Deposition @ Max Yield: ~{EDep_RMS[idx]:.2f}')
    plt.vlines(L_RL[idx], 0, EDep_RMS[idx], colors = 'r', ls = ':', label = '__nolegend__')
    plt.xlabel(xlbl)
    plt.ylabel('RMS Fractional Energy Deposition [ / incident $e^-$ @ 10 GeV]')
    # plt.ylabel('RMS Energy Deposition / incident $e^-$ [MeV]')
    # plt.ylim(0, 5.75e-1)
    plt.legend(loc = 'upper left')
    plt.title('Energy Deposition vs Radiation Length')
    plt.show()

    cm = 'inferno'
    bn = (25, 75)
    scale = 100

    plt.subplot(1, 2, 1)
    plt.hist2d(E, TW, bins = bn, weights = np.full(np.shape(TW), scale / iterations), norm = mpl.colors.LogNorm(), cmap = cm)
    plt.ylim(0, 100)
    plt.xlim(0, 7e3)
    plt.title('$e^+$ Shower Size vs $e^+$ Energy')
    plt.xlabel('Energy [MeV]')
    plt.ylabel('Spot Size [mm]')

    plt.subplot(1, 2, 2)
    h = plt.hist2d(E, A,  bins = bn, weights = np.full(np.shape(TW), scale / iterations), norm = mpl.colors.LogNorm(), cmap = cm)
    plt.colorbar(h[3], label = 'Counts / 100 incident $e^-$')
    plt.ylim(0, 1)
    plt.xlim(0, 7e3)
    plt.title('$e^+$ Diffraction Angle vs $e^+$ Energy')
    plt.xlabel('Energy [MeV]')
    plt.ylabel('Angle [rad]')

    plt.suptitle(ttlStr, fontsize = 14)
    plt.show()

    plt.subplot(1, 2, 1)
    plt.scatter(L_RL, pYield, label = '__nolegend__')
    plt.plot(L_RL[idx], pYield[idx], 'ro', label = f'Max Positron Yield: ~{pYield[idx]:.2f} / $e^-$')
    plt.vlines(L_RL[idx], 0, pYield[idx], colors = 'r', ls = ':', label = '__nolegend__')

    # plt.xticks(list(plt.xticks()[0]) + [L_RL[idx]])
    # # a[np.where(a == 2.75)] = f'~{L_RL[idx] * LXe:.2f} cm'
    # jdx = np.where(plt.xticks()[0] == [L_RL[idx]][0])[0][0]
    # l = plt.xticks()[1]
    # l[jdx] = f'~{L_RL[idx] * LXe:.2f} cm'
    # plt.xticks(plt.xticks(), labels = l)
    # print(plt.xticks()[1])

    plt.text(L_RL[idx], pYield[idx] / 100, f'~{L_RL[idx] * LTarget:.2f} cm', fontsize = 12)
    # plt.hlines(pYield[idx], 0, L_RL[idx], colors = 'r', ls = ':', label = '__nolegend__')
    plt.xlabel(xlbl)
    plt.ylabel('Normalized Positron Yield [per Incident $e^-$ @ 10 GeV]')
    plt.xlim(0, 4.25)
    plt.ylim(0, 2.5)
    # plt.ylim(0, max(pYield) * 1.05)
    plt.yticks(np.arange(0, 2.5, .25))
    plt.legend(loc = 'upper left'
    # , bbox_to_anchor = (0, .95)
    )
    plt.title('$e^+$ Yield vs Radiation Length')

    plt.subplot(1, 2, 2)
    plt.scatter(L_RL, E_RMS)
    plt.plot(L_RL[idx], E_RMS[idx], 'ro', label = f'RMS Energy @ Max Yield: ~{E_RMS[idx]:.2f} MeV')
    plt.vlines(L_RL[idx], 0, E_RMS[idx], colors = 'r', ls = ':', label = '__nolegend__')
    plt.text(L_RL[idx], E_RMS[idx] / 10, f'~{L_RL[idx] * LTarget:.2f} cm', fontsize = 12)
    # plt.hlines(E_RMS[idx], 0, L_RL[idx], colors = 'r', ls = ':', label = '__nolegend__')
    plt.xlim(0, 4.25)
    plt.ylim(0, 1300)
    # plt.ylim(0, max(E_RMS) * 1.05)
    plt.yticks(np.arange(0, max(E_RMS), 100))
    plt.xlabel(xlbl)
    plt.ylabel('RMS Positron Energy [MeV]')
    plt.legend()
    plt.title('RMS $e^+$ Energy vs Radiation Length')

    plt.suptitle(ts2, fontsize = 14)
    plt.show()

    # E_filter = 0

    # TW_RMS = [TW_RMS[idx] if val > E_filter else None for idx,val in enumerate(E_RMS)]
    # A_RMS = [A_RMS[idx] if val > E_filter else None for idx,val in enumerate(E_RMS)]

    plt.subplot(1, 2, 1)
    plt.scatter(L_RL, TW_RMS, label = '__nolegend__')
    plt.plot(L_RL[idx], TW_RMS[idx], 'ro', label = f'Traverse Width @ Max Yield: ~{TW_RMS[idx]:.2f} mm')
    plt.vlines(L_RL[idx], 0, TW_RMS[idx], colors = 'r', ls = ':', label = '__nolegend__')
    # plt.hlines(pYield[idx], 0, L_RL[idx], colors = 'r', ls = ':', label = '__nolegend__')
    plt.xlabel(xlbl)
    plt.ylabel('RMS Spot Size [mm]')
    plt.xlim(0, 4.25)
    plt.ylim(0, max(TW_RMS) * 1.05)
    # plt.yticks(np.arange(0, 2.5, .25))
    plt.legend(loc = 'upper left'
    # , bbox_to_anchor = (0, .95)
    )
    plt.title('RMS $e^+$ Shower Spot Size vs Radiation Length')

    plt.subplot(1, 2, 2)
    plt.scatter(L_RL, A_RMS)
    plt.plot(L_RL[idx], A_RMS[idx], 'ro', label = f'Diffraction Angle @ Max Yield: ~{A_RMS[idx]:.2f} rad')
    plt.vlines(L_RL[idx], 0, A_RMS[idx], colors = 'r', ls = ':', label = '__nolegend__')
    # plt.hlines(E_RMS[idx], 0, L_RL[idx], colors = 'r', ls = ':', label = '__nolegend__')
    plt.xlim(0, 4.25)
    plt.ylim(0, max(A_RMS) * 1.05)
    # plt.yticks(np.arange(0, max(A_RMS), 100))
    plt.xlabel(xlbl)
    plt.ylabel('RMS Angle [rad]')
    plt.legend()
    plt.title('RMS $e^+$ Angle vs Radiation Length')

    plt.suptitle(ts2, fontsize = 14)
    plt.show()

if mode == 'Xe':
    LTarget = LXe
    ttlStr = '$L_{RL}(Xe) = 2.8720 cm \\quad \\vert \\quad \\frac{d}{L_{LR}(Xe)} \\approx 2.75$ \n Energy Cutoff: 100 MeV'
    pltStr = 'Radiation Lengths [$L_{RL}(Xe) = 2.8720$ cm]'

    edep0 = pd.read_csv('data/XeDep0.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    edep1 = pd.read_csv('data/XeDep1.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    edep2 = pd.read_csv('data/XeDep2.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    edep3 = pd.read_csv('data/XeDep3.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    edep4 = pd.read_csv('data/XeDep4.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    edep5 = pd.read_csv('data/XeDep5.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    edep6 = pd.read_csv('data/XeDep6.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    edep7 = pd.read_csv('data/XeDep7.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    edep8 = pd.read_csv('data/XeDep8.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    edep9 = pd.read_csv('data/XeDep9.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    edep10 = pd.read_csv('data/XeDep10.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    edep11 = pd.read_csv('data/XeDep11.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    edep12 = pd.read_csv('data/XeDep12.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    edep13 = pd.read_csv('data/XeDep13.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    edep14 = pd.read_csv('data/XeDep14.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    edep15 = pd.read_csv('data/XeDep15.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()

    # edep0 = pd.read_csv('build/out0_nt_Data.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    # edep1 = pd.read_csv('build/out1_nt_Data.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    # edep2 = pd.read_csv('build/out2_nt_Data.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    # edep3 = pd.read_csv('build/out3_nt_Data.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    # edep4 = pd.read_csv('build/out4_nt_Data.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    # edep5 = pd.read_csv('build/out5_nt_Data.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    # edep6 = pd.read_csv('build/out6_nt_Data.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    # edep7 = pd.read_csv('build/out7_nt_Data.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    # edep8 = pd.read_csv('build/out8_nt_Data.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    # edep9 = pd.read_csv('build/out9_nt_Data.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    # edep10 = pd.read_csv('build/out10_nt_Data.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    # edep11 = pd.read_csv('build/out11_nt_Data.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    # edep12 = pd.read_csv('build/out12_nt_Data.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    # edep13 = pd.read_csv('build/out13_nt_Data.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    # edep14 = pd.read_csv('build/out14_nt_Data.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    # edep15 = pd.read_csv('build/out15_nt_Data.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()

    # df0 = pd.read_csv('data/Xe_1mm_0.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    # df1 = pd.read_csv('data/Xe_1mm_1.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    # df2 = pd.read_csv('data/Xe_1mm_2.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    # df3 = pd.read_csv('data/Xe_1mm_3.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    # df4 = pd.read_csv('data/Xe_1mm_4.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    # df5 = pd.read_csv('data/Xe_1mm_5.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    # df6 = pd.read_csv('data/Xe_1mm_6.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    # df7 = pd.read_csv('data/Xe_1mm_7.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    # df8 = pd.read_csv('data/Xe_1mm_8.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    # df9 = pd.read_csv('data/Xe_1mm_9.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    # df10 = pd.read_csv('data/Xe_1mm_10.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    # df11 = pd.read_csv('data/Xe_1mm_11.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    # df12 = pd.read_csv('data/Xe_1mm_12.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    # df13 = pd.read_csv('data/Xe_1mm_13.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    # df14 = pd.read_csv('data/Xe_1mm_14.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    # df15 = pd.read_csv('data/Xe_1mm_15.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()

    df0 = pd.read_csv('data/Xe0.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    df1 = pd.read_csv('data/Xe1.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    df2 = pd.read_csv('data/Xe2.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    df3 = pd.read_csv('data/Xe3.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    df4 = pd.read_csv('data/Xe4.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    df5 = pd.read_csv('data/Xe5.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    df6 = pd.read_csv('data/Xe6.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    df7 = pd.read_csv('data/Xe7.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    df8 = pd.read_csv('data/Xe8.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    df9 = pd.read_csv('data/Xe9.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    df10 = pd.read_csv('data/Xe10.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    df11 = pd.read_csv('data/Xe11.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    df12 = pd.read_csv('data/Xe12.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    df13 = pd.read_csv('data/Xe13.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    df14 = pd.read_csv('data/Xe14.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    df15 = pd.read_csv('data/Xe15.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()

    if cutoff:
        ts2 = f'Spot Size Cutoff : {w} mm'
        
        df0 = df0[df0[:, 1] <= w]
        df1 = df1[df1[:, 1] <= w]
        df2 = df2[df2[:, 1] <= w]
        df3 = df3[df3[:, 1] <= w]
        df4 = df4[df4[:, 1] <= w]
        df5 = df5[df5[:, 1] <= w]
        df6 = df6[df6[:, 1] <= w]
        df7 = df7[df7[:, 1] <= w]
        df8 = df8[df8[:, 1] <= w]
        df9 = df9[df9[:, 1] <= w]
        df10 = df10[df10[:, 1] <= w]
        df11 = df11[df11[:, 1] <= w]
        df12 = df12[df12[:, 1] <= w]
        df13 = df13[df13[:, 1] <= w]
        df14 = df14[df14[:, 1] <= w]
        df15 = df15[df15[:, 1] <= w]
        
        plots(pltStr, ttlStr, ts2)
    else:
        plots(pltStr, ttlStr)
elif mode == 'Ta':
    LTarget = LTa
    ttlStr = '$L_{RL}(Ta) = 0.4094 cm \\quad \\vert \\quad \\frac{d}{L_{LR}(Ta)} \\approx 2.75$ \n Energy Cutoff: 100 MeV'
    pltStr = 'Radiation Lengths [$L_{RL}(Ta) = 0.4094$ cm]'
    
    edep0 = pd.read_csv('data/TaDep0.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    edep1 = pd.read_csv('data/TaDep1.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    edep2 = pd.read_csv('data/TaDep2.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    edep3 = pd.read_csv('data/TaDep3.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    edep4 = pd.read_csv('data/TaDep4.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    edep5 = pd.read_csv('data/TaDep5.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    edep6 = pd.read_csv('data/TaDep6.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    edep7 = pd.read_csv('data/TaDep7.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    edep8 = pd.read_csv('data/TaDep8.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    edep9 = pd.read_csv('data/TaDep9.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    edep10 = pd.read_csv('data/TaDep10.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    edep11 = pd.read_csv('data/TaDep11.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    edep12 = pd.read_csv('data/TaDep12.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    edep13 = pd.read_csv('data/TaDep13.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    edep14 = pd.read_csv('data/TaDep14.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    edep15 = pd.read_csv('data/TaDep15.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()

    df0 = pd.read_csv('data/Ta0.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    df1 = pd.read_csv('data/Ta1.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    df2 = pd.read_csv('data/Ta2.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    df3 = pd.read_csv('data/Ta3.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    df4 = pd.read_csv('data/Ta4.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    df5 = pd.read_csv('data/Ta5.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    df6 = pd.read_csv('data/Ta6.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    df7 = pd.read_csv('data/Ta7.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    df8 = pd.read_csv('data/Ta8.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    df9 = pd.read_csv('data/Ta9.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    df10 = pd.read_csv('data/Ta10.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    df11 = pd.read_csv('data/Ta11.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    df12 = pd.read_csv('data/Ta12.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    df13 = pd.read_csv('data/Ta13.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    df14 = pd.read_csv('data/Ta14.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    df15 = pd.read_csv('data/Ta15.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()

    plots(pltStr, ttlStr)
else:
    xedep0 = pd.read_csv('data/XeDep0.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    xedep1 = pd.read_csv('data/XeDep1.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    xedep2 = pd.read_csv('data/XeDep2.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    xedep3 = pd.read_csv('data/XeDep3.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    xedep4 = pd.read_csv('data/XeDep4.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    xedep5 = pd.read_csv('data/XeDep5.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    xedep6 = pd.read_csv('data/XeDep6.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    xedep7 = pd.read_csv('data/XeDep7.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    xedep8 = pd.read_csv('data/XeDep8.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    xedep9 = pd.read_csv('data/XeDep9.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    xedep10 = pd.read_csv('data/XeDep10.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    xedep11 = pd.read_csv('data/XeDep11.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    xedep12 = pd.read_csv('data/XeDep12.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    xedep13 = pd.read_csv('data/XeDep13.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    xedep14 = pd.read_csv('data/XeDep14.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    xedep15 = pd.read_csv('data/XeDep15.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()

    tadep0 = pd.read_csv('data/TaDep0.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    tadep1 = pd.read_csv('data/TaDep1.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    tadep2 = pd.read_csv('data/TaDep2.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    tadep3 = pd.read_csv('data/TaDep3.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    tadep4 = pd.read_csv('data/TaDep4.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    tadep5 = pd.read_csv('data/TaDep5.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    tadep6 = pd.read_csv('data/TaDep6.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    tadep7 = pd.read_csv('data/TaDep7.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    tadep8 = pd.read_csv('data/TaDep8.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    tadep9 = pd.read_csv('data/TaDep9.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    tadep10 = pd.read_csv('data/TaDep10.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    tadep11 = pd.read_csv('data/TaDep11.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    tadep12 = pd.read_csv('data/TaDep12.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    tadep13 = pd.read_csv('data/TaDep13.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    tadep14 = pd.read_csv('data/TaDep14.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
    tadep15 = pd.read_csv('data/TaDep15.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()

    df0 = pd.read_csv('data/Ta0.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    df1 = pd.read_csv('data/Ta1.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    df2 = pd.read_csv('data/Ta2.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    df3 = pd.read_csv('data/Ta3.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    df4 = pd.read_csv('data/Ta4.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    df5 = pd.read_csv('data/Ta5.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    df6 = pd.read_csv('data/Ta6.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    df7 = pd.read_csv('data/Ta7.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    df8 = pd.read_csv('data/Ta8.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    df9 = pd.read_csv('data/Ta9.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    df10 = pd.read_csv('data/Ta10.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    df11 = pd.read_csv('data/Ta11.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    df12 = pd.read_csv('data/Ta12.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    df13 = pd.read_csv('data/Ta13.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    df14 = pd.read_csv('data/Ta14.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
    df15 = pd.read_csv('data/Ta15.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()

    pYield = [len(df[:, 0]) / iterations for df in [df0, df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15]]
    idx = np.argmax(pYield)

    XeEDep_RMS = [np.sqrt(np.mean(edep ** 2)) / E0 for edep in [xedep0, xedep1, xedep2, xedep3, xedep4, xedep5, xedep6, xedep7, xedep8, 
                                                        xedep9, xedep10, xedep11, xedep12, xedep13, xedep14, xedep15]]
    
    TaEDep_RMS = [np.sqrt(np.mean(edep ** 2)) / E0 for edep in [tadep0, tadep1, tadep2, tadep3, tadep4, tadep5, tadep6, tadep7, tadep8, 
                                                        tadep9, tadep10, tadep11, tadep12, tadep13, tadep14, tadep15]]

    print(f'Xe EDep @ Max Yield:  {XeEDep_RMS[idx] * E0 :.2f} MeV')
    print(f'Ta EDep @ Max Yield: {TaEDep_RMS[idx] * E0 :.2f} MeV')

    plt.scatter(L_RL, TaEDep_RMS, label = '__nolegend__')
    plt.scatter(L_RL, XeEDep_RMS, label = '__nolegend__')
    plt.plot(L_RL[idx], XeEDep_RMS[idx], 'ro', label = f'Fractional Xenon Energy Deposition @ Max Yield: ~{XeEDep_RMS[idx]:.2f}')
    plt.plot(L_RL[idx], TaEDep_RMS[idx], 'bo', label = f' Fractional Tantalum Energy Deposition @ Max Yield: ~{TaEDep_RMS[idx]:.2f}')
    plt.vlines(L_RL[idx], 0, XeEDep_RMS[idx], colors = 'r', ls = ':', label = '__nolegend__')
    plt.vlines(L_RL[idx], 0, TaEDep_RMS[idx], colors = 'b', ls = ':', label = '__nolegend__')
    
    plt.xlabel('Radiation Lengths [$L_{RL}(Ta) = 0.4094$ cm | $L_{RL}(Xe) = 2.8720$ cm]')
    plt.ylabel('RMS Fractional Energy Deposition [/ incident $e^-$ @ 10 GeV]')
    plt.ylim(0, 5.75e-1)
    plt.legend(loc = 'upper left')
    plt.title('Energy Deposition vs Radiation Length')
    plt.show()

# df0 = pd.read_csv('build/out0_nt_e+.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
# df1 = pd.read_csv('build/out1_nt_e+.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
# df2 = pd.read_csv('build/out2_nt_e+.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
# df3 = pd.read_csv('build/out3_nt_e+.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
# df4 = pd.read_csv('build/out4_nt_e+.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
# df5 = pd.read_csv('build/out5_nt_e+.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
# df6 = pd.read_csv('build/out6_nt_e+.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
# df7 = pd.read_csv('build/out7_nt_e+.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
# df8 = pd.read_csv('build/out8_nt_e+.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
# df9 = pd.read_csv('build/out9_nt_e+.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
# df10 = pd.read_csv('build/out10_nt_e+.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
# df11 = pd.read_csv('build/out11_nt_e+.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
# df12 = pd.read_csv('build/out12_nt_e+.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
# df13 = pd.read_csv('build/out13_nt_e+.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
# df14 = pd.read_csv('build/out14_nt_e+.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()
# df15 = pd.read_csv('build/out15_nt_e+.csv', header = 6, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]']).to_numpy()