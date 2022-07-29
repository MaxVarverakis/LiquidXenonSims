# import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

LXe = 2.8720
LTa = 0.4094
L_RL = np.arange(.25, 4.25, .25)
iterations = 1e5
E_i = 1e4 # MeV
w = 5 # mm
E0 = .511 # MeV // Rest energy of positron
E_filter = 1e2

mode = 'Xe'
cutoff = False
disp = False
plot = True


def expectationValue(x):
    return np.mean(np.square(x)) - np.square(np.mean(x))

def calc_emittance(p, data):
    ib = data[:, 0] > E_filter

    px = p[ib, 0]
    tw = data[ib, 1]
    r = data[ib, 3]

    X = np.multiply(tw, np.cos(r))
    x2 = expectationValue(X)
    px2 = expectationValue(px)
    xpx = np.mean(np.multiply(X, px)) - np.mean(X) * np.mean(px)
    emittance = 1 / E0 * np.sqrt(x2 * px2 - xpx ** 2)

    return emittance

def plotArgs():
    dfs = [df0, df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15]
    ps = [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15]
    edeps = [edep0, edep1, edep2, edep3, edep4, edep5, edep6, edep7, edep8, edep9, edep10, edep11, edep12, edep13, edep14, edep15]

    pYield = [len(df[:, 0]) / iterations for df in dfs]
    # print(pYield)

    EDep_RMS = [np.sqrt(np.mean(edep ** 2)) / E_i for edep in edeps]
    E_RMS = [np.sqrt(np.mean(df[:, 0] ** 2)) for df in dfs]
    TW_RMS = [np.sqrt(np.mean(df[:, 1] ** 2)) for df in dfs]
    A_RMS = [np.sqrt(np.mean(df[:, 2] ** 2)) for df in dfs]
    # R_RMS = [np.sqrt(np.mean(df[:, 3] ** 2)) for df in dfs]

    idx = np.argmax(pYield)
    data = dfs[idx]
    P = ps[idx]
    # data = data[0]

    # TW_RMS = [TW_RMS[idx] if val > E_filter else None for idx,val in enumerate(E_RMS)]
    # A_RMS = [A_RMS[idx] if val > E_filter else None for idx,val in enumerate(E_RMS)]
    include_bool = data[:, 0] > E_filter
    
    pX = P[include_bool, 0]
    pY = P[include_bool, 1]
    pZ = P[include_bool, 2]
    pE = P[include_bool, 3]
    # pNorm = P[include_bool, 4]
    E = data[include_bool, 0]
    TW = data[include_bool, 1]
    A = data[include_bool, 2]
    R = data[include_bool, 3]

    # TW = [TW[idx] if val > E_filter else None for idx,val in enumerate(E)]
    # A = [A[idx] if val > E_filter else None for idx,val in enumerate(E)]
    # R = [R[idx] if val > E_filter else None for idx,val in enumerate(E)]
    # pX = [pX[idx] if val > E_filter else None for idx,val in enumerate(E)]
    # pY = [pY[idx] if val > E_filter else None for idx,val in enumerate(E)]
    # pZ = [pZ[idx] if val > E_filter else None for idx,val in enumerate(E)]
    # pE = [pE[idx] if val > E_filter else None for idx,val in enumerate(E)]

    ttlStr = '$L_{RL}(%s) = %.4f cm \\quad \\vert \\quad \\frac{d}{L_{LR}(%s)} \\approx$ %.2f \n Energy Cutoff: %.1f MeV' % (mode, LTarget, mode, L_RL[idx], E_filter)
    xlbl = 'Radiation Lengths [$L_{RL}(%s) = %.4f$ cm]' % (mode, LTarget)

    cm = 'inferno'
    bn = (75, 75)
    scale = 100
    
    # results = np.sqrt(np.square(pE) - (np.square(pX) + np.square(pY) + np.square(pZ)))
    # good_results = results[(.5 < results) & (results < .52)]
    # filtered_results = results[(results < .5) | (results > .52)]
    # print(len(good_results) / len(results))
    # print(len(filtered_results) / len(results))
    # print(pNorm)

    # plt.hist(results, bins = 100, label = '__nolegend__')
    # plt.hist(filtered_results, bins = 100, alpha = .75, label = 'Values > .53 and < .49')
    # plt.xlabel('$E^2 - (p_x^2 + p_y^2 + p_z^2)$')
    # plt.ylabel('Count')
    # plt.legend()
    # plt.show()
    # plt.scatter(E, pE)
    
    # print(sum(include_bool))

    # print(f'Emittance : {emittance} mm rad')

    # print(calc_emittance(p4, df4))

    # print(df0, p0)

    emittance = [calc_emittance(p, data) for p, data in zip(ps, dfs)]
    # print(emittance)

    return dfs, ps, edeps, pYield, EDep_RMS, E_RMS, TW_RMS, A_RMS, idx, data, P, include_bool, pX, pY, pZ, pE, E, TW, A, R, ttlStr, xlbl, cm, bn, scale, emittance

def compare(ts2 = ''):
    xlbl = 'Radiation Lengths [$L_{RL}(Ta) = 0.4094$ cm | $L_{RL}(Xe) = 2.8720$ cm]'

    Xedfs = [Xe0, Xe1, Xe2, Xe3, Xe4, Xe5, Xe6, Xe7, Xe8, Xe9, Xe10, Xe11, Xe12, Xe13, Xe14, Xe15]
    Tadfs = [Ta0, Ta1, Ta2, Ta3, Ta4, Ta5, Ta6, Ta7, Ta8, Ta9, Ta10, Ta11, Ta12, Ta13, Ta14, Ta15]
    Xeps = [Xep0, Xep1, Xep2, Xep3, Xep4, Xep5, Xep6, Xep7, Xep8, Xep9, Xep10, Xep11, Xep12, Xep13, Xep14, Xep15]
    Taps = [Tap0, Tap1, Tap2, Tap3, Tap4, Tap5, Tap6, Tap7, Tap8, Tap9, Tap10, Tap11, Tap12, Tap13, Tap14, Tap15]
    Xeedeps = [xedep0, xedep1, xedep2, xedep3, xedep4, xedep5, xedep6, xedep7, xedep8, 
                                                        xedep9, xedep10, xedep11, xedep12, xedep13, xedep14, xedep15]
    Taedeps = [tadep0, tadep1, tadep2, tadep3, tadep4, tadep5, tadep6, tadep7, tadep8, 
                                                        tadep9, tadep10, tadep11, tadep12, tadep13, tadep14, tadep15]

    XepYield = [len(df[:, 0]) / iterations for df in Xedfs]
    TapYield = [len(df[:, 0]) / iterations for df in Tadfs]
    idx = np.argmax(XepYield)

    XeE_RMS = [np.sqrt(np.mean(df[:, 0] ** 2)) for df in Xedfs]
    XeTW_RMS = [np.sqrt(np.mean(df[:, 1] ** 2)) for df in Xedfs]
    XeA_RMS = [np.sqrt(np.mean(df[:, 2] ** 2)) for df in Xedfs]
    
    TaE_RMS = [np.sqrt(np.mean(df[:, 0] ** 2)) for df in Tadfs]
    TaTW_RMS = [np.sqrt(np.mean(df[:, 1] ** 2)) for df in Tadfs]
    TaA_RMS = [np.sqrt(np.mean(df[:, 2] ** 2)) for df in Tadfs]

    XeEDep_RMS = [np.sqrt(np.mean(edep ** 2)) / E_i for edep in Xeedeps]
    
    TaEDep_RMS = [np.sqrt(np.mean(edep ** 2)) / E_i for edep in Taedeps]

    XeEmittance = [calc_emittance(p, data) for p, data in zip(Xeps, Xedfs)]
    TaEmittance = [calc_emittance(p, data) for p, data in zip(Taps, Tadfs)]

    plt.scatter(L_RL, TaEmittance, label = '__nolegend__')
    plt.scatter(L_RL, XeEmittance, label = '__nolegend__')
    plt.plot(L_RL[idx], XeEmittance[idx], 'ro', label = f'Xe Emittance @ Max Yield: ~{XeEmittance[idx]:.2f} mm$\cdot$rad')
    plt.plot(L_RL[idx], TaEmittance[idx], 'bo', label = f'Ta Emittance @ Max Yield: ~{TaEmittance[idx]:.2f} mm$\cdot$rad')
    plt.vlines(L_RL[idx], 0, XeEmittance[idx], colors = 'r', ls = ':', label = '__nolegend__')
    plt.vlines(L_RL[idx], 0, TaEmittance[idx], colors = 'b', ls = ':', label = '__nolegend__')
    plt.xlabel(xlbl)
    plt.ylabel('RMS Emittance [mm$\cdot$rad]')
    plt.legend(
        # loc = 'upper left'
        )
    plt.title(f'Emittance vs Radiation Length \n Energy Cutoff: {E_filter:.2f} MeV')

    plt.suptitle(ts2, fontsize = 14)
    plt.show()
    
    print(f'Xe EDep @ Max Yield:  {XeEDep_RMS[idx] * E_i :.2f} MeV')
    print(f'Ta EDep @ Max Yield: {TaEDep_RMS[idx] * E_i :.2f} MeV')

    plt.scatter(L_RL, TaEDep_RMS, label = '__nolegend__')
    plt.scatter(L_RL, XeEDep_RMS, label = '__nolegend__')
    plt.plot(L_RL[idx], XeEDep_RMS[idx], 'ro', label = f'Fractional Xenon Energy Deposition @ Max Yield: ~{XeEDep_RMS[idx]:.2f}')
    plt.plot(L_RL[idx], TaEDep_RMS[idx], 'bo', label = f' Fractional Tantalum Energy Deposition @ Max Yield: ~{TaEDep_RMS[idx]:.2f}')
    plt.vlines(L_RL[idx], 0, XeEDep_RMS[idx], colors = 'r', ls = ':', label = '__nolegend__')
    plt.vlines(L_RL[idx], 0, TaEDep_RMS[idx], colors = 'b', ls = ':', label = '__nolegend__')
    
    plt.xlabel(xlbl)
    plt.ylabel('RMS Fractional Energy Deposition [/ incident $e^-$ @ 10 GeV]')
    plt.ylim(0, 5.75e-1)
    plt.legend(loc = 'upper left')
    plt.title('Energy Deposition vs Radiation Length')
    
    plt.suptitle(ts2, fontsize = 14)
    plt.show()

    plt.subplot(1, 2, 1)
    plt.scatter(L_RL, TapYield, marker = '.', label = '__nolegend__')
    plt.scatter(L_RL, XepYield, marker = '.', label = '__nolegend__')
    plt.plot(L_RL[idx], XepYield[idx], 'ro', label = f'Xe Max Yield: ~{XepYield[idx]:.2f} / $e^-$')
    plt.plot(L_RL[idx], TapYield[idx], 'bo', label = f'Ta Max Yield: ~{TapYield[idx]:.2f} / $e^-$')
    plt.vlines(L_RL[idx], 0, TapYield[idx], colors = 'b', ls = ':', label = '__nolegend__')
    plt.vlines(L_RL[idx], 0, XepYield[idx], colors = 'r', ls = ':', label = '__nolegend__')

    # plt.xticks(list(plt.xticks()[0]) + [L_RL[idx]])
    # # a[np.where(a == 2.75)] = f'~{L_RL[idx] * LXe:.2f} cm'
    # jdx = np.where(plt.xticks()[0] == [L_RL[idx]][0])[0][0]
    # l = plt.xticks()[1]
    # l[jdx] = f'~{L_RL[idx] * LXe:.2f} cm'
    # plt.xticks(plt.xticks(), labels = l)
    # print(plt.xticks()[1])

    # plt.text(L_RL[idx], XepYield[idx] / 100, f'~{L_RL[idx] * LTarget:.2f} cm', fontsize = 12)
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
    plt.scatter(L_RL, TaE_RMS, marker = '.', label = '__nolegend__')
    plt.scatter(L_RL, XeE_RMS, marker = '.', label = '__nolegend__')
    plt.plot(L_RL[idx], XeE_RMS[idx], 'ro', alpha = .5, label = f'Xe RMS Energy @ Max Yield: ~{TaE_RMS[idx]:.2f} MeV')
    plt.plot(L_RL[idx], TaE_RMS[idx], 'bo', alpha = .5, label = f'Ta RMS Energy @ Max Yield: ~{XeE_RMS[idx]:.2f} MeV')
    plt.vlines(L_RL[idx], 0, TaE_RMS[idx], colors = 'b', ls = ':', label = '__nolegend__')
    plt.vlines(L_RL[idx], 0, XeE_RMS[idx], colors = 'r', ls = ':', label = '__nolegend__')
    # plt.text(L_RL[idx], XeE_RMS[idx] / 10, f'~{L_RL[idx] * LTarget:.2f} cm', fontsize = 12)
    # plt.hlines(E_RMS[idx], 0, L_RL[idx], colors = 'r', ls = ':', label = '__nolegend__')
    plt.xlim(0, 4.25)
    plt.ylim(0, 1300)
    # plt.ylim(0, max(E_RMS) * 1.05)
    plt.yticks(np.arange(0, max(TaE_RMS), 100))
    plt.xlabel(xlbl)
    plt.ylabel('RMS Positron Energy [MeV]')
    plt.legend()
    plt.title('RMS $e^+$ Energy vs Radiation Length')

    plt.suptitle(ts2, fontsize = 14)
    plt.show()

    plt.subplot(1, 2, 1)
    plt.scatter(L_RL, TaTW_RMS, marker = '.', label = '__nolegend__')
    plt.scatter(L_RL, XeTW_RMS, marker = '.', label = '__nolegend__')
    plt.plot(L_RL[idx], XeTW_RMS[idx], 'ro', label = f'Xe Traverse Width @ Max Yield: ~{XeTW_RMS[idx]:.2f} mm')
    plt.plot(L_RL[idx], TaTW_RMS[idx], 'bo', label = f'Ta Traverse Width @ Max Yield: ~{TaTW_RMS[idx]:.2f} mm')
    plt.vlines(L_RL[idx], 0, XeTW_RMS[idx], colors = 'r', ls = ':', label = '__nolegend__')
    plt.vlines(L_RL[idx], 0, TaTW_RMS[idx], colors = 'b', ls = ':', label = '__nolegend__')

    # plt.xticks(list(plt.xticks()[0]) + [L_RL[idx]])
    # # a[np.where(a == 2.75)] = f'~{L_RL[idx] * LXe:.2f} cm'
    # jdx = np.where(plt.xticks()[0] == [L_RL[idx]][0])[0][0]
    # l = plt.xticks()[1]
    # l[jdx] = f'~{L_RL[idx] * LXe:.2f} cm'
    # plt.xticks(plt.xticks(), labels = l)
    # print(plt.xticks()[1])

    # plt.text(L_RL[idx], XepYield[idx] / 100, f'~{L_RL[idx] * LTarget:.2f} cm', fontsize = 12)
    # plt.hlines(pYield[idx], 0, L_RL[idx], colors = 'r', ls = ':', label = '__nolegend__')
    plt.xlabel(xlbl)
    plt.ylabel('RMS Spot Size [mm]')
    plt.xlim(0, 4.25)
    plt.ylim(0, max(XeTW_RMS) * 1.05)
    # plt.yticks(np.arange(0, 2.5, .25))
    plt.legend(loc = 'upper left'
    # , bbox_to_anchor = (0, .95)
    )
    plt.title('RMS $e^+$ Shower Spot Size vs Radiation Length')

    plt.subplot(1, 2, 2)
    plt.scatter(L_RL, TaA_RMS, marker = '.', label = '__nolegend__')
    plt.scatter(L_RL, XeA_RMS, marker = '.', label = '__nolegend__')
    plt.plot(L_RL[idx], XeA_RMS[idx], 'ro', label = f'Xe Diffraction Angle @ Max Yield: ~{TaA_RMS[idx]:.2f} rad')
    plt.plot(L_RL[idx], TaA_RMS[idx], 'bo', label = f'Ta Diffraction Angle @ Max Yield: ~{XeA_RMS[idx]:.2f} rad')
    plt.vlines(L_RL[idx], 0, TaA_RMS[idx], colors = 'b', ls = ':', label = '__nolegend__')
    plt.vlines(L_RL[idx], 0, XeA_RMS[idx], colors = 'r', ls = ':', label = '__nolegend__')
    # plt.text(L_RL[idx], XeE_RMS[idx] / 10, f'~{L_RL[idx] * LTarget:.2f} cm', fontsize = 12)
    # plt.hlines(E_RMS[idx], 0, L_RL[idx], colors = 'r', ls = ':', label = '__nolegend__')
    plt.xlim(0, 4.25)
    plt.ylim(0, max(TaA_RMS) * 1.05)
    plt.xlabel(xlbl)
    plt.ylabel('RMS Angle [rad]')
    plt.legend()
    plt.title('RMS $e^+$ Angle vs Radiation Length')

    plt.suptitle(ts2, fontsize = 14)
    plt.show()

def plots(ts2 = '', spot = 'upper left'):

    dfs, ps, edeps, pYield, EDep_RMS, E_RMS, TW_RMS, A_RMS, idx, data, P, include_bool, pX, pY, pZ, pE, E, TW, A, R, ttlStr, xlbl, cm, bn, scale, emittance = plotArgs()

    plt.scatter(L_RL, emittance, label = '__nolegend__')
    plt.plot(L_RL[idx], emittance[idx], 'ro', label = f'Emittance @ Max Yield: ~{emittance[idx]:.2f} mm$\cdot$rad')
    plt.vlines(L_RL[idx], 0, emittance[idx], colors = 'r', ls = ':', label = '__nolegend__')
    plt.xlabel(xlbl)
    plt.ylabel('RMS Emittance [mm$\cdot$rad]')
    plt.legend(
        loc = spot
        )
    plt.title(f'Emittance vs Radiation Length \n Energy Cutoff: {E_filter:.2f} MeV')
    
    plt.suptitle(ts2, fontsize = 14)
    plt.show()

    # plt.subplot(1, 2, 1)
    # pHist2 = plt.hist2d(pX, pY, bins = 1000, 
    # # norm = mpl.colors.LogNorm(), 
    # cmap = cm)
    # plt.colorbar(pHist2[3], label = 'Counts')
    # plt.xlim(-15, 15)
    # plt.ylim(-15, 15)
    # plt.title('X-Y Momentum Distribution')
    # plt.xlabel('P$_x$')
    # plt.ylabel('P$_y$')

    # plt.subplot(1, 2, 2)
    # pHist1 = plt.hist2d(pX, pY, bins = 1000, 
    # norm = mpl.colors.LogNorm(), 
    # cmap = cm)
    # plt.colorbar(pHist1[3], label = 'Counts')
    # plt.xlim(-50, 50)
    # plt.ylim(-50, 50)
    # plt.title('Log Scale X-Y Momentum Distribution')
    # plt.xlabel('P$_x$')
    # plt.ylabel('P$_y$')
    
    # plt.suptitle(ttlStr, fontsize = 14)
    # plt.show()

    # plt.hist(R)
    # plt.xlabel('Rotational Angle [rad]')
    # plt.ylabel('Count')
    # plt.title(ttlStr)
    # plt.show()

    print(f'EDep @ Max Yield:  {EDep_RMS[idx] * E_i :.2f} MeV')
    plt.scatter(L_RL, EDep_RMS, label = '__nolegend__')
    plt.plot(L_RL[idx], EDep_RMS[idx], 'ro', label = f'Fractional Energy Deposition @ Max Yield: ~{EDep_RMS[idx]:.2f}')
    plt.vlines(L_RL[idx], 0, EDep_RMS[idx], colors = 'r', ls = ':', label = '__nolegend__')
    plt.xlabel(xlbl)
    plt.ylabel('RMS Fractional Energy Deposition [ / incident $e^-$ @ 10 GeV]')
    # plt.ylabel('RMS Energy Deposition / incident $e^-$ [MeV]')
    # plt.ylim(0, 5.75e-1)
    plt.legend(loc = 'upper left')
    plt.suptitle(ts2, fontsize = 14)
    
    plt.title('Energy Deposition vs Radiation Length')
    plt.show()
    
    # plt.scatter(L_RL, R_RMS)
    # plt.xlim(0, 4.25)
    # plt.ylabel('RMS Rotational Angle [rad]')
    # plt.xlabel(xlbl)
    # plt.show()

    plt.subplot(1, 2, 1)
    plt.hist2d(E, TW, bins = bn, weights = np.full(np.shape(TW), scale / iterations), norm = mpl.colors.LogNorm(), cmap = cm)
    plt.ylim(0, 10)
    plt.xlim(0, 7e3)
    plt.title('$e^+$ Shower Size vs $e^+$ Energy')
    plt.xlabel('Energy [MeV]')
    plt.ylabel('Spot Size [mm]')

    plt.subplot(1, 2, 2)
    h = plt.hist2d(E, A,  bins = bn, weights = np.full(np.shape(TW), scale / iterations), norm = mpl.colors.LogNorm(), cmap = cm)
    plt.colorbar(h[3], label = 'Counts / 100 incident $e^-$')
    plt.ylim(0, .25)
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
    plt.legend(
        # loc = 'upper left'
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

    p0 = pd.read_csv('data/XeP0.csv', header = 7, names = ['Px', 'Py', 'Pz', 'E [MeV]']).to_numpy()
    p1 = pd.read_csv('data/XeP1.csv', header = 7, names = ['Px', 'Py', 'Pz', 'E [MeV]']).to_numpy()
    p2 = pd.read_csv('data/XeP2.csv', header = 7, names = ['Px', 'Py', 'Pz', 'E [MeV]']).to_numpy()
    p3 = pd.read_csv('data/XeP3.csv', header = 7, names = ['Px', 'Py', 'Pz', 'E [MeV]']).to_numpy()
    p4 = pd.read_csv('data/XeP4.csv', header = 7, names = ['Px', 'Py', 'Pz', 'E [MeV]']).to_numpy()
    p5 = pd.read_csv('data/XeP5.csv', header = 7, names = ['Px', 'Py', 'Pz', 'E [MeV]']).to_numpy()
    p6 = pd.read_csv('data/XeP6.csv', header = 7, names = ['Px', 'Py', 'Pz', 'E [MeV]']).to_numpy()
    p7 = pd.read_csv('data/XeP7.csv', header = 7, names = ['Px', 'Py', 'Pz', 'E [MeV]']).to_numpy()
    p8 = pd.read_csv('data/XeP8.csv', header = 7, names = ['Px', 'Py', 'Pz', 'E [MeV]']).to_numpy()
    p9 = pd.read_csv('data/XeP9.csv', header = 7, names = ['Px', 'Py', 'Pz', 'E [MeV]']).to_numpy()
    p10 = pd.read_csv('data/XeP10.csv', header = 7, names = ['Px', 'Py', 'Pz', 'E [MeV]']).to_numpy()
    p11 = pd.read_csv('data/XeP11.csv', header = 7, names = ['Px', 'Py', 'Pz', 'E [MeV]']).to_numpy()
    p12 = pd.read_csv('data/XeP12.csv', header = 7, names = ['Px', 'Py', 'Pz', 'E [MeV]']).to_numpy()
    p13 = pd.read_csv('data/XeP13.csv', header = 7, names = ['Px', 'Py', 'Pz', 'E [MeV]']).to_numpy()
    p14 = pd.read_csv('data/XeP14.csv', header = 7, names = ['Px', 'Py', 'Pz', 'E [MeV]']).to_numpy()
    p15 = pd.read_csv('data/XeP15.csv', header = 7, names = ['Px', 'Py', 'Pz', 'E [MeV]']).to_numpy()

    df0 = pd.read_csv('data/Xe0.csv', header = 7, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]']).to_numpy()
    df1 = pd.read_csv('data/Xe1.csv', header = 7, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]']).to_numpy()
    df2 = pd.read_csv('data/Xe2.csv', header = 7, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]']).to_numpy()
    df3 = pd.read_csv('data/Xe3.csv', header = 7, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]']).to_numpy()
    df4 = pd.read_csv('data/Xe4.csv', header = 7, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]']).to_numpy()
    df5 = pd.read_csv('data/Xe5.csv', header = 7, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]']).to_numpy()
    df6 = pd.read_csv('data/Xe6.csv', header = 7, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]']).to_numpy()
    df7 = pd.read_csv('data/Xe7.csv', header = 7, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]']).to_numpy()
    df8 = pd.read_csv('data/Xe8.csv', header = 7, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]']).to_numpy()
    df9 = pd.read_csv('data/Xe9.csv', header = 7, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]']).to_numpy()
    df10 = pd.read_csv('data/Xe10.csv', header = 7, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]']).to_numpy()
    df11 = pd.read_csv('data/Xe11.csv', header = 7, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]']).to_numpy()
    df12 = pd.read_csv('data/Xe12.csv', header = 7, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]']).to_numpy()
    df13 = pd.read_csv('data/Xe13.csv', header = 7, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]']).to_numpy()
    df14 = pd.read_csv('data/Xe14.csv', header = 7, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]']).to_numpy()
    df15 = pd.read_csv('data/Xe15.csv', header = 7, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]']).to_numpy()

    # df0 = pd.read_csv('build/out0_nt_e+.csv', header = 7, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]']).to_numpy()
    # df1 = pd.read_csv('build/out1_nt_e+.csv', header = 7, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]']).to_numpy()
    # df2 = pd.read_csv('build/out2_nt_e+.csv', header = 7, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]']).to_numpy()
    # df3 = pd.read_csv('build/out3_nt_e+.csv', header = 7, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]']).to_numpy()
    # df4 = pd.read_csv('test/out4_nt_e+.csv', header = 7, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]']).to_numpy()
    # p4 = pd.read_csv('test/out4_nt_M4V.csv', header = 8, names = ['Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski Norm']).to_numpy()
    # df5 = pd.read_csv('build/out5_nt_e+.csv', header = 7, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]']).to_numpy()
    # df6 = pd.read_csv('build/out6_nt_e+.csv', header = 7, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]']).to_numpy()
    # df7 = pd.read_csv('build/out7_nt_e+.csv', header = 7, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]']).to_numpy()
    # df8 = pd.read_csv('build/out8_nt_e+.csv', header = 7, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]']).to_numpy()
    # df9 = pd.read_csv('build/out9_nt_e+.csv', header = 7, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]']).to_numpy()
    # df10 = pd.read_csv('build/out10_nt_e+.csv', header = 7, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]']).to_numpy()
    # df11 = pd.read_csv('build/out11_nt_e+.csv', header = 7, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]']).to_numpy()
    # df12 = pd.read_csv('build/out12_nt_e+.csv', header = 7, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]']).to_numpy()
    # df13 = pd.read_csv('build/out13_nt_e+.csv', header = 7, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]']).to_numpy()
    # df14 = pd.read_csv('build/out14_nt_e+.csv', header = 7, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]']).to_numpy()
    # df15 = pd.read_csv('build/out15_nt_e+.csv', header = 7, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]']).to_numpy()

    if plot:
        if cutoff:
            # spot = 'lower right'
            ts2 = f'Spot Size Cutoff : {w} mm'
            # ts2 = ''
            
            p0 = p0[df0[:, 1] <= w]
            p1 = p1[df1[:, 1] <= w]
            p2 = p2[df2[:, 1] <= w]
            p3 = p3[df3[:, 1] <= w]
            p4 = p4[df4[:, 1] <= w]
            p5 = p5[df5[:, 1] <= w]
            p6 = p6[df6[:, 1] <= w]
            p7 = p7[df7[:, 1] <= w]
            p8 = p8[df8[:, 1] <= w]
            p9 = p9[df9[:, 1] <= w]
            p10 = p10[df10[:, 1] <= w]
            p11 = p11[df11[:, 1] <= w]
            p12 = p12[df12[:, 1] <= w]
            p13 = p13[df13[:, 1] <= w]
            p14 = p14[df14[:, 1] <= w]
            p15 = p15[df15[:, 1] <= w]
            
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
            
            plots(ts2)
        else:
            plots()
    else:
        plotArgs()
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

    df0 = pd.read_csv('data/Ta0.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
    df1 = pd.read_csv('data/Ta1.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
    df2 = pd.read_csv('data/Ta2.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
    df3 = pd.read_csv('data/Ta3.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
    df4 = pd.read_csv('data/Ta4.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
    df5 = pd.read_csv('data/Ta5.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
    df6 = pd.read_csv('data/Ta6.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
    df7 = pd.read_csv('data/Ta7.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
    df8 = pd.read_csv('data/Ta8.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
    df9 = pd.read_csv('data/Ta9.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
    df10 = pd.read_csv('data/Ta10.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
    df11 = pd.read_csv('data/Ta11.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
    df12 = pd.read_csv('data/Ta12.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
    df13 = pd.read_csv('data/Ta13.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
    df14 = pd.read_csv('data/Ta14.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
    df15 = pd.read_csv('data/Ta15.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
    
    p0 = pd.read_csv('data/Ta0.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
    p1 = pd.read_csv('data/Ta1.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
    p2 = pd.read_csv('data/Ta2.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
    p3 = pd.read_csv('data/Ta3.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
    p4 = pd.read_csv('data/Ta4.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
    p5 = pd.read_csv('data/Ta5.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
    p6 = pd.read_csv('data/Ta6.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
    p7 = pd.read_csv('data/Ta7.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
    p8 = pd.read_csv('data/Ta8.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
    p9 = pd.read_csv('data/Ta9.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
    p10 = pd.read_csv('data/Ta10.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
    p11 = pd.read_csv('data/Ta11.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
    p12 = pd.read_csv('data/Ta12.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
    p13 = pd.read_csv('data/Ta13.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
    p14 = pd.read_csv('data/Ta14.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
    p15 = pd.read_csv('data/Ta15.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
    
    if plot:
        plots()
    else:
        plotArgs()
else:
    if disp:
        ttlStr = '$L_{RL}(Ta) = 0.4094 cm \\quad \\vert \\quad \\frac{d}{L_{LR}(Ta)} \\approx 2.75$ \n Energy Cutoff: 100 MeV'
    else:
        ttlStr = ''

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

    Ta0 = pd.read_csv('data/Ta0.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
    Ta1 = pd.read_csv('data/Ta1.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
    Ta2 = pd.read_csv('data/Ta2.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
    Ta3 = pd.read_csv('data/Ta3.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
    Ta4 = pd.read_csv('data/Ta4.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
    Ta5 = pd.read_csv('data/Ta5.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
    Ta6 = pd.read_csv('data/Ta6.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
    Ta7 = pd.read_csv('data/Ta7.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
    Ta8 = pd.read_csv('data/Ta8.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
    Ta9 = pd.read_csv('data/Ta9.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
    Ta10 = pd.read_csv('data/Ta10.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
    Ta11 = pd.read_csv('data/Ta11.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
    Ta12 = pd.read_csv('data/Ta12.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
    Ta13 = pd.read_csv('data/Ta13.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
    Ta14 = pd.read_csv('data/Ta14.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
    Ta15 = pd.read_csv('data/Ta15.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]

    Xe0 = pd.read_csv('data/Xe0.csv', header = 7, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]']).to_numpy()
    Xe1 = pd.read_csv('data/Xe1.csv', header = 7, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]']).to_numpy()
    Xe2 = pd.read_csv('data/Xe2.csv', header = 7, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]']).to_numpy()
    Xe3 = pd.read_csv('data/Xe3.csv', header = 7, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]']).to_numpy()
    Xe4 = pd.read_csv('data/Xe4.csv', header = 7, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]']).to_numpy()
    Xe5 = pd.read_csv('data/Xe5.csv', header = 7, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]']).to_numpy()
    Xe6 = pd.read_csv('data/Xe6.csv', header = 7, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]']).to_numpy()
    Xe7 = pd.read_csv('data/Xe7.csv', header = 7, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]']).to_numpy()
    Xe8 = pd.read_csv('data/Xe8.csv', header = 7, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]']).to_numpy()
    Xe9 = pd.read_csv('data/Xe9.csv', header = 7, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]']).to_numpy()
    Xe10 = pd.read_csv('data/Xe10.csv', header = 7, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]']).to_numpy()
    Xe11 = pd.read_csv('data/Xe11.csv', header = 7, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]']).to_numpy()
    Xe12 = pd.read_csv('data/Xe12.csv', header = 7, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]']).to_numpy()
    Xe13 = pd.read_csv('data/Xe13.csv', header = 7, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]']).to_numpy()
    Xe14 = pd.read_csv('data/Xe14.csv', header = 7, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]']).to_numpy()
    Xe15 = pd.read_csv('data/Xe15.csv', header = 7, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]']).to_numpy()

    Tap0 = pd.read_csv('data/Ta0.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
    Tap1 = pd.read_csv('data/Ta1.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
    Tap2 = pd.read_csv('data/Ta2.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
    Tap3 = pd.read_csv('data/Ta3.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
    Tap4 = pd.read_csv('data/Ta4.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
    Tap5 = pd.read_csv('data/Ta5.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
    Tap6 = pd.read_csv('data/Ta6.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
    Tap7 = pd.read_csv('data/Ta7.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
    Tap8 = pd.read_csv('data/Ta8.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
    Tap9 = pd.read_csv('data/Ta9.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
    Tap10 = pd.read_csv('data/Ta10.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
    Tap11 = pd.read_csv('data/Ta11.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
    Tap12 = pd.read_csv('data/Ta12.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
    Tap13 = pd.read_csv('data/Ta13.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
    Tap14 = pd.read_csv('data/Ta14.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
    Tap15 = pd.read_csv('data/Ta15.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]

    Xep0 = pd.read_csv('data/XeP0.csv', header = 7, names = ['Px', 'Py', 'Pz', 'E [MeV]']).to_numpy()
    Xep1 = pd.read_csv('data/XeP1.csv', header = 7, names = ['Px', 'Py', 'Pz', 'E [MeV]']).to_numpy()
    Xep2 = pd.read_csv('data/XeP2.csv', header = 7, names = ['Px', 'Py', 'Pz', 'E [MeV]']).to_numpy()
    Xep3 = pd.read_csv('data/XeP3.csv', header = 7, names = ['Px', 'Py', 'Pz', 'E [MeV]']).to_numpy()
    Xep4 = pd.read_csv('data/XeP4.csv', header = 7, names = ['Px', 'Py', 'Pz', 'E [MeV]']).to_numpy()
    Xep5 = pd.read_csv('data/XeP5.csv', header = 7, names = ['Px', 'Py', 'Pz', 'E [MeV]']).to_numpy()
    Xep6 = pd.read_csv('data/XeP6.csv', header = 7, names = ['Px', 'Py', 'Pz', 'E [MeV]']).to_numpy()
    Xep7 = pd.read_csv('data/XeP7.csv', header = 7, names = ['Px', 'Py', 'Pz', 'E [MeV]']).to_numpy()
    Xep8 = pd.read_csv('data/XeP8.csv', header = 7, names = ['Px', 'Py', 'Pz', 'E [MeV]']).to_numpy()
    Xep9 = pd.read_csv('data/XeP9.csv', header = 7, names = ['Px', 'Py', 'Pz', 'E [MeV]']).to_numpy()
    Xep10 = pd.read_csv('data/XeP10.csv', header = 7, names = ['Px', 'Py', 'Pz', 'E [MeV]']).to_numpy()
    Xep11 = pd.read_csv('data/XeP11.csv', header = 7, names = ['Px', 'Py', 'Pz', 'E [MeV]']).to_numpy()
    Xep12 = pd.read_csv('data/XeP12.csv', header = 7, names = ['Px', 'Py', 'Pz', 'E [MeV]']).to_numpy()
    Xep13 = pd.read_csv('data/XeP13.csv', header = 7, names = ['Px', 'Py', 'Pz', 'E [MeV]']).to_numpy()
    Xep14 = pd.read_csv('data/XeP14.csv', header = 7, names = ['Px', 'Py', 'Pz', 'E [MeV]']).to_numpy()
    Xep15 = pd.read_csv('data/XeP15.csv', header = 7, names = ['Px', 'Py', 'Pz', 'E [MeV]']).to_numpy()

    compare(ttlStr)