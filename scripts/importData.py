# import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit as cf

LXe = 2.8720
LTa = 0.4094
WRe = .3430
manual = 0.3504
L_RL = np.arange(.5, 8.5, .5)
iterations = 1e4 # number of electrons simulated
E_drive = 10e3 # MeV
w = 10 # mm
E0 = .511 # MeV // Rest energy of positron

tick_spacing = 1.

# cutoffs
lowE_filter = 2 # MeV
E_filter = 22 # MeV
trans_filter = 10 # mm

mode = 'comp'
component = 'X'
cutoff = False
disp = False
plot = True
beam = False
wBool = False

plt.rcParams.update({'font.size': 15})

seed = 9978 # Generated using numpy.random.randint(0, 1e5)
np.random.seed(5)

def expModel(x, a, b, c = 0, d = 0, e = 0):
    X = np.array(x)

    # return a * X * np.exp(-X**2 / b) + c * np.exp(-(X - d)**2 / e)
    return a * X * np.exp(-X**2 / (2 * b**2))

def expectationValue(x):
    return np.mean(np.square(x)) - np.square(np.mean(x))

def transverse_convolution(data, sigma = 6):
    xoffset = data[:, 1] * np.cos(data[:, 3])
    yoffset = data[:, 1] * np.sin(data[:, 3])
    std = sigma * np.random.randn(len(xoffset), 2)

    x = xoffset + std[:, 0]
    y = yoffset + std[:, 1]
    t = np.sqrt(x**2 + y**2)
    
    return t, x, y

def calc_conv_emittance(x, px):

    x2 = expectationValue(x)
    px2 = expectationValue(px)
    xpx = np.mean(np.multiply(x, px)) - np.mean(x) * np.mean(px)
    emittance = 1 / E0 * np.sqrt(x2 * px2 - xpx ** 2)

    return emittance

def calc_emittance(p, data, comp):
    # ib = data[:, 0] > E_filter
    tw = data[:, 1]
    r = data[:, 3]

    if comp == 'X':
        px = p[:, 0]
        X = np.multiply(tw, np.cos(r))
    else:
        px = p[:, 1]
        X = np.multiply(tw, np.sin(r))

    x2 = expectationValue(X)
    px2 = expectationValue(px)
    xpx = np.mean(np.multiply(X, px)) - np.mean(X) * np.mean(px)
    emittance = 1 / E0 * np.sqrt(x2 * px2 - xpx ** 2)

    return emittance

def plotArgs(windows = False):

    # plt.quiver(pos[:, 0], pos[:, 1], p[:, 0], p[:, 1], p[:, 2], cmap = 'Blues')
    # plt.show()

    raw_dfs = [df0, df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15]
    raw_ps = [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15]
    edeps = [edep0, edep1, edep2, edep3, edep4, edep5, edep6, edep7, edep8, edep9, edep10, edep11, edep12, edep13, edep14, edep15]

    # filter data
    convolved_trans = [transverse_convolution(df) for df in raw_dfs]
    # print(np.shape(convolved_trans), np.shape(raw_dfs))

    mask = [(df[:, 0] > lowE_filter) & (df[:, 0] < E_filter) & (convolved_trans[idx][0] < trans_filter) for idx,df in enumerate(raw_dfs)]
    # print(convolved_trans[0][0][mask[0]])

    # mask = [(df[:, 0] > lowE_filter) & (df[:, 0] < E_filter) & (df[:, 1] < trans_filter) for df in raw_dfs]
    # new_df = raw_dfs[0]
    dfs = [df[include] for df,include in zip(raw_dfs, mask)]
    ps = [p[include] for p,include in zip(raw_ps, mask)]
    cts = [[ct[0][include], ct[1][include], ct[2][include]] for ct,include in zip(convolved_trans, mask)]
    # print(np.shape(cts[0][1]), np.shape(ps[0][:, 0]))
    # print(calc_conv_emittance(cts[0][1], ps[0][:, 0]))

    conv_emittance = [calc_conv_emittance(x[1], px[:, 0]) for x, px in zip(cts, ps)]
    emittance = [calc_emittance(p, data, component) for p, data in zip(ps, dfs)]

    # print(conv_emittance)

    # print(pYield)
    # print(df0)

    # sigmas = []
    # for idx,_ in enumerate(L_RL):
    #     hist = plt.hist(transverse_convolution(dfs[idx]), bins = 500)
    #     plt.close()
    #     counts, bins = hist[0], hist[1]
    #     bin_centers = [0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)]

    #     fit, _ = cf(expModel, bin_centers, counts, p0 = [387, 6])
    #     # fit, _ = cf(expModel, bin_centers, counts, p0 = [387, 6, 150, 50, 3])
    #     sigmas.append(fit[1])

    #     # print(fit[1])

    #     # plt.plot(bin_centers, expModel(bin_centers, *fit))
    #     # plt.show()
    
    # plt.scatter(L_RL, sigmas)
    # plt.show()

    if beam:
        beamFactor = 1e-4
    else:
        beamFactor = 1

    EDep_RMS = [np.mean((edep[:, 0] * beamFactor)) / E_drive for edep in edeps]
    if windows:
        WInEDep_RMS = [np.mean((edep[:, 1] * beamFactor)) for edep in edeps]
        WOutEDep_RMS = [np.mean((edep[:, 2] * beamFactor)) for edep in edeps]
    else:
        WInEDep_RMS = np.zeros(len(EDep_RMS))
        WOutEDep_RMS = np.zeros(len(EDep_RMS))
    E_RMS = [np.mean(df[:, 0]) for df in dfs]
    # print([np.mean(df[:, 0]) for df in dfs])
    TW_RMS = [np.mean(df[:, 1]) for df in dfs]
    TW_Conv = [np.mean(transverse_convolution(df)) for df in dfs]
    # print(TW_RMS)
    # print(TW_Conv)
    # TW_RMS = [np.sqrt(np.mean(df[:, 1] ** 2)) for df in dfs]
    A_RMS = [np.mean(df[:, 2]) for df in dfs]
    # R_RMS = [np.sqrt(np.mean(df[:, 3] ** 2)) for df in dfs]
    
    pYield = [len(df[:, 0]) / iterations for df in dfs]

    # Extract parameters for minimum required yield
    # min_req_yield_idx = pYield.index(min([y for y in pYield if y >= 1.2]))
    # print(f'Yield : {pYield[min_req_yield_idx]:.2f}\nRL : {L_RL[min_req_yield_idx]}\nEdep : {EDep_RMS[min_req_yield_idx]:.2f}\nBe Dep : {WOutEDep_RMS[min_req_yield_idx]:.2f}\n')
    
    idx = np.argmax(pYield)
    data = dfs[idx]
    P = ps[idx]
    
    pX = P[:, 0]
    pY = P[:, 1]
    pZ = P[:, 2]
    pE = P[:, 3]
    # pNorm = P[mask[idx], 4]
    E = data[:, 0]
    TW = data[:, 1]
    A = data[:, 2]
    R = data[:, 3]

    # TW = [TW[idx] if val > E_filter else None for idx,val in enumerate(E)]
    # A = [A[idx] if val > E_filter else None for idx,val in enumerate(E)]
    # R = [R[idx] if val > E_filter else None for idx,val in enumerate(E)]
    # pX = [pX[idx] if val > E_filter else None for idx,val in enumerate(E)]
    # pY = [pY[idx] if val > E_filter else None for idx,val in enumerate(E)]
    # pZ = [pZ[idx] if val > E_filter else None for idx,val in enumerate(E)]
    # pE = [pE[idx] if val > E_filter else None for idx,val in enumerate(E)]

    if disp:
        ttlStr = '$L_{RL}(%s) = %.4f cm \\quad \\vert \\quad \\frac{d}{L_{LR}(%s)} \\approx$ %.2f \n Energy Cutoff: %.1f MeV' % (mode, LTarget, mode, L_RL[idx], E_filter)
    else:
        ttlStr = ''
    xlbl = 'Radiation Lengths [$L_{RL}(%s) = %.4f$ cm]' % (mode, LTarget)

    cm = 'inferno'
    bn = (175, 350)
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
    
    # print(sum(mask))

    # print(f'Emittance : {emittance} mm rad')

    # print(calc_emittance(p4, df4))

    # print(df0, p0)

    # print(emittance)

    # print(WOutEDep_RMS[6], L_RL[6])

    return dfs, ps, edeps, pYield, EDep_RMS, WInEDep_RMS, WOutEDep_RMS, E_RMS, TW_RMS, A_RMS, idx, data, P, mask, pX, pY, pZ, pE, E, TW, A, R, ttlStr, xlbl, cm, bn, scale, conv_emittance, TW_Conv

def compare(ts2 = ''):
    if disp:
        xlbl = 'Radiation Lengths [$L_{RL}(Ta) = 0.4094$ cm | $L_{RL}(Xe) = 2.8720$ cm \ $L_{RL}(W_{75}Re_{25}) = 2.8720$ cm]'
    else:
        xlbl = 'Radiation Lengths'


    raw_Xedfs = [Xe0, Xe1, Xe2, Xe3, Xe4, Xe5, Xe6, Xe7, Xe8, Xe9, Xe10, Xe11, Xe12, Xe13, Xe14, Xe15]
    raw_Tadfs = [Ta0, Ta1, Ta2, Ta3, Ta4, Ta5, Ta6, Ta7, Ta8, Ta9, Ta10, Ta11, Ta12, Ta13, Ta14, Ta15]
    raw_WRedfs = [WRe0, WRe1, WRe2, WRe3, WRe4, WRe5, WRe6, WRe7, WRe8, WRe9, WRe10, WRe11, WRe12, WRe13, WRe14, WRe15]
    raw_Xeps = [Xep0, Xep1, Xep2, Xep3, Xep4, Xep5, Xep6, Xep7, Xep8, Xep9, Xep10, Xep11, Xep12, Xep13, Xep14, Xep15]
    raw_Taps = [Tap0, Tap1, Tap2, Tap3, Tap4, Tap5, Tap6, Tap7, Tap8, Tap9, Tap10, Tap11, Tap12, Tap13, Tap14, Tap15]
    raw_WReps = [WRep0, WRep1, WRep2, WRep3, WRep4, WRep5, WRep6, WRep7, WRep8, WRep9, WRep10, WRep11, WRep12, WRep13, WRep14, WRep15]
    Xeedeps = [xedep0, xedep1, xedep2, xedep3, xedep4, xedep5, xedep6, xedep7, xedep8, 
                                                        xedep9, xedep10, xedep11, xedep12, xedep13, xedep14, xedep15]
    Taedeps = [tadep0, tadep1, tadep2, tadep3, tadep4, tadep5, tadep6, tadep7, tadep8, 
                                                        tadep9, tadep10, tadep11, tadep12, tadep13, tadep14, tadep15]
    WReedeps = [wreedep0, wreedep1, wreedep2, wreedep3, wreedep4, wreedep5, wreedep6, wreedep7, wreedep8, 
                                                        wreedep9, wreedep10, wreedep11, wreedep12, wreedep13, wreedep14, wreedep15]

    Xeconvolved_trans = [transverse_convolution(df) for df in raw_Xedfs]
    Taconvolved_trans = [transverse_convolution(df) for df in raw_Tadfs]
    WReconvolved_trans = [transverse_convolution(df) for df in raw_WRedfs]
    # print(np.shape(Xeconvolved_trans[0][0]))
    # print(Xeconvolved_trans[:][0][0])
    rawXeTW = [np.mean(trans[0]) for trans in Xeconvolved_trans]
    rawTaTW = [np.mean(trans[0]) for trans in Taconvolved_trans]
    rawWReTW = [np.mean(trans[0]) for trans in WReconvolved_trans]

    XeMask = [(df[:, 0] > lowE_filter) & (df[:, 0] < E_filter) & (Xeconvolved_trans[idx][0] < trans_filter) for idx,df in enumerate(raw_Xedfs)]
    TaMask = [(df[:, 0] > lowE_filter) & (df[:, 0] < E_filter) & (Taconvolved_trans[idx][0] < trans_filter) for idx,df in enumerate(raw_Tadfs)]
    WReMask = [(df[:, 0] > lowE_filter) & (df[:, 0] < E_filter) & (WReconvolved_trans[idx][0] < trans_filter) for idx,df in enumerate(raw_WRedfs)]
    
    Xects = [[ct[0][include], ct[1][include], ct[2][include]] for ct,include in zip(Xeconvolved_trans, XeMask)]
    Tacts = [[ct[0][include], ct[1][include], ct[2][include]] for ct,include in zip(Taconvolved_trans, TaMask)]
    WRects = [[ct[0][include], ct[1][include], ct[2][include]] for ct,include in zip(WReconvolved_trans, WReMask)]
    XeTW = [np.mean(trans[0]) for trans in Xects]
    TaTW = [np.mean(trans[0]) for trans in Tacts]
    WReTW = [np.mean(trans[0]) for trans in WRects]
    # print(np.shape(Xects[0]))
    # print(np.shape(Xects[0][0]))
    # print(rawTaTW)
    # print(TaTW)

    # dfs = [df[include] for df,include in zip(raw_dfs, mask)]
    # ps = [p[include] for p,include in zip(raw_ps, mask)]
    
    Xedfs = [df[include] for df,include in zip(raw_Xedfs, XeMask)]
    Tadfs = [df[include] for df,include in zip(raw_Tadfs, TaMask)]
    WRedfs = [df[include] for df,include in zip(raw_WRedfs, WReMask)]
    Xeps = [p[include] for p,include in zip(raw_Xeps, XeMask)]
    Taps = [p[include] for p,include in zip(raw_Taps, TaMask)]
    WReps = [p[include] for p,include in zip(raw_WReps, WReMask)]

    Xeconv_emittance = [calc_conv_emittance(x[1], px[:, 0]) for x, px in zip(Xects, Xeps)]
    Taconv_emittance = [calc_conv_emittance(x[1], px[:, 0]) for x, px in zip(Tacts, Taps)]
    WReconv_emittance = [calc_conv_emittance(x[1], px[:, 0]) for x, px in zip(WRects, WReps)]
    rawXeconv_emittance = [calc_conv_emittance(x[1], px[:, 0]) for x, px in zip(Xeconvolved_trans, raw_Xeps)]
    rawTaconv_emittance = [calc_conv_emittance(x[1], px[:, 0]) for x, px in zip(Taconvolved_trans, raw_Taps)]
    rawWReconv_emittance = [calc_conv_emittance(x[1], px[:, 0]) for x, px in zip(WReconvolved_trans, raw_WReps)]
    # print(f'Xe: {Xeconv_emittance}\n')
    # print(f'Ta: {Taconv_emittance}\n')

    rawXepYield = [len(df[:, 0]) / iterations for df in raw_Xedfs]
    rawTapYield = [len(df[:, 0]) / iterations for df in raw_Tadfs]
    rawWRepYield = [len(df[:, 0]) / iterations for df in raw_WRedfs]
    rawXeidx = np.argmax(rawXepYield)
    rawTaidx = np.argmax(rawTapYield)
    rawWReidx = np.argmax(rawWRepYield)

    XepYield = [len(df[:, 0]) / iterations for df in Xedfs]
    TapYield = [len(df[:, 0]) / iterations for df in Tadfs]
    WRepYield = [len(df[:, 0]) / iterations for df in WRedfs]
    # print('Xe:', XepYield)
    # print('Ta: ', TapYield)
    idx = np.argmax(XepYield)
    Xeidx = np.argmax(XepYield)
    Taidx = np.argmax(TapYield)
    WReidx = np.argmax(WRepYield)

    XeE = [np.mean(df[:, 0]) for df in Xedfs]
    # print(np.subtract(XeE_RMS, [np.sqrt(np.mean(df[:, 0]**2)) for df in Xedfs]))
    XeTW_RMS = [np.mean(df[:, 1]) for df in Xedfs]
    cutXeA = [np.mean(df[:, 2]) for df in Xedfs]
    XeA = [np.mean(df[:, 2]) for df in raw_Xedfs]
    
    TaE = [np.mean(df[:, 0]) for df in Tadfs]
    TaTW_RMS = [np.mean(df[:, 1]) for df in Tadfs]
    cutTaA = [np.mean(df[:, 2]) for df in Tadfs]
    TaA = [np.mean(df[:, 2]) for df in raw_Tadfs]
    
    WReE = [np.mean(df[:, 0]) for df in WRedfs]
    WReTW_RMS = [np.mean(df[:, 1]) for df in WRedfs]
    cutWReA = [np.mean(df[:, 2]) for df in WRedfs]
    WReA = [np.mean(df[:, 2]) for df in raw_WRedfs]

    rawXeE = [np.mean(df[:, 0]) for df in raw_Xedfs]
    rawTaE = [np.mean(df[:, 0]) for df in raw_Tadfs]
    rawWReE = [np.mean(df[:, 0]) for df in raw_WRedfs]

    # XeEDep_RMS = [np.mean(edep[:, 0]) / E_drive for edep in Xeedeps]
    XeEDep_RMS = [np.mean(edep[:, 0]) for edep in Xeedeps]
    
    TaEDep_RMS = [np.mean(edep[:, 0]) / E_drive for edep in Taedeps]
    
    WReEDep_RMS = [np.mean(edep[:, 0]) / E_drive for edep in WReedeps]

    # xXeEmittance = [calc_emittance(p, data, 'X') for p, data in zip(Xeps, Xedfs)]
    # xTaEmittance = [calc_emittance(p, data, 'X') for p, data in zip(Taps, Tadfs)]
    # yXeEmittance = [calc_emittance(p, data, 'Y') for p, data in zip(Xeps, Xedfs)]
    # yTaEmittance = [calc_emittance(p, data, 'Y') for p, data in zip(Taps, Tadfs)]
    
    rawyieldLim = max(max(rawXepYield), max(rawTapYield), max(rawWRepYield)) * 1.15
    yieldLim = max(max(XepYield), max(TapYield), max(WRepYield)) * 1.15
    ELim = max(max(rawXeE), max(rawTaE), max(rawWReE)) * 1.05
    rawELim = max(max(rawXeE), max(rawTaE), max(rawWReE)) * 1.05
    TWLim = max(max(rawXeTW), max(rawTaTW), max(rawWReTW)) * 1.05
    ALim = max(max(XeA), max(TaA), max(WReA)) * 1.05
    EmitLim = max(max(rawXeconv_emittance), max(rawTaconv_emittance), max(rawWReconv_emittance)) * 1.05

    # print(L_RL[Xeidx], Xeidx)
    # print(Xe10[0:5, 0])
    # print(max(Xe10[:, 0]))
    nTa, bins, _ = plt.hist(Ta10[:, 0], bins = 75, range = (0, 200), label = 'Ta', facecolor = 'tab:blue', edgecolor = 'tab:blue')
    n_LXe, _, _ = plt.hist(Xe10[:, 0], bins = 75, range = (0, 200), label = 'LXe', alpha = .85, facecolor = 'tab:orange', edgecolor = 'tab:orange')
    n_WRe, _, _ = plt.hist(WRe10[:, 0], bins = 75, range = (0, 200), label = 'W$_{75}$Re$_{25}$', alpha = 1, facecolor = 'none', edgecolor = 'k')
    plt.ticklabel_format(style = 'sci', axis = 'y', scilimits = (0, 0))
    plt.legend()
    plt.xlim(0, 200)
    plt.xlabel('Outgoing $e^+$ Energy [MeV]')
    plt.ylabel('$e^+$ per $N_0$ Incident $e^-$')
    plt.show()
    # print(bins[1] - bins[0])

    # ax = plt.figure().add_subplot(projection = '3d')
    # # ax.scatter(xTaEmittance, yTaEmittance, L_RL, label = 'Tantalum')
    # # ax.scatter(xXeEmittance, yXeEmittance, L_RL, label = 'Liquid Xenon')
    # # ax.plot(xXeEmittance[idx], yXeEmittance[idx], L_RL[idx], 'ro', label = f'Xe Emittance @ Max Yield: ~{xXeEmittance[idx]:.2f} mm$\cdot$rad')
    # # ax.plot(xTaEmittance[idx], yTaEmittance[idx], L_RL[idx], 'bo', label = f'Ta Emittance @ Max Yield: ~{xTaEmittance[idx]:.2f} mm$\cdot$rad')
    # ax.set_xlabel('Radiation Lengths')
    # ax.set_ylabel('$e^+$ Yield')
    # ax.set_zlabel('Window E$_{\textrm{dep}}$')
    # # ax.legend()
    # plt.title(f'')
    # plt.show()

    # plt.scatter(L_RL, xTaEmittance, label = '__nolegend__')
    # plt.scatter(L_RL, xXeEmittance, label = '__nolegend__')
    # plt.plot(L_RL[idx], xXeEmittance[idx], 'ro', label = f'Xe Emittance @ Max Yield: ~{xXeEmittance[idx]:.2f} mm$\cdot$rad')
    # plt.plot(L_RL[idx], xTaEmittance[idx], 'bo', label = f'Ta Emittance @ Max Yield: ~{xTaEmittance[idx]:.2f} mm$\cdot$rad')
    # plt.vlines(L_RL[idx], 0, xXeEmittance[idx], colors = 'r', ls = ':', label = '__nolegend__')
    # plt.vlines(L_RL[idx], 0, xTaEmittance[idx], colors = 'b', ls = ':', label = '__nolegend__')
    # plt.xlabel(xlbl)
    # plt.ylabel('Normalized RMS Emittance [mm$\cdot$rad]')
    # plt.legend(
    #     # loc = 'upper left'
    #     )
    # plt.title(f'{component}-Emittance vs Radiation Length \n Energy Cutoff: {E_filter:.2f} MeV')

    # plt.suptitle(ts2, fontsize = 14)
    # plt.show()
    
    # print(f'Xe Emittance @ Max Yield:  {Xeconv_emittance[Xeidx]:.2f} mm rad')
    # print(f'Ta Emittance @ Max Yield: {Taconv_emittance[Taidx]:.2f} mm rad')
    # print(f'Xe Emittance @ Max Yield:  {xXeEmittance[Xeidx]:.2f} mm rad')
    # print(f'Ta Emittance @ Max Yield: {xTaEmittance[Taidx]:.2f} mm rad')
    # print(f'Xe EDep @ Max Yield:  {XeEDep_RMS[Xeidx] * E_drive :.2f} MeV')
    # print(f'Ta EDep @ Max Yield: {TaEDep_RMS[Taidx] * E_drive :.2f} MeV')
    
    plt.scatter(L_RL, XeEDep_RMS, color = 'k', marker = 'o', facecolor = 'none', label = '__nolegend__')
    # plt.scatter(L_RL, WReEDep_RMS, color = 'tab:orange', marker = '^', facecolor = 'none', label = '__nolegend__')
    # plt.scatter(L_RL[Xeidx], XeEDep_RMS[Xeidx], color = 'k', marker = 'o', label = f'EDep@ Max Yield: ~{XeEDep_RMS[Xeidx]:.2f}')
    plt.xlabel(xlbl)
    plt.ylabel('Energy Deposition [MeV]')
    # plt.ylim(0, 5.75e-1)
    # plt.legend(loc = 'upper left')
    # plt.title('Energy Deposition vs Target Width')
    
    # plt.suptitle(ts2, fontsize = 14)
    plt.close()

    # Positron Distribution
    scale = 1
    RL = 10
    # h = plt.hist2d(transverse_convolution(Xe10)[1], transverse_convolution(Xe10)[2], bins = 250, weights = np.full(np.shape(Xeconvolved_trans[RL][1]), scale / iterations), norm = mpl.colors.LogNorm(), cmap = 'inferno')
    h = plt.hist2d(Xeconvolved_trans[RL][1], Xeconvolved_trans[RL][2], bins = 250, weights = np.full(np.shape(Xeconvolved_trans[RL][1]), scale / iterations), norm = mpl.colors.LogNorm(), cmap = 'inferno')
    plt.colorbar(h[3], label = '$e^+$ / incident $e^-$')
    # plt.hist2d(Xeconvolved_trans[RL][1], Xeconvolved_trans[RL][2], bins = 100, weights = np.full(np.shape(Xeconvolved_trans[RL][1]), scale / iterations), norm = mpl.colors.LogNorm(), cmap = 'inferno')
    # plt.scatter(Xeconvolved_trans[RL][1], Xeconvolved_trans[RL][2], color = 'k', marker = 'o', facecolor = 'none')
    plt.xlabel('X [mm]')
    plt.ylabel('Y [mm]')
    plt.title('LXe $e^+$ Distribution at Target Exit \n at 5.5 Radiation Lengths')
    plt.close()

    plt.scatter(L_RL, TaA, color = 'b', marker = 'o', facecolor = 'none', label = 'Ta')
    plt.scatter(L_RL, XeA, color = 'r', marker = 'v', facecolor = 'none', label = 'Xe')
    plt.scatter(L_RL, WReA, color = 'g', marker = 's', facecolor = 'none', label = 'W$_{75}$Re$_{25}$')
    plt.xlabel(xlbl)
    plt.ylabel('Angular Divergence [rad]')
    plt.legend(loc = 'upper left')
    # plt.title('Energy Deposition vs Target Width')
    # plt.suptitle(ts2, fontsize = 14)
    plt.close()

    # plt.scatter(L_RL, TaEDep_RMS, marker = '.', label = '__nolegend__')
    plt.scatter(L_RL, TaEDep_RMS, color = 'b', marker = 'o', facecolor = 'none', label = 'Ta')
    # plt.scatter(L_RL[Taidx], TaEDep_RMS[Taidx], color = 'b', marker = 'o', label = f'Ta @ Max Yield: ~{TaEDep_RMS[Taidx]:.2f}')
    plt.scatter(L_RL, XeEDep_RMS, color = 'r', marker = 'v', facecolor = 'none', label = 'Xe')
    plt.scatter(L_RL, WReEDep_RMS, color = 'g', marker = '^', facecolor = 'none', label = 'W$_{75}$Re$_{25}$')
    # plt.scatter(L_RL[Xeidx], XeEDep_RMS[Xeidx], color = 'r', marker = 'v', label = f'Xe @ Max Yield: ~{XeEDep_RMS[Xeidx]:.2f}')
    # plt.plot(L_RL[Taidx], TaEDep_RMS[Taidx], 'b.', label = f'Ta @ Max Yield: ~{TaEDep_RMS[Taidx]:.2f}')
    # plt.vlines(L_RL[Xeidx], 0, XeEDep_RMS[Xeidx], colors = 'k', ls = ':', label = '__nolegend__')
    # plt.vlines(L_RL[Taidx], 0, TaEDep_RMS[Taidx], colors = 'b', ls = ':', label = '__nolegend__')
    plt.xlabel(xlbl)
    plt.ylabel('Mean Energy Deposition [MeV]')
    # plt.ylim(0, 5.75e-1)
    plt.legend(loc = 'upper left')
    # plt.title('Energy Deposition vs Target Width')
    
    # plt.suptitle(ts2, fontsize = 14)
    plt.close()

    plt.scatter(L_RL, rawTapYield, color = 'b', marker = 'o', facecolors = 'none', label = '__nolegend__')
    plt.scatter(L_RL, rawXepYield, color = 'tab:orange', marker = 'v', facecolors = 'none', label = '__nolegend__')
    plt.scatter(L_RL, rawWRepYield, color = 'g', marker = '^', facecolors = 'none', label = '__nolegend__')
    # plt.vlines(L_RL[rawTaidx], 0, rawTapYield[rawTaidx], colors = 'b', ls = ':', label = '__nolegend__')
    # plt.vlines(L_RL[rawXeidx], 0, rawXepYield[rawXeidx], colors = 'r', ls = ':', label = '__nolegend__')
    plt.scatter(L_RL[rawTaidx], rawTapYield[rawTaidx], color = 'b', marker = 'o', facecolors = 'none', label = f'Ta @ Max Yield: {rawTapYield[rawTaidx]:.2f}')
    plt.scatter(L_RL[rawWReidx], rawWRepYield[rawWReidx], color = 'g', marker = '^', facecolors = 'none', label = f'WRe @ Max Yield: {rawWRepYield[rawWReidx]:.2f}')
    plt.scatter(L_RL[rawXeidx], rawXepYield[rawXeidx], color = 'tab:orange', marker = 'v', facecolors = 'none', label = f'LXe @ Max Yield: {rawXepYield[rawXeidx]:.2f}')

    plt.scatter(L_RL, TapYield, color = 'm', marker = 's', facecolors = 'none', label = '__nolegend__')
    plt.scatter(L_RL, XepYield, color = 'k', facecolors = 'none', marker = 'D', label = '__nolegend__')
    plt.scatter(L_RL, WRepYield, color = 'r', facecolors = 'none', marker = 'P', label = '__nolegend__')
    # plt.vlines(L_RL[Taidx], 0, TapYield[Taidx], colors = 'g', ls = ':', label = '__nolegend__')
    # plt.vlines(L_RL[Xeidx], 0, XepYield[Xeidx], colors = 'k', ls = ':', label = '__nolegend__')
    plt.scatter(L_RL[Taidx], TapYield[Taidx], color = 'm', marker = 's', facecolors = 'none', label = f'Cut Ta @ Max Yield: {TapYield[Taidx]:.2f}')
    plt.scatter(L_RL[WReidx], WRepYield[WReidx], color = 'r', facecolors = 'none', marker = 'P', label = f'Cut WRe @ Max Yield: {WRepYield[WReidx]:.2f}')
    plt.scatter(L_RL[Xeidx], XepYield[Xeidx], color = 'k', facecolors = 'none', marker = 'D', label = f'Cut LXe @ Max Yield: {XepYield[Xeidx]:.2f}')
    # plt.vlines([L_RL[rawTaidx], L_RL[Taidx]], [0, 0], [rawTapYield[rawTaidx], TapYield[Taidx]], colors = 'k', alpha = .65, ls = ':', label = '__nolegend__')

    # plt.grid(True)
    plt.xlabel(xlbl, fontsize = 25)
    plt.ylabel('Positron Yield per Incident $e^-$', fontsize = 25)
    plt.xlim(0, 8.25)
    plt.ylim(0, rawyieldLim)
    # plt.ylim(0, max(pYield) * 1.05)
    plt.yticks(np.arange(0, rawyieldLim, 2 * tick_spacing))
    plt.legend(
        loc = 'upper left'
        , prop = {'size': 15}
        # , bbox_to_anchor = (0, .95)
    )

    # plt.title('$e^+$ Yield vs Target Width')
    plt.show()


    ThreeYield = [0.0734, 0.2795, 0.5555, 0.8958, 1.2182, 1.4917, 1.6664, 1.7734, 1.7268, 1.6523, 1.5122, 1.3387, 1.2015, 1.0226, 0.8517, 0.7052]
    ThreeDep = [8.466318395, 29.499465427, 71.90428823200001, 137.693837301, 235.837895449, 359.105902964, 503.8495554990001, 663.2790885200001, 833.870772774, 1009.6561966700001, 1177.03120647, 1332.95452443, 1494.1037825999997, 1626.4981914999998, 1755.7057912999999, 1872.0091575]
    SixYield = [0.0774, 0.3053, 0.6778, 1.1719, 1.7363, 2.2791, 2.759, 3.0589, 3.2081, 3.2644, 3.1468, 2.9441, 2.6691, 2.3968, 2.1276, 1.8234]
    SixDep = [8.87625778, 32.453977965, 83.99451510599998, 171.40446013, 308.07443162000004, 488.06674472000003, 715.50018046, 980.72648032, 1274.850875865, 1601.9226579, 1908.20147123, 2240.68833926, 2559.8160849, 2847.518059, 3121.6878680100003, 3384.12926126]
    TenYield = [0.0826, 0.3338, 0.7967, 1.4, 2.1842, 2.9992, 3.7727, 4.3357, 4.8376, 5.0349, 5.0792, 4.9288, 4.6499, 4.3002, 3.8855, 3.4306]
    TenDep = [9.10452495, 35.160500846, 93.1288448, 198.62779838, 367.26810605000003, 599.2267310740001, 918.7654689200001, 1287.4317261300002, 1720.9499592699997, 2199.30720334, 2695.5264249899997, 3219.60705819, 3712.2559959, 4245.62200209, 4723.632987700001, 5162.6777615]

    # data = {'3 GeV Yield': ThreeYield, '3 GeV EDep': ThreeDep, '6 GeV Yield': SixYield, '6 GeV EDep': SixDep, '10 GeV Yield': TenYield, '10 GeV EDep': TenDep}
    # new = pd.DataFrame(data, index = L_RL)
    # new.to_excel("Yield_EDep_data.xlsx")
    # print(new)


    # print(XepYield)
    # print(XeEDep_RMS)

    ThreeIdx = np.argmax(ThreeYield)
    SixIdx = np.argmax(SixYield)
    TenIdx = np.argmax(TenYield)

    # print(f'3 GeV Max Yield : {max(ThreeYield):.2f} \t EDep : {ThreeDep[ThreeIdx]:.2f} \t RL : {L_RL[ThreeIdx]}')
    # print(f' 3 GeV Yield : {ThreeYield[6]:.2f} \t EDep : {ThreeDep[6]:.2f} \t RL : {L_RL[6]} \t EDep Reduction : {1e2 * (1 - ThreeDep[6] / ThreeDep[ThreeIdx]):.2f} % \t Yield Reduction : {1e2 * (1 - ThreeYield[6] / max(ThreeYield)):.2f} %')

    # print(f' 3 GeV Yield : {ThreeYield[6]:.2f} \t Prop Dep : {ThreeDep[6] / 3e3:.4f} \t RL : {L_RL[6]} \t EDep Reduction : {1e2 * (1 - ThreeDep[6] / ThreeDep[ThreeIdx]):.2f} % \t Yield Reduction : {1e2 * (1 - ThreeYield[6] / max(ThreeYield)):.2f} %')
    # print(f' 6 GeV Yield : {SixYield[7]:.2f} \t Prop Dep : {SixDep[7] / 6e3:.4f} \t RL : {L_RL[7]} \t EDep Reduction : {1e2 * (1 - SixDep[7] / SixDep[SixIdx]):.2f} % \t Yield Reduction : {1e2 * (1 - SixYield[7] / max(SixYield)):.2f} %')
    # print(f'10 GeV Yield : {TenYield[8]:.2f} \t Prop Dep : {TenDep[8] / 10e3:.4f} \t RL : {L_RL[8]} \t EDep Reduction : {1e2 * (1 - TenDep[8] / TenDep[TenIdx]):.2f} % \t Yield Reduction : {1e2 * (1 - TenYield[8] / max(TenYield)):.2f} %')

    plt.scatter(ThreeYield, ThreeDep, color = 'b', marker = 'o', facecolors = 'none', label = '3 GeV')
    plt.scatter(SixYield, SixDep, color = 'r', marker = 'v', facecolors = 'none', label = '6 GeV')
    plt.scatter(TenYield, TenDep, color = 'k', facecolors = 'none', marker = 'D', label = '10 GeV')
    # plt.scatter(ThreeYield[6], ThreeDep[6], color = 'b', marker = 'o', facecolors = 'b', label = f'{L_RL[6]} Radiation Lengths')
    # plt.scatter(SixYield[7], SixDep[7], color = 'r', marker = 'v', facecolors = 'r', label = f'{L_RL[7]} Radiation Lengths')
    # plt.scatter(TenYield[8], TenDep[8], color = 'k', facecolors = 'k', marker = 'D', label = f'{L_RL[8]} Radiation Lengths')
    plt.scatter(ThreeYield[6], ThreeDep[6], color = 'b', marker = 'o', facecolors = 'b', label = '__nolegend__')
    plt.scatter(SixYield[7], SixDep[7], color = 'r', marker = 'v', facecolors = 'r', label = '__nolegend__')
    plt.scatter(TenYield[8], TenDep[8], color = 'k', facecolors = 'k', marker = 'D', label = '__nolegend__')
    plt.legend()
    plt.xlabel('Positron Yield per Incident $e^-$')
    plt.ylabel('Mean Energy Deposition [MeV]')
    plt.close()

    # print(XepYield)
    # print(XeEDep_RMS)
    # plt.scatter(rawXepYield, XeEDep_RMS, color = 'r', marker = 'v', facecolors = 'none', label = 'Raw Yield')
    plt.scatter(XepYield, XeEDep_RMS, color = 'k', facecolors = 'none', marker = 'D', label = 'After Cuts')
    plt.legend()
    plt.xlabel('Positron Yield')
    plt.ylabel('Mean Energy Deposition [MeV]')
    plt.close()
    

    plt.subplot(1, 2, 1)
    plt.scatter(L_RL, rawTapYield, color = 'b', marker = 'o', facecolors = 'none', label = '__nolegend__')
    plt.scatter(L_RL, rawXepYield, color = 'r', marker = 'v', facecolors = 'none', label = '__nolegend__')
    # plt.vlines(L_RL[rawTaidx], 0, rawTapYield[rawTaidx], colors = 'b', ls = ':', label = '__nolegend__')
    # plt.vlines(L_RL[rawXeidx], 0, rawXepYield[rawXeidx], colors = 'r', ls = ':', label = '__nolegend__')
    plt.scatter(L_RL[rawTaidx], rawTapYield[rawTaidx], color = 'b', marker = 'o', facecolors = 'none', label = f'Ta Max Yield: {rawTapYield[rawTaidx]:.2f}')
    plt.scatter(L_RL[rawXeidx], rawXepYield[rawXeidx], color = 'r', marker = 'v', facecolors = 'none', label = f'LXe Max Yield: {rawXepYield[rawXeidx]:.2f}')

    plt.scatter(L_RL, TapYield, color = 'g', marker = 's', facecolors = 'none', label = '__nolegend__')
    plt.scatter(L_RL, XepYield, color = 'k', facecolors = 'none', marker = 'D', label = '__nolegend__')
    # plt.vlines(L_RL[Taidx], 0, TapYield[Taidx], colors = 'g', ls = ':', label = '__nolegend__')
    # plt.vlines(L_RL[Xeidx], 0, XepYield[Xeidx], colors = 'k', ls = ':', label = '__nolegend__')
    plt.scatter(L_RL[Taidx], TapYield[Taidx], color = 'g', marker = 's', facecolors = 'none', label = f'Cut Ta Max Yield: {TapYield[Taidx]:.2f}')
    plt.scatter(L_RL[Xeidx], XepYield[Xeidx], color = 'k', facecolors = 'none', marker = 'D', label = f'Cut LXe Max Yield: {XepYield[Xeidx]:.2f}')
    # plt.vlines([L_RL[rawTaidx], L_RL[Taidx]], [0, 0], [rawTapYield[rawTaidx], TapYield[Taidx]], colors = 'k', alpha = .65, ls = ':', label = '__nolegend__')

    # plt.grid(True)
    plt.xlabel(xlbl)
    plt.ylabel('Positron Yield')
    plt.xlim(0, 8.25)
    plt.ylim(0, yieldLim)
    # plt.ylim(0, max(pYield) * 1.05)
    plt.yticks(np.arange(0, rawyieldLim, 2 * tick_spacing))
    plt.legend(
        loc = 'upper left'
        , prop={'size': 10}
        # , bbox_to_anchor = (0, .95)
    )

    # plt.title('$e^+$ Yield vs Target Width')
    
    plt.subplot(1, 2, 2)
    _, bins, _ = plt.hist(Ta10[:, 0], bins = 75, range = (0, 200), label = 'Ta')
    plt.hist(Xe10[:, 0], bins = 75, range = (0, 200), label = 'LXe', alpha = 1)
    plt.ticklabel_format(style = 'sci', axis = 'y', scilimits = (0, 0))
    plt.legend()
    plt.xlim(0, 200)
    plt.xlabel('Outgoing $e^+$ Energy [MeV]')
    plt.ylabel('$e^+$ per 10,000 Incident $e^-$')
    plt.close()
    # plt.scatter(L_RL, rawTaE, color = 'b', marker = 'o', facecolors = 'none', label = '__nolegend__')
    # plt.scatter(L_RL, rawXeE, color = 'r', marker = 'v', facecolors = 'none', label = '__nolegend__')
    # # plt.vlines(L_RL[rawTaidx], 0, rawTapYield[rawTaidx], colors = 'b', ls = ':', label = '__nolegend__')
    # # plt.vlines(L_RL[rawXeidx], 0, rawXepYield[rawXeidx], colors = 'r', ls = ':', label = '__nolegend__')
    # # plt.scatter(L_RL[rawTaidx], rawTaE[rawTaidx], color = 'b', marker = 'o', facecolors = 'none', label = f'Ta Max Yield: {rawTaE[rawTaidx]:.2f}')
    # # plt.scatter(L_RL[rawXeidx], rawXeE[rawXeidx], color = 'r', marker = 'v', facecolors = 'none', label = f'LXe Max Yield: {rawXeE[rawXeidx]:.2f}')

    # plt.scatter(L_RL, TaE, color = 'g', marker = 's', facecolors = 'none', label = '__nolegend__')
    # plt.scatter(L_RL, XeE, color = 'k', facecolors = 'none', marker = 'D', label = '__nolegend__')
    # # plt.vlines(L_RL[Taidx], 0, TapYield[Taidx], colors = 'g', ls = ':', label = '__nolegend__')
    # # plt.vlines(L_RL[Xeidx], 0, XepYield[Xeidx], colors = 'k', ls = ':', label = '__nolegend__')
    # # plt.scatter(L_RL[Taidx], TaE[Taidx], color = 'g', marker = 's', facecolors = 'none', label = f'Cut Ta Max Yield: {TaE[Taidx]:.2f}')
    # # plt.scatter(L_RL[Xeidx], XeE[Xeidx], color = 'k', facecolors = 'none', marker = 'D', label = f'Cut LXe Max Yield: {XeE[Xeidx]:.2f}')
    # # plt.vlines([L_RL[rawTaidx], L_RL[Taidx]], [0, 0], [rawTapYield[rawTaidx], TapYield[Taidx]], colors = 'k', alpha = .65, ls = ':', label = '__nolegend__')

    # # plt.grid(True)
    # plt.xlabel(xlbl)
    # plt.ylabel('Mean Positron Energy [MeV]')
    # plt.xlim(0, 8.25)
    # plt.ylim(0, ELim)
    # plt.ylim(0, max(pYield) * 1.05)
    # plt.yticks(np.arange(0, ELim))
    # plt.legend(
        # loc = 'upper left'
        # , bbox_to_anchor = (0, .95)
    # )

    # plt.title('Mean $e^+$ Energy vs Target Width')

    # Yield Scatter Plot
    plt.subplot(1, 2, 1)
    plt.scatter(L_RL, rawTapYield, marker = '.', label = '__nolegend__')
    plt.scatter(L_RL, rawXepYield, marker = '.', label = '__nolegend__')
    plt.plot(L_RL[rawXeidx], rawXepYield[rawXeidx], 'r.', label = f'LXe Max Yield: {rawXepYield[rawXeidx]:.2f}')
    plt.plot(L_RL[rawTaidx], rawTapYield[rawTaidx], 'b.', label = f'Ta Max Yield: {rawTapYield[rawTaidx]:.2f}')
    plt.vlines(L_RL[rawTaidx], 0, rawTapYield[rawTaidx], colors = 'b', ls = ':', label = '__nolegend__')
    plt.vlines(L_RL[rawXeidx], 0, rawXepYield[rawXeidx], colors = 'r', ls = ':', label = '__nolegend__')

    plt.xlabel(xlbl)
    plt.ylabel('Positron Yield')
    plt.xlim(0, 8.25)
    plt.ylim(0, rawyieldLim)
    # plt.ylim(0, max(pYield) * 1.05)
    plt.yticks(np.arange(0, rawyieldLim, 5 * tick_spacing))
    plt.legend(
        loc = 'upper left'
        # , bbox_to_anchor = (0, .95)
    )
    plt.title('Raw Yield')
    
    plt.subplot(1, 2, 2)
    plt.scatter(L_RL, TapYield, marker = '.', label = '__nolegend__')
    plt.scatter(L_RL, XepYield, marker = '.', label = '__nolegend__')
    plt.plot(L_RL[Xeidx], XepYield[Xeidx], 'r.', label = f'LXe Max Yield: {XepYield[Xeidx]:.2f}')
    plt.plot(L_RL[Taidx], TapYield[Taidx], 'b.', label = f'Ta Max Yield: {TapYield[Taidx]:.2f}')
    plt.vlines(L_RL[Taidx], 0, TapYield[Taidx], colors = 'b', ls = ':', label = '__nolegend__')
    plt.vlines(L_RL[Xeidx], 0, XepYield[Xeidx], colors = 'r', ls = ':', label = '__nolegend__')

    plt.xlabel(xlbl)
    plt.ylabel('Positron Yield')
    plt.xlim(0, 8.25)
    plt.ylim(0, yieldLim)
    # plt.ylim(0, max(pYield) * 1.05)
    plt.yticks(np.arange(0, yieldLim, tick_spacing))
    plt.legend(
        loc = 'upper left'
        # , bbox_to_anchor = (0, .95)
    )
    plt.title('With Energy and Offset Constraints')

    plt.suptitle('$e^+$ Yield vs Target Width')
    plt.close()

    plt.subplot(1, 2, 1)
    plt.scatter(L_RL, TapYield, marker = '.', label = '__nolegend__')
    plt.scatter(L_RL, XepYield, marker = '.', label = '__nolegend__')
    plt.plot(L_RL[Xeidx], XepYield[Xeidx], 'r.', label = f'LXe Max Yield: ~{XepYield[Xeidx]:.2f} / $e^-$')
    plt.plot(L_RL[Taidx], TapYield[Taidx], 'b.', label = f'Ta Max Yield: ~{TapYield[Taidx]:.2f} / $e^-$')
    plt.vlines(L_RL[Taidx], 0, TapYield[Taidx], colors = 'b', ls = ':', label = '__nolegend__')
    plt.vlines(L_RL[Xeidx], 0, XepYield[Xeidx], colors = 'r', ls = ':', label = '__nolegend__')

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
    plt.ylabel('Normalized Positron Yield')
    plt.xlim(0, 8.25)
    plt.ylim(0, yieldLim)
    # plt.ylim(0, max(pYield) * 1.05)
    plt.yticks(np.arange(0, yieldLim, tick_spacing))
    plt.legend(
        loc = 'upper left'
        # , bbox_to_anchor = (0, .95)
    )
    # plt.title('$e^+$ Yield vs Target Width')

    plt.subplot(1, 2, 2)
    plt.scatter(L_RL, TaTW_RMS, marker = '.', facecolors = 'none', label = '__nolegend__')
    plt.scatter(L_RL, XeTW_RMS, marker = '.', facecolors = 'none', label = '__nolegend__')
    plt.plot(L_RL[Xeidx], XeTW_RMS[Xeidx], 'r.', label = f'Xe Traverse Width @ Max Yield: ~{XeTW_RMS[Xeidx]:.2f} mm')
    plt.plot(L_RL[Taidx], TaTW_RMS[Taidx], 'b.', label = f'Ta Traverse Width @ Max Yield: ~{TaTW_RMS[Taidx]:.2f} mm')
    # plt.vlines(L_RL[Xeidx], 0, XeTW_RMS[Xeidx], colors = 'r', ls = ':', label = '__nolegend__')
    # plt.vlines(L_RL[Taidx], 0, TaTW_RMS[Taidx], colors = 'b', ls = ':', label = '__nolegend__')
    plt.xlabel(xlbl)
    plt.ylabel('Mean Spot Size [mm]')
    plt.xlim(0, 8.25)
    plt.ylim(0, TWLim)
    # plt.yticks(np.arange(0, 2.5, .25))
    # plt.legend(loc = 'upper left'
    # , bbox_to_anchor = (0, .95)
    # )
    # plt.title('Mean $e^+$ Shower Spot Size vs Target Width')
    plt.close()
    # plt.scatter(L_RL, TaE_RMS, marker = '.', label = '__nolegend__')
    # plt.scatter(L_RL, XeE_RMS, marker = '.', label = '__nolegend__')
    # plt.plot(L_RL[idx], XeE_RMS[idx], 'r.', alpha = 1, label = f'Xe @ Max Yield: ~{TaE_RMS[idx]:.2f} MeV')
    # plt.plot(L_RL[idx], TaE_RMS[idx], 'b.', alpha = 1, label = f'Ta @ Max Yield: ~{XeE_RMS[idx]:.2f} MeV')
    # plt.vlines(L_RL[idx], 0, TaE_RMS[idx], colors = 'b', ls = ':', label = '__nolegend__')
    # plt.vlines(L_RL[idx], 0, XeE_RMS[idx], colors = 'r', ls = ':', label = '__nolegend__')
    # # plt.text(L_RL[idx], XeE_RMS[idx] / 10, f'~{L_RL[idx] * LTarget:.2f} cm', fontsize = 12)
    # # plt.hlines(E_RMS[idx], 0, L_RL[idx], colors = 'r', ls = ':', label = '__nolegend__')
    # plt.xlim(0, 8.25)
    # plt.ylim(0, ELim)
    # # plt.ylim(0, max(E_RMS) * 1.05)
    # plt.yticks(np.arange(0, max(TaE_RMS), 100))
    # plt.xlabel(xlbl)
    # plt.ylabel('RMS Positron Energy [MeV]')
    # plt.legend()
    # plt.title('$e^+$ Energy vs Target Width')

    # plt.suptitle(ttlStr, fontsize = 14)
    # plt.show()

    # Transverse Displacement and Transverse Emittance
    plt.subplot(1, 2, 1)
    # plt.scatter(L_RL, Taconvolved_trans, marker = '.', color = 'b', facecolors = 'none', label = '__nolegend__')
    # plt.scatter(L_RL, XeTW_RMS, marker = '.', color = 'r', facecolors = 'none', label = '__nolegend__')
    # plt.plot(L_RL[idx], XeTW_RMS[idx], 'r.', label = f'Xe Traverse Width @ Max Yield: ~{XeTW_RMS[idx]:.2f} mm')
    # plt.plot(L_RL[idx], TaTW_RMS[idx], 'b.', label = f'Ta Traverse Width @ Max Yield: ~{TaTW_RMS[idx]:.2f} mm')

    plt.scatter(L_RL, rawTaTW, color = 'b', marker = 'o', facecolors = 'none', label = 'Ta')
    plt.scatter(L_RL, rawXeTW, color = 'r', marker = 'v', facecolors = 'none', label = 'Xe')
    plt.scatter(L_RL, rawWReTW, color = 'g', marker = '^', facecolors = 'none', label = '$W_{75}Re_{25}$')
    # plt.vlines(L_RL[rawTaidx], 0, rawTapYield[rawTaidx], colors = 'b', ls = ':', label = '__nolegend__')
    # plt.vlines(L_RL[rawXeidx], 0, rawXepYield[rawXeidx], colors = 'r', ls = ':', label = '__nolegend__')
    # plt.scatter(L_RL[rawTaidx], Taconvolved_trans[rawTaidx], color = 'b', marker = 'o', facecolors = 'none', label = f'Ta @ Max Yield: {Taconvolved_trans[rawTaidx]:.2f} mm')
    # plt.scatter(L_RL[rawXeidx], Xeconvolved_trans[rawXeidx], color = 'r', marker = 'v', facecolors = 'none', label = f'LXe @ Max Yield: {Xeconvolved_trans[rawXeidx]:.2f} mm')

    plt.scatter(L_RL, TaTW, color = 'm', marker = 's', facecolors = 'none', label = 'Cut Ta')
    plt.scatter(L_RL, XeTW, color = 'k', facecolors = 'none', marker = 'D', label = 'Cut Xe')
    plt.scatter(L_RL, WReTW, color = 'tab:orange', facecolors = 'none', marker = 'X', label = 'Cut $W_{75}Re_{25}$')
    # plt.vlines(L_RL[Taidx], 0, TapYield[Taidx], colors = 'g', ls = ':', label = '__nolegend__')
    # plt.vlines(L_RL[Xeidx], 0, XepYield[Xeidx], colors = 'k', ls = ':', label = '__nolegend__')
    # plt.scatter(L_RL[Taidx], Tacts[0][Taidx], color = 'g', marker = 's', facecolors = 'none', label = f'Cut Ta @ Max Yield: {Tacts[0][Taidx]:.2f} mm')
    # plt.scatter(L_RL[Xeidx], Xects[0][Xeidx], color = 'k', facecolors = 'none', marker = 'D', label = f'Cut LXe @ Max Yield: {Xects[0][Xeidx]:.2f} mm')
    # plt.vlines(L_RL[idx], 0, XeTW_RMS[idx], colors = 'r', ls = ':', label = '__nolegend__')
    # plt.vlines(L_RL[idx], 0, TaTW_RMS[idx], colors = 'b', ls = ':', label = '__nolegend__')

    plt.xlabel(xlbl)
    plt.ylabel('Mean Spot Size [mm]')
    plt.xlim(0, 8.25)
    plt.ylim(0, TWLim)
    # plt.yticks(np.arange(0, 2.5, .25))
    plt.legend(loc = 'upper left'
    # , bbox_to_anchor = (0, .95)
    )
    # plt.title('Mean $e^+$ Shower Spot Size vs Target Width')
    
    plt.subplot(1, 2, 2)
    plt.scatter(L_RL, rawTaconv_emittance, color = 'b', marker = 'o', facecolors = 'none', label = '__nolegend__')
    plt.scatter(L_RL, rawXeconv_emittance, color = 'r', marker = 'v', facecolors = 'none', label = '__nolegend__')
    plt.scatter(L_RL, rawWReconv_emittance, color = 'g', marker = '^', facecolors = 'none', label = '__nolegend__')
    # plt.vlines(L_RL[rawTaidx], 0, rawTapYield[rawTaidx], colors = 'b', ls = ':', label = '__nolegend__')
    # plt.vlines(L_RL[rawXeidx], 0, rawXepYield[rawXeidx], colors = 'r', ls = ':', label = '__nolegend__')
    # plt.scatter(L_RL[rawTaidx], rawTaconv_emittance[rawTaidx], color = 'b', marker = 'o', facecolors = 'none', label = f'Ta @ Max Yield: {rawTaconv_emittance[rawTaidx]:.2f} mm\cdot rad')
    # plt.scatter(L_RL[rawXeidx], rawXeconv_emittance[rawXeidx], color = 'r', marker = 'v', facecolors = 'none', label = f'LXe @ Max Yield: {rawXeconv_emittance[rawXeidx]:.2f} mm\cdot rad')

    plt.scatter(L_RL, Taconv_emittance, color = 'm', marker = 's', facecolors = 'none', label = '__nolegend__')
    plt.scatter(L_RL, Xeconv_emittance, color = 'k', facecolors = 'none', marker = 'D', label = '__nolegend__')
    plt.scatter(L_RL, WReconv_emittance, color = 'tab:orange', facecolors = 'none', marker = 'X', label = '__nolegend__')
    # plt.vlines(L_RL[Taidx], 0, TapYield[Taidx], colors = 'g', ls = ':', label = '__nolegend__')
    # plt.vlines(L_RL[Xeidx], 0, XepYield[Xeidx], colors = 'k', ls = ':', label = '__nolegend__')
    # plt.scatter(L_RL[Taidx], Taconv_emittance[Taidx], color = 'g', marker = 's', facecolors = 'none', label = f'Cut Ta @ Max Yield: {Taconv_emittance[Taidx]:.2f} mm\cdot rad')
    # plt.scatter(L_RL[Xeidx], Xeconv_emittance[Xeidx], color = 'k', facecolors = 'none', marker = 'D', label = f'Cut LXe @ Max Yield: {Xeconv_emittance[Xeidx]:.2f} mm\cdot rad')
    # plt.vlines(L_RL[idx], 0, XeTW_RMS[idx], colors = 'r', ls = ':', label = '__nolegend__')
    # plt.vlines(L_RL[idx], 0, TaTW_RMS[idx], colors = 'b', ls = ':', label = '__nolegend__')

    plt.xlabel(xlbl)
    plt.ylabel('Mean Normalized Emittance [mm$\cdot$rad]')
    plt.xlim(0, 8.25)
    plt.ylim(0, EmitLim)
    # plt.yticks(np.arange(0, 2.5, .25))
    # plt.legend(loc = 'upper left'
    # , bbox_to_anchor = (0, .95)
    # )
    # plt.title('Mean Normalized $e^+$ Emittance vs Target Width')

    plt.close()

    plt.subplot(1, 2, 2)
    plt.scatter(L_RL, TaA, marker = '.', label = '__nolegend__')
    plt.scatter(L_RL, XeA, marker = '.', label = '__nolegend__')
    plt.plot(L_RL[idx], XeA[idx], 'r.', label = f'Xe Diffraction Angle @ Max Yield: ~{TaA[idx]:.2f} rad')
    plt.plot(L_RL[idx], TaA[idx], 'b.', label = f'Ta Diffraction Angle @ Max Yield: ~{XeA[idx]:.2f} rad')
    plt.vlines(L_RL[idx], 0, TaA[idx], colors = 'b', ls = ':', label = '__nolegend__')
    plt.vlines(L_RL[idx], 0, XeA[idx], colors = 'r', ls = ':', label = '__nolegend__')
    # plt.text(L_RL[idx], XeE_RMS[idx] / 10, f'~{L_RL[idx] * LTarget:.2f} cm', fontsize = 12)
    # plt.hlines(E_RMS[idx], 0, L_RL[idx], colors = 'r', ls = ':', label = '__nolegend__')
    plt.xlim(0, 8.25)
    plt.ylim(0, ALim)
    plt.xlabel(xlbl)
    plt.ylabel('Mean Angle [rad]')
    plt.legend()
    # plt.title('RMS $e^+$ Azimuthal Angle vs Target Width')

    # plt.suptitle(ttlStr, fontsize = 14)
    plt.close()

def plots(ts2 = '', spot = 'upper left', windows = False):

    _, _, _, pYield, EDep_RMS, WInEDep_RMS, WOutEDep_RMS, E_RMS, TW_RMS, A_RMS, idx, _, _, _, _, _, _, _, E, TW, A, _, ttlStr, xlbl, cm, bn, scale, emittance, TW = plotArgs(windows)

    if windows:
        # print(f'Target E-Dep @ Max Yield:  {EDep_RMS[idx] * E_drive :.2f} MeV')
        # print(f'Entrance Window E-Dep @ Max Yield:  {WInEDep_RMS[idx]:.2f} MeV')
        # print(f'Exit Window E-Dep @ Max Yield:  {WOutEDep_RMS[idx]:.2f} MeV')
        # plt.scatter(L_RL, EDep_RMS, label = '__nolegend__')
        plt.scatter(L_RL, WInEDep_RMS, label = '__nolegend__')
        plt.scatter(L_RL, WOutEDep_RMS, label = '__nolegend__')
        # plt.plot(L_RL[idx], EDep_RMS[idx], 'bo', label = f'Target @ Max Yield: ~{EDep_RMS[idx]:.2f}')
        plt.plot(L_RL[idx], WInEDep_RMS[idx], 'bo', label = f'Entrance Window @ Max Yield: ~{WInEDep_RMS[idx]:.2f} MeV')
        plt.plot(L_RL[idx], WOutEDep_RMS[idx], 'ro', label = f'Exit Window @ Max Yield: ~{WOutEDep_RMS[idx]:.2f} MeV')
        # plt.vlines(L_RL[idx], 0, EDep_RMS[idx], colors = 'b', ls = ':', label = '__nolegend__')
        plt.vlines(L_RL[idx], 0, WOutEDep_RMS[idx], colors = 'r', ls = ':', label = '__nolegend__')
        plt.vlines(L_RL[idx], 0, WInEDep_RMS[idx], colors = 'b', ls = ':', label = '__nolegend__')
        plt.xlabel(xlbl)
        plt.ylabel(f'RMS Energy Deposition [MeV / incident $e^-$ @ {E_drive * 1e-3} GeV]')
        # plt.ylabel('RMS Energy Deposition / incident $e^-$ [MeV]')
        # plt.ylim(0, 5.75e-1)
        plt.legend(loc = 'upper left')
        plt.suptitle(ts2, fontsize = 14)
        
        plt.title('Energy Deposition in Beryllium Windows vs Width of Target')
        plt.close()

    # ax = plt.figure().add_subplot(projection = '3d')
    # ax.plot(L_RL, pYield, WOutEDep_RMS)
    # ax.set_xlabel('Radiation Lengths')
    # ax.set_ylabel('$e^+$ Yield')
    # ax.set_zlabel('Window E$_{dep}$ [MeV / incident $e^-$ @ %i GeV]' % (E_drive * 1e-3))
    # plt.title(f'')
    # plt.show()

    plt.scatter(L_RL, TW_RMS, marker = 'o', color = 'k', facecolors = 'none')
    plt.xlabel(xlbl)
    plt.ylabel('Mean Spot Size [mm]')
    plt.close()

    plt.scatter(L_RL, emittance, label = '__nolegend__')
    plt.plot(L_RL[idx], emittance[idx], 'ro', label = f'Emittance @ Max Yield: ~{emittance[idx]:.2f} mm$\cdot$rad')
    plt.vlines(L_RL[idx], 0, emittance[idx], colors = 'r', ls = ':', label = '__nolegend__')
    plt.xlabel(xlbl)
    plt.ylabel('Normalized RMS Emittance [mm$\cdot$rad]')
    plt.legend(
        loc = spot
        )
    plt.title(f'{component}-Emittance vs Target Width \n Energy Cutoff: {E_filter:.2f} MeV')
    # print(f'{component}-Emittance : {emittance[idx]} mm rad')
    plt.suptitle(ts2, fontsize = 14)
    plt.close()

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

    # print(f'EDep @ Max Yield:  {EDep_RMS[idx] * E_drive :.2f} MeV')
    
    plt.scatter(L_RL, EDep_RMS, label = '__nolegend__')
    plt.plot(L_RL[idx], EDep_RMS[idx], 'ro', label = f'Fractional Energy Deposition @ Max Yield: ~{EDep_RMS[idx]:.2f}')
    plt.vlines(L_RL[idx], 0, EDep_RMS[idx], colors = 'r', ls = ':', label = '__nolegend__')
    plt.xlabel(xlbl)
    plt.ylabel(f'RMS Fractional Energy Deposition [ / incident $e^-$ @ {E_drive * 1e-3} GeV]')
    # plt.ylabel('RMS Energy Deposition / incident $e^-$ [MeV]')
    # plt.ylim(0, 5.75e-1)
    plt.legend(loc = 'upper left')
    # plt.suptitle(ts2, fontsize = 14)
    
    plt.title('Energy Deposition vs Target Width')
    plt.close()
    
    # plt.scatter(L_RL, R_RMS)
    # plt.xlim(0, 4.25)
    # plt.ylabel('RMS Rotational Angle [rad]')
    # plt.xlabel(xlbl)
    # plt.show()

    # if mode == 'Ta':
    #     bn = (200, 350)
    # plt.subplot(1, 2, 1)
    # plt.hist2d(E, TW, bins = bn, weights = np.full(np.shape(TW), scale / iterations), norm = mpl.colors.LogNorm(), cmap = cm)
    # plt.ylim(0, 10)
    # plt.xlim(0, 7e3)
    # plt.title('$e^+$ Shower Size vs $e^+$ Energy')
    # plt.xlabel('Energy [MeV]')
    # plt.ylabel('Spot Size [mm]')

    # if mode == 'Ta':
    #     bn = (200, 4000)
    # plt.subplot(1, 2, 2)
    # h = plt.hist2d(E, A,  bins = bn, weights = np.full(np.shape(TW), scale / iterations), norm = mpl.colors.LogNorm(), cmap = cm)
    # plt.colorbar(h[3], label = 'Counts / 100 incident $e^-$')
    # plt.ylim(0, .15)
    # plt.xlim(0, 7e3)
    # plt.title('$e^+$ Diffraction Angle vs $e^+$ Energy')
    # plt.xlabel('Energy [MeV]')
    # plt.ylabel('Angle [rad]')

    # plt.suptitle(ttlStr, fontsize = 14)
    # plt.close()

    plt.scatter(L_RL, pYield, label = '__nolegend__')
    plt.plot(L_RL[idx], pYield[idx], 'ro', label = f'Max Positron Yield: ~{pYield[idx]:.2f} / $e^-$')
    plt.vlines(L_RL[idx], 0, pYield[idx], colors = 'r', ls = ':', label = '__nolegend__')

    # plt.text(L_RL[idx], pYield[idx] / 100, f'~{L_RL[idx] * LTarget:.2f} cm', fontsize = 12)
    # plt.hlines(pYield[idx], 0, L_RL[idx], colors = 'r', ls = ':', label = '__nolegend__')
    plt.xlabel(xlbl)
    plt.ylabel(f'Normalized Positron Yield') # [per Incident $e^-$ @ {E_drive * 1e-3} GeV]')
    plt.xlim(0, 8.25)
    plt.ylim(0, max(pYield) * 1.15)
    # plt.ylim(0, max(pYield) * 1.05)
    plt.yticks(np.arange(0, max(pYield) * 1.15, tick_spacing))
    # plt.legend(
    #     loc = 'upper left'
    #     # , bbox_to_anchor = (0, .95)
    # )
    plt.title('$e^+$ Yield vs Target Width')
    plt.show()

    plt.subplot(1, 2, 1)
    plt.scatter(L_RL, pYield, label = '__nolegend__')
    plt.plot(L_RL[idx], pYield[idx], 'ro', label = f'Max Positron Yield: ~{pYield[idx]:.2f} / $e^-$')
    plt.vlines(L_RL[idx], 0, pYield[idx], colors = 'r', ls = ':', label = '__nolegend__')

    # plt.text(L_RL[idx], pYield[idx] / 100, f'~{L_RL[idx] * LTarget:.2f} cm', fontsize = 12)
    # plt.hlines(pYield[idx], 0, L_RL[idx], colors = 'r', ls = ':', label = '__nolegend__')
    plt.xlabel(xlbl)
    plt.ylabel(f'Normalized Positron Yield') # [per Incident $e^-$ @ {E_drive * 1e-3} GeV]')
    plt.xlim(0, 8.25)
    plt.ylim(0, max(pYield) * 1.15)
    # plt.ylim(0, max(pYield) * 1.05)
    plt.yticks(np.arange(0, max(pYield) * 1.15, tick_spacing))
    # plt.legend(
    #     loc = 'upper left'
    #     # , bbox_to_anchor = (0, .95)
    # )
    plt.title('$e^+$ Yield vs Target Width')

    plt.subplot(1, 2, 2)
    plt.scatter(L_RL, WInEDep_RMS, label = '__nolegend__')
    plt.scatter(L_RL, WOutEDep_RMS, label = '__nolegend__')
    # plt.plot(L_RL[idx], EDep_RMS[idx], 'bo', label = f'Target @ Max Yield: ~{EDep_RMS[idx]:.2f}')
    plt.plot(L_RL[idx], WInEDep_RMS[idx], 'bo', label = f'Entrance Window @ Max Yield: ~{WInEDep_RMS[idx]:.2f} MeV')
    plt.plot(L_RL[idx], WOutEDep_RMS[idx], 'ro', label = f'Exit Window @ Max Yield: ~{WOutEDep_RMS[idx]:.2f} MeV')
    # plt.vlines(L_RL[idx], 0, EDep_RMS[idx], colors = 'b', ls = ':', label = '__nolegend__')
    plt.vlines(L_RL[idx], 0, WOutEDep_RMS[idx], colors = 'r', ls = ':', label = '__nolegend__')
    plt.vlines(L_RL[idx], 0, WInEDep_RMS[idx], colors = 'b', ls = ':', label = '__nolegend__')
    plt.xlabel(xlbl)
    plt.ylabel(f'RMS Energy Deposition [MeV / incident $e^-$ @ {E_drive * 1e-3} GeV]')
    # plt.ylabel('RMS Energy Deposition / incident $e^-$ [MeV]')
    # plt.ylim(0, 5.75e-1)
    # plt.legend(loc = 'upper left')
    plt.suptitle(ts2, fontsize = 14)
    
    plt.title('Energy Deposition in Beryllium Windows vs Width of Target')
    plt.close()
    # plt.scatter(L_RL, E_RMS)
    # plt.plot(L_RL[idx], E_RMS[idx], 'ro', label = f'RMS Energy @ Max Yield: ~{E_RMS[idx]:.2f} MeV')
    # plt.vlines(L_RL[idx], 0, E_RMS[idx], colors = 'r', ls = ':', label = '__nolegend__')
    # plt.text(L_RL[idx], E_RMS[idx] / 10, f'~{L_RL[idx] * LTarget:.2f} cm', fontsize = 12)
    # # plt.hlines(E_RMS[idx], 0, L_RL[idx], colors = 'r', ls = ':', label = '__nolegend__')
    # plt.xlim(0, 8.25)
    # plt.ylim(0, max(E_RMS) * 1.05)
    # # plt.ylim(0, max(E_RMS) * 1.05)
    # plt.yticks(np.arange(0, max(E_RMS), 100))
    # plt.xlabel(xlbl)
    # plt.ylabel('RMS Positron Energy [MeV]')
    # plt.legend()
    # plt.title('RMS $e^+$ Energy vs Target Width')

    # plt.suptitle(f'Energy Cutoff: {E_filter:.2f} MeV', fontsize = 14)
    # plt.show()

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
    plt.xlim(0, 8.25)
    plt.ylim(0, max(TW_RMS) * 1.05)
    # plt.yticks(np.arange(0, 2.5, .25))
    plt.legend(
        # loc = 'upper left'
    # , bbox_to_anchor = (0, .95)
    )
    plt.title('RMS $e^+$ Shower Spot Size vs Target Width')

    plt.subplot(1, 2, 2)
    plt.scatter(L_RL, A_RMS)
    plt.plot(L_RL[idx], A_RMS[idx], 'ro', label = f'Diffraction Angle @ Max Yield: ~{A_RMS[idx]:.2f} rad')
    plt.vlines(L_RL[idx], 0, A_RMS[idx], colors = 'r', ls = ':', label = '__nolegend__')
    # plt.hlines(E_RMS[idx], 0, L_RL[idx], colors = 'r', ls = ':', label = '__nolegend__')
    plt.xlim(0, 8.25)
    plt.ylim(0, max(A_RMS) * 1.05)
    # plt.yticks(np.arange(0, max(A_RMS), 100))
    plt.xlabel(xlbl)
    plt.ylabel('RMS Angle [rad]')
    plt.legend()
    plt.title('RMS $e^+$ Angle vs Target Width')

    plt.suptitle(f'Energy Cutoff: {E_filter:.2f} MeV', fontsize = 14)
    plt.close()

if __name__ == '__main__':
    if mode == 'Xe':
        LTarget = LXe
        
        edep0 = pd.read_csv('10GeVData/XeDep0.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep1 = pd.read_csv('10GeVData/XeDep1.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep2 = pd.read_csv('10GeVData/XeDep2.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep3 = pd.read_csv('10GeVData/XeDep3.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep4 = pd.read_csv('10GeVData/XeDep4.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep5 = pd.read_csv('10GeVData/XeDep5.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep6 = pd.read_csv('10GeVData/XeDep6.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep7 = pd.read_csv('10GeVData/XeDep7.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep8 = pd.read_csv('10GeVData/XeDep8.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep9 = pd.read_csv('10GeVData/XeDep9.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep10 = pd.read_csv('10GeVData/XeDep10.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep11 = pd.read_csv('10GeVData/XeDep11.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep12 = pd.read_csv('10GeVData/XeDep12.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep13 = pd.read_csv('10GeVData/XeDep13.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep14 = pd.read_csv('10GeVData/XeDep14.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep15 = pd.read_csv('10GeVData/XeDep15.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()

        df0 = pd.read_csv('10GeVData/Xe0.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df1 = pd.read_csv('10GeVData/Xe1.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df2 = pd.read_csv('10GeVData/Xe2.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df3 = pd.read_csv('10GeVData/Xe3.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df4 = pd.read_csv('10GeVData/Xe4.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df5 = pd.read_csv('10GeVData/Xe5.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df6 = pd.read_csv('10GeVData/Xe6.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df7 = pd.read_csv('10GeVData/Xe7.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df8 = pd.read_csv('10GeVData/Xe8.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df9 = pd.read_csv('10GeVData/Xe9.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df10 = pd.read_csv('10GeVData/Xe10.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df11 = pd.read_csv('10GeVData/Xe11.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df12 = pd.read_csv('10GeVData/Xe12.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df13 = pd.read_csv('10GeVData/Xe13.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df14 = pd.read_csv('10GeVData/Xe14.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df15 = pd.read_csv('10GeVData/Xe15.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        
        p0 = pd.read_csv('10GeVData/Xe0.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p1 = pd.read_csv('10GeVData/Xe1.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p2 = pd.read_csv('10GeVData/Xe2.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p3 = pd.read_csv('10GeVData/Xe3.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p4 = pd.read_csv('10GeVData/Xe4.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p5 = pd.read_csv('10GeVData/Xe5.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p6 = pd.read_csv('10GeVData/Xe6.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p7 = pd.read_csv('10GeVData/Xe7.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p8 = pd.read_csv('10GeVData/Xe8.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p9 = pd.read_csv('10GeVData/Xe9.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p10 = pd.read_csv('10GeVData/Xe10.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p11 = pd.read_csv('10GeVData/Xe11.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p12 = pd.read_csv('10GeVData/Xe12.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p13 = pd.read_csv('10GeVData/Xe13.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p14 = pd.read_csv('10GeVData/Xe14.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p15 = pd.read_csv('10GeVData/Xe15.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]

        if plot:
        #     if cutoff:
        #         # spot = 'lower right'
        #         ts2 = f'Spot Size Cutoff : {w} mm'
        #         # ts2 = ''
                
        #         p0 = p0[df0[:, 1] <= w]
        #         p1 = p1[df1[:, 1] <= w]
        #         p2 = p2[df2[:, 1] <= w]
        #         p3 = p3[df3[:, 1] <= w]
        #         p4 = p4[df4[:, 1] <= w]
        #         p5 = p5[df5[:, 1] <= w]
        #         p6 = p6[df6[:, 1] <= w]
        #         p7 = p7[df7[:, 1] <= w]
        #         p8 = p8[df8[:, 1] <= w]
        #         p9 = p9[df9[:, 1] <= w]
        #         p10 = p10[df10[:, 1] <= w]
        #         p11 = p11[df11[:, 1] <= w]
        #         p12 = p12[df12[:, 1] <= w]
        #         p13 = p13[df13[:, 1] <= w]
        #         p14 = p14[df14[:, 1] <= w]
        #         p15 = p15[df15[:, 1] <= w]
                
        #         df0 = df0[df0[:, 1] <= w]
        #         df1 = df1[df1[:, 1] <= w]
        #         df2 = df2[df2[:, 1] <= w]
        #         df3 = df3[df3[:, 1] <= w]
        #         df4 = df4[df4[:, 1] <= w]
        #         df5 = df5[df5[:, 1] <= w]
        #         df6 = df6[df6[:, 1] <= w]
        #         df7 = df7[df7[:, 1] <= w]
        #         df8 = df8[df8[:, 1] <= w]
        #         df9 = df9[df9[:, 1] <= w]
        #         df10 = df10[df10[:, 1] <= w]
        #         df11 = df11[df11[:, 1] <= w]
        #         df12 = df12[df12[:, 1] <= w]
        #         df13 = df13[df13[:, 1] <= w]
        #         df14 = df14[df14[:, 1] <= w]
        #         df15 = df15[df15[:, 1] <= w]
                
        #         plots(ts2, windows = wBool)
        #     else:
                plots(windows = wBool)
        else:
            plotArgs(windows = wBool)
    elif mode == 'Ta':
        LTarget = LTa
        ttlStr = '$L_{RL}(Ta) = 0.4094 cm \\quad \\vert \\quad \\frac{d}{L_{LR}(Ta)} \\approx 2.75$ \n Energy Cutoff: 100 MeV'
        pltStr = 'Radiation Lengths [$L_{RL}(Ta) = 0.4094$ cm]'
        
        edep0 = pd.read_csv('10GeVData/TaDep0.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep1 = pd.read_csv('10GeVData/TaDep1.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep2 = pd.read_csv('10GeVData/TaDep2.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep3 = pd.read_csv('10GeVData/TaDep3.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep4 = pd.read_csv('10GeVData/TaDep4.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep5 = pd.read_csv('10GeVData/TaDep5.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep6 = pd.read_csv('10GeVData/TaDep6.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep7 = pd.read_csv('10GeVData/TaDep7.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep8 = pd.read_csv('10GeVData/TaDep8.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep9 = pd.read_csv('10GeVData/TaDep9.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep10 = pd.read_csv('10GeVData/TaDep10.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep11 = pd.read_csv('10GeVData/TaDep11.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep12 = pd.read_csv('10GeVData/TaDep12.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep13 = pd.read_csv('10GeVData/TaDep13.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep14 = pd.read_csv('10GeVData/TaDep14.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep15 = pd.read_csv('10GeVData/TaDep15.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()

        df0 = pd.read_csv('10GeVData/Ta0.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df1 = pd.read_csv('10GeVData/Ta1.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df2 = pd.read_csv('10GeVData/Ta2.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df3 = pd.read_csv('10GeVData/Ta3.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df4 = pd.read_csv('10GeVData/Ta4.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df5 = pd.read_csv('10GeVData/Ta5.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df6 = pd.read_csv('10GeVData/Ta6.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df7 = pd.read_csv('10GeVData/Ta7.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df8 = pd.read_csv('10GeVData/Ta8.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df9 = pd.read_csv('10GeVData/Ta9.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df10 = pd.read_csv('10GeVData/Ta10.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df11 = pd.read_csv('10GeVData/Ta11.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df12 = pd.read_csv('10GeVData/Ta12.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df13 = pd.read_csv('10GeVData/Ta13.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df14 = pd.read_csv('10GeVData/Ta14.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df15 = pd.read_csv('10GeVData/Ta15.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        
        p0 = pd.read_csv('10GeVData/Ta0.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p1 = pd.read_csv('10GeVData/Ta1.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p2 = pd.read_csv('10GeVData/Ta2.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p3 = pd.read_csv('10GeVData/Ta3.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p4 = pd.read_csv('10GeVData/Ta4.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p5 = pd.read_csv('10GeVData/Ta5.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p6 = pd.read_csv('10GeVData/Ta6.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p7 = pd.read_csv('10GeVData/Ta7.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p8 = pd.read_csv('10GeVData/Ta8.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p9 = pd.read_csv('10GeVData/Ta9.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p10 = pd.read_csv('10GeVData/Ta10.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p11 = pd.read_csv('10GeVData/Ta11.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p12 = pd.read_csv('10GeVData/Ta12.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p13 = pd.read_csv('10GeVData/Ta13.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p14 = pd.read_csv('10GeVData/Ta14.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p15 = pd.read_csv('10GeVData/Ta15.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        
        if plot:
            plots()
        else:
            plotArgs()
    elif mode == 'WRe':
        LTarget = WRe
        # ttlStr = '$L_{RL}(Ta) = 0.4094 cm \\quad \\vert \\quad \\frac{d}{L_{LR}(Ta)} \\approx 2.75$ \n Energy Cutoff: 100 MeV'
        pltStr = 'Radiation Lengths [$L_{RL}(W_{75}Re_{25}) = 0.343$ cm]'

        edep0 = pd.read_csv('10GeVData/WreDep0.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep1 = pd.read_csv('10GeVData/WreDep1.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep2 = pd.read_csv('10GeVData/WreDep2.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep3 = pd.read_csv('10GeVData/WreDep3.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep4 = pd.read_csv('10GeVData/WreDep4.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep5 = pd.read_csv('10GeVData/WreDep5.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep6 = pd.read_csv('10GeVData/WreDep6.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep7 = pd.read_csv('10GeVData/WreDep7.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep8 = pd.read_csv('10GeVData/WreDep8.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep9 = pd.read_csv('10GeVData/WreDep9.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep10 = pd.read_csv('10GeVData/WReDep10.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep11 = pd.read_csv('10GeVData/WReDep11.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep12 = pd.read_csv('10GeVData/WReDep12.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep13 = pd.read_csv('10GeVData/WReDep13.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep14 = pd.read_csv('10GeVData/WReDep14.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep15 = pd.read_csv('10GeVData/WReDep15.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()

        df0 = pd.read_csv('10GeVData/WRe0.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df1 = pd.read_csv('10GeVData/WRe1.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df2 = pd.read_csv('10GeVData/WRe2.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df3 = pd.read_csv('10GeVData/WRe3.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df4 = pd.read_csv('10GeVData/WRe4.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df5 = pd.read_csv('10GeVData/WRe5.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df6 = pd.read_csv('10GeVData/WRe6.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df7 = pd.read_csv('10GeVData/WRe7.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df8 = pd.read_csv('10GeVData/WRe8.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df9 = pd.read_csv('10GeVData/WRe9.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df10 = pd.read_csv('10GeVData/WRe10.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df11 = pd.read_csv('10GeVData/WRe11.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df12 = pd.read_csv('10GeVData/WRe12.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df13 = pd.read_csv('10GeVData/WRe13.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df14 = pd.read_csv('10GeVData/WRe14.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df15 = pd.read_csv('10GeVData/WRe15.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        
        p0 = pd.read_csv('10GeVData/WRe0.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p1 = pd.read_csv('10GeVData/WRe1.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p2 = pd.read_csv('10GeVData/WRe2.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p3 = pd.read_csv('10GeVData/WRe3.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p4 = pd.read_csv('10GeVData/WRe4.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p5 = pd.read_csv('10GeVData/WRe5.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p6 = pd.read_csv('10GeVData/WRe6.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p7 = pd.read_csv('10GeVData/WRe7.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p8 = pd.read_csv('10GeVData/WRe8.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p9 = pd.read_csv('10GeVData/WRe9.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p10 = pd.read_csv('10GeVData/WRe10.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p11 = pd.read_csv('10GeVData/WRe11.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p12 = pd.read_csv('10GeVData/WRe12.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p13 = pd.read_csv('10GeVData/WRe13.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p14 = pd.read_csv('10GeVData/WRe14.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p15 = pd.read_csv('10GeVData/WRe15.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]

        # edep0 = pd.read_csv('data/WReDep0.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        # edep1 = pd.read_csv('data/WReDep1.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        # edep2 = pd.read_csv('data/WReDep2.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        # edep3 = pd.read_csv('data/WReDep3.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        # edep4 = pd.read_csv('data/WReDep4.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        # edep5 = pd.read_csv('data/WReDep5.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        # edep6 = pd.read_csv('data/WReDep6.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        # edep7 = pd.read_csv('data/WReDep7.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        # edep8 = pd.read_csv('data/WReDep8.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        # edep9 = pd.read_csv('data/WReDep9.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        # edep10 = pd.read_csv('data/WReDep10.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        # edep11 = pd.read_csv('data/WReDep11.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        # edep12 = pd.read_csv('data/WReDep12.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        # edep13 = pd.read_csv('data/WReDep13.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        # edep14 = pd.read_csv('data/WReDep14.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        # edep15 = pd.read_csv('data/WReDep15.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()

        # df0 = pd.read_csv('data/WRe0.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        # df1 = pd.read_csv('data/WRe1.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        # df2 = pd.read_csv('data/WRe2.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        # df3 = pd.read_csv('data/WRe3.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        # df4 = pd.read_csv('data/WRe4.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        # df5 = pd.read_csv('data/WRe5.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        # df6 = pd.read_csv('data/WRe6.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        # df7 = pd.read_csv('data/WRe7.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        # df8 = pd.read_csv('data/WRe8.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        # df9 = pd.read_csv('data/WRe9.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        # df10 = pd.read_csv('data/WRe10.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        # df11 = pd.read_csv('data/WRe11.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        # df12 = pd.read_csv('data/WRe12.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        # df13 = pd.read_csv('data/WRe13.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        # df14 = pd.read_csv('data/WRe14.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        # df15 = pd.read_csv('data/WRe15.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        
        # p0 = pd.read_csv('data/WRe0.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        # p1 = pd.read_csv('data/WRe1.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        # p2 = pd.read_csv('data/WRe2.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        # p3 = pd.read_csv('data/WRe3.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        # p4 = pd.read_csv('data/WRe4.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        # p5 = pd.read_csv('data/WRe5.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        # p6 = pd.read_csv('data/WRe6.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        # p7 = pd.read_csv('data/WRe7.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        # p8 = pd.read_csv('data/WRe8.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        # p9 = pd.read_csv('data/WRe9.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        # p10 = pd.read_csv('data/WRe10.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        # p11 = pd.read_csv('data/WRe11.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        # p12 = pd.read_csv('data/WRe12.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        # p13 = pd.read_csv('data/WRe13.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        # p14 = pd.read_csv('data/WRe14.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        # p15 = pd.read_csv('data/WRe15.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        
        if plot:
            plots()
        else:
            plotArgs()
    elif mode == 'comp':
        if disp:
            # ttlStr = '$L_{RL}(Ta) = 0.4094 cm \\quad \\vert \\quad \\frac{d}{L_{LR}(Ta)} \\approx 2.75$ \n Energy Cutoff: 100 MeV'
            ttlStr = f'Energy Cutoff: {E_filter:.2f} MeV | Transverse Spread Cutoff: {trans_filter:.2f} mm'
        else:
            ttlStr = ''

        xedep0 = pd.read_csv('10GeVData/XeDep0.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        xedep1 = pd.read_csv('10GeVData/XeDep1.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        xedep2 = pd.read_csv('10GeVData/XeDep2.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        xedep3 = pd.read_csv('10GeVData/XeDep3.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        xedep4 = pd.read_csv('10GeVData/XeDep4.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        xedep5 = pd.read_csv('10GeVData/XeDep5.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        xedep6 = pd.read_csv('10GeVData/XeDep6.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        xedep7 = pd.read_csv('10GeVData/XeDep7.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        xedep8 = pd.read_csv('10GeVData/XeDep8.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        xedep9 = pd.read_csv('10GeVData/XeDep9.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        xedep10 = pd.read_csv('10GeVData/XeDep10.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        xedep11 = pd.read_csv('10GeVData/XeDep11.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        xedep12 = pd.read_csv('10GeVData/XeDep12.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        xedep13 = pd.read_csv('10GeVData/XeDep13.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        xedep14 = pd.read_csv('10GeVData/XeDep14.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        xedep15 = pd.read_csv('10GeVData/XeDep15.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        
        tadep0 = pd.read_csv('10GeVData/TaDep0.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        tadep1 = pd.read_csv('10GeVData/TaDep1.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        tadep2 = pd.read_csv('10GeVData/TaDep2.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        tadep3 = pd.read_csv('10GeVData/TaDep3.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        tadep4 = pd.read_csv('10GeVData/TaDep4.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        tadep5 = pd.read_csv('10GeVData/TaDep5.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        tadep6 = pd.read_csv('10GeVData/TaDep6.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        tadep7 = pd.read_csv('10GeVData/TaDep7.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        tadep8 = pd.read_csv('10GeVData/TaDep8.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        tadep9 = pd.read_csv('10GeVData/TaDep9.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        # tadep10 = np.zeros((1, 3))
        tadep10 = pd.read_csv('10GeVData/TaDep10.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        tadep11 = pd.read_csv('10GeVData/TaDep11.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        tadep12 = pd.read_csv('10GeVData/TaDep12.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        tadep13 = pd.read_csv('10GeVData/TaDep13.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        tadep14 = pd.read_csv('10GeVData/TaDep14.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        # tadep15 = np.zeros((1, 3))
        tadep15 = pd.read_csv('10GeVData/TaDep15.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()

        wreedep0 = pd.read_csv('10GeVData/WreDep0.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        wreedep1 = pd.read_csv('10GeVData/WreDep1.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        wreedep2 = pd.read_csv('10GeVData/WreDep2.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        wreedep3 = pd.read_csv('10GeVData/WreDep3.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        wreedep4 = pd.read_csv('10GeVData/WreDep4.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        wreedep5 = pd.read_csv('10GeVData/WreDep5.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        wreedep6 = pd.read_csv('10GeVData/WreDep6.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        wreedep7 = pd.read_csv('10GeVData/WreDep7.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        wreedep8 = pd.read_csv('10GeVData/WreDep8.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        wreedep9 = pd.read_csv('10GeVData/WreDep9.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        wreedep10 = pd.read_csv('10GeVData/WReDep10.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        wreedep11 = pd.read_csv('10GeVData/WReDep11.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        wreedep12 = pd.read_csv('10GeVData/WReDep12.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        wreedep13 = pd.read_csv('10GeVData/WReDep13.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        wreedep14 = pd.read_csv('10GeVData/WReDep14.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        wreedep15 = pd.read_csv('10GeVData/WReDep15.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()


        # tadep0 = pd.read_csv('10GeVData/TaDep0.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
        # tadep1 = pd.read_csv('10GeVData/TaDep1.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
        # tadep2 = pd.read_csv('10GeVData/TaDep2.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
        # tadep3 = pd.read_csv('10GeVData/TaDep3.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
        # tadep4 = pd.read_csv('10GeVData/TaDep4.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
        # tadep5 = pd.read_csv('10GeVData/TaDep5.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
        # tadep6 = pd.read_csv('10GeVData/TaDep6.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
        # tadep7 = pd.read_csv('10GeVData/TaDep7.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
        # tadep8 = pd.read_csv('10GeVData/TaDep8.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
        # tadep9 = pd.read_csv('10GeVData/TaDep9.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
        # tadep10 = pd.read_csv('10GeVData/TaDep10.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
        # tadep11 = pd.read_csv('10GeVData/TaDep11.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
        # tadep12 = pd.read_csv('10GeVData/TaDep12.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
        # tadep13 = pd.read_csv('10GeVData/TaDep13.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
        # tadep14 = pd.read_csv('10GeVData/TaDep14.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()
        # tadep15 = pd.read_csv('10GeVData/TaDep15.csv', header = 4, names = ['Energy Deposition [MeV]']).to_numpy()

        Ta0 = pd.read_csv('10GeVData/Ta0.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        Ta1 = pd.read_csv('10GeVData/Ta1.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        Ta2 = pd.read_csv('10GeVData/Ta2.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        Ta3 = pd.read_csv('10GeVData/Ta3.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        Ta4 = pd.read_csv('10GeVData/Ta4.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        Ta5 = pd.read_csv('10GeVData/Ta5.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        Ta6 = pd.read_csv('10GeVData/Ta6.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        Ta7 = pd.read_csv('10GeVData/Ta7.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        Ta8 = pd.read_csv('10GeVData/Ta8.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        Ta9 = pd.read_csv('10GeVData/Ta9.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        Ta10 = pd.read_csv('10GeVData/Ta10.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        Ta11 = pd.read_csv('10GeVData/Ta11.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        Ta12 = pd.read_csv('10GeVData/Ta12.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        Ta13 = pd.read_csv('10GeVData/Ta13.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        Ta14 = pd.read_csv('10GeVData/Ta14.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        Ta15 = pd.read_csv('10GeVData/Ta15.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]

        Xe0 = pd.read_csv('10GeVData/Xe0.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        Xe1 = pd.read_csv('10GeVData/Xe1.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        Xe2 = pd.read_csv('10GeVData/Xe2.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        Xe3 = pd.read_csv('10GeVData/Xe3.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        Xe4 = pd.read_csv('10GeVData/Xe4.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        Xe5 = pd.read_csv('10GeVData/Xe5.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        Xe6 = pd.read_csv('10GeVData/Xe6.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        Xe7 = pd.read_csv('10GeVData/Xe7.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        Xe8 = pd.read_csv('10GeVData/Xe8.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        Xe9 = pd.read_csv('10GeVData/Xe9.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        Xe10 = pd.read_csv('10GeVData/Xe10.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        Xe11 = pd.read_csv('10GeVData/Xe11.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        Xe12 = pd.read_csv('10GeVData/Xe12.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        Xe13 = pd.read_csv('10GeVData/Xe13.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        Xe14 = pd.read_csv('10GeVData/Xe14.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        Xe15 = pd.read_csv('10GeVData/Xe15.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        
        WRe0 = pd.read_csv('10GeVData/WRe0.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        WRe1 = pd.read_csv('10GeVData/WRe1.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        WRe2 = pd.read_csv('10GeVData/WRe2.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        WRe3 = pd.read_csv('10GeVData/WRe3.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        WRe4 = pd.read_csv('10GeVData/WRe4.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        WRe5 = pd.read_csv('10GeVData/WRe5.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        WRe6 = pd.read_csv('10GeVData/WRe6.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        WRe7 = pd.read_csv('10GeVData/WRe7.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        WRe8 = pd.read_csv('10GeVData/WRe8.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        WRe9 = pd.read_csv('10GeVData/WRe9.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        WRe10 = pd.read_csv('10GeVData/WRe10.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        WRe11 = pd.read_csv('10GeVData/WRe11.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        WRe12 = pd.read_csv('10GeVData/WRe12.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        WRe13 = pd.read_csv('10GeVData/WRe13.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        WRe14 = pd.read_csv('10GeVData/WRe14.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        WRe15 = pd.read_csv('10GeVData/WRe15.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]


        Tap0 = pd.read_csv('10GeVData/Ta0.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        Tap1 = pd.read_csv('10GeVData/Ta1.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        Tap2 = pd.read_csv('10GeVData/Ta2.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        Tap3 = pd.read_csv('10GeVData/Ta3.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        Tap4 = pd.read_csv('10GeVData/Ta4.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        Tap5 = pd.read_csv('10GeVData/Ta5.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        Tap6 = pd.read_csv('10GeVData/Ta6.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        Tap7 = pd.read_csv('10GeVData/Ta7.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        Tap8 = pd.read_csv('10GeVData/Ta8.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        Tap9 = pd.read_csv('10GeVData/Ta9.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        Tap10 = pd.read_csv('10GeVData/Ta10.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        Tap11 = pd.read_csv('10GeVData/Ta11.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        Tap12 = pd.read_csv('10GeVData/Ta12.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        Tap13 = pd.read_csv('10GeVData/Ta13.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        Tap14 = pd.read_csv('10GeVData/Ta14.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        Tap15 = pd.read_csv('10GeVData/Ta15.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]

        Xep0 = pd.read_csv('10GeVData/Xe0.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        Xep1 = pd.read_csv('10GeVData/Xe1.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        Xep2 = pd.read_csv('10GeVData/Xe2.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        Xep3 = pd.read_csv('10GeVData/Xe3.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        Xep4 = pd.read_csv('10GeVData/Xe4.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        Xep5 = pd.read_csv('10GeVData/Xe5.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        Xep6 = pd.read_csv('10GeVData/Xe6.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        Xep7 = pd.read_csv('10GeVData/Xe7.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        Xep8 = pd.read_csv('10GeVData/Xe8.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        Xep9 = pd.read_csv('10GeVData/Xe9.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        Xep10 = pd.read_csv('10GeVData/Xe10.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        Xep11 = pd.read_csv('10GeVData/Xe11.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        Xep12 = pd.read_csv('10GeVData/Xe12.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        Xep13 = pd.read_csv('10GeVData/Xe13.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        Xep14 = pd.read_csv('10GeVData/Xe14.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        Xep15 = pd.read_csv('10GeVData/Xe15.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]    

        WRep0 = pd.read_csv('10GeVData/WRe0.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        WRep1 = pd.read_csv('10GeVData/WRe1.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        WRep2 = pd.read_csv('10GeVData/WRe2.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        WRep3 = pd.read_csv('10GeVData/WRe3.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        WRep4 = pd.read_csv('10GeVData/WRe4.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        WRep5 = pd.read_csv('10GeVData/WRe5.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        WRep6 = pd.read_csv('10GeVData/WRe6.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        WRep7 = pd.read_csv('10GeVData/WRe7.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        WRep8 = pd.read_csv('10GeVData/WRe8.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        WRep9 = pd.read_csv('10GeVData/WRe9.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        WRep10 = pd.read_csv('10GeVData/WRe10.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        WRep11 = pd.read_csv('10GeVData/WRe11.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        WRep12 = pd.read_csv('10GeVData/WRe12.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        WRep13 = pd.read_csv('10GeVData/WRe13.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        WRep14 = pd.read_csv('10GeVData/WRe14.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        WRep15 = pd.read_csv('10GeVData/WRe15.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]

        compare(ttlStr)
    else:
        LTarget = .343 # cm

        edep0 = pd.read_csv('build/out0_nt_Data.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep1 = pd.read_csv('build/out1_nt_Data.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep2 = pd.read_csv('build/out2_nt_Data.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep3 = pd.read_csv('build/out3_nt_Data.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep4 = pd.read_csv('build/out4_nt_Data.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep5 = pd.read_csv('build/out5_nt_Data.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep6 = pd.read_csv('build/out6_nt_Data.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep7 = pd.read_csv('build/out7_nt_Data.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep8 = pd.read_csv('build/out8_nt_Data.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep9 = pd.read_csv('build/out9_nt_Data.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep10 = pd.read_csv('build/out10_nt_Data.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep11 = pd.read_csv('build/out11_nt_Data.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep12 = pd.read_csv('build/out12_nt_Data.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep13 = pd.read_csv('build/out13_nt_Data.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep14 = pd.read_csv('build/out14_nt_Data.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()
        edep15 = pd.read_csv('build/out15_nt_Data.csv', header = 6, names = ['Energy Deposition [MeV]', 'In Window Dep [MeV]','Out Window Dep [MeV]']).to_numpy()

        df0 = pd.read_csv('build/out0_nt_e+.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df1 = pd.read_csv('build/out1_nt_e+.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df2 = pd.read_csv('build/out2_nt_e+.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df3 = pd.read_csv('build/out3_nt_e+.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df4 = pd.read_csv('build/out4_nt_e+.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df5 = pd.read_csv('build/out5_nt_e+.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df6 = pd.read_csv('build/out6_nt_e+.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df7 = pd.read_csv('build/out7_nt_e+.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df8 = pd.read_csv('build/out8_nt_e+.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df9 = pd.read_csv('build/out9_nt_e+.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df10 = pd.read_csv('build/out10_nt_e+.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df11 = pd.read_csv('build/out11_nt_e+.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df12 = pd.read_csv('build/out12_nt_e+.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df13 = pd.read_csv('build/out13_nt_e+.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df14 = pd.read_csv('build/out14_nt_e+.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        df15 = pd.read_csv('build/out15_nt_e+.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, :4]
        
        p0 = pd.read_csv('build/out0_nt_e+.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p1 = pd.read_csv('build/out1_nt_e+.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p2 = pd.read_csv('build/out2_nt_e+.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p3 = pd.read_csv('build/out3_nt_e+.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p4 = pd.read_csv('build/out4_nt_e+.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p5 = pd.read_csv('build/out5_nt_e+.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p6 = pd.read_csv('build/out6_nt_e+.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p7 = pd.read_csv('build/out7_nt_e+.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p8 = pd.read_csv('build/out8_nt_e+.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p9 = pd.read_csv('build/out9_nt_e+.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p10 = pd.read_csv('build/out10_nt_e+.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p11 = pd.read_csv('build/out11_nt_e+.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p12 = pd.read_csv('build/out12_nt_e+.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p13 = pd.read_csv('build/out13_nt_e+.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p14 = pd.read_csv('build/out14_nt_e+.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]
        p15 = pd.read_csv('build/out15_nt_e+.csv', header = 12, names = ['Energy [MeV]', 'Traverse Width [mm]', 'Angle [rad]', 'Rotational Angle [rad]', 'Px', 'Py', 'Pz', 'E [MeV]', 'Minkowski']).to_numpy()[:, 4:]

        if plot:
            plots(windows = wBool)
        else:
            plotArgs(windows = wBool)

