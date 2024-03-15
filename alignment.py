# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 10:40:52 2023

@author: ruizh
"""

import matplotlib.pyplot as plt
import numpy as np
import align_tools as alts
import time
import pickle
from scipy.optimize import minimize, curve_fit

# settings for inter-grating distance
d_start = 17500
d_end = 22500
nr_d = 11 
grat_d = np.linspace(d_start, d_end, num = nr_d, endpoint = True)[::-1].astype(np.int32)

# used gratings
lst_grat = ['G1', 'G2']
nr_grat = len(lst_grat)

# settings for grating rotation
nr_rotangle = 11
rot_range = 400000
G1_ori = 1250000
G2_ori = 0
G1_start = G1_ori - rot_range 
G1_end = G1_ori + rot_range 
G2_start = G2_ori - rot_range
G2_end = G2_ori + rot_range
rot_angle_1 = np.linspace(G1_start, G1_end, num = nr_rotangle).astype(np.int32)
rot_angle_2 = np.linspace(G2_start, G2_end, num = nr_rotangle).astype(np.int32)
rot_angle = np.vstack((rot_angle_1, rot_angle_2))

# path settings
path_di = r'E:\20230809\di_50avg.tif'
path_flat = r'E:\20230809\flat_50avg.tif'
dir_imgs = r'E:\20230809\alignment 1'

di = plt.imread(path_di)
di.astype(np.int16)
print('dark image read')
flat = plt.imread(path_flat)
flat.astype(np.int16)
print('flat image read')

vi_kernel = alts.generate_vignette_filter((4095, 4095), percent = 0.9, sigma = 50)

test_cutout = False
process = False
method = 'curve-fitting'
#method = 'minimization'

if test_cutout:
# empiracally choose threshold for locating harmonics
    cut_out = -2500
    i = 9
    j = 0
    k = 5   
    img = alts.load_image(dir_imgs, grat_d[i], lst_grat[j], rot_angle[j][k])
    img = ((img - di) / (flat - di))
    img_vi = alts.vignette_image(img, vi_kernel)
    img_ft = np.abs(np.fft.fftshift(np.fft.fft2(img_vi)))
    centers = alts.seg_peaks(img_ft, cut_out, radius_1 = 50, radius_2 = 20)
    popt, abs_dev = alts.line_fit(centers)
    angle = np.arctan(popt[0]) # rad
    print(f' slop: {angle:.3f}\n center_nr: {centers.shape[0]}\n diviation: {abs_dev:.2f}')
    
    offset_x = 1500
    offset_y = 1750
    crop_y = (offset_y, 4095-offset_y)
    crop_x = (offset_x, 4095-offset_x)
    fz = 14
    fig, axs = plt.subplots(2,2, figsize = (16,8))
    axs[0,0].set_title('(a)', fontsize = fz)
    axs[0,0].imshow(img[crop_y[0]:crop_y[1], crop_x[0]:crop_x[1]], vmin = 0.3, vmax = 0.5, cmap = 'gray')
    axs[0,0].set_xticks([])
    axs[0,0].set_yticks([])
    axs[0,1].set_title('(b)', fontsize = fz)
    axs[0,1].imshow(np.log(img_ft[crop_y[0]:crop_y[1], crop_x[0]:crop_x[1]]), cmap = 'gray')
    axs[0,1].set_xticks([])
    axs[0,1].set_yticks([])
    alts.plot_seg_peak(img_ft, centers, crop_y, crop_x, axs = axs[1,0])
    axs[1,0].set_title('(c)', fontsize = fz)
    alts.plot_fit_line(img_ft, popt, centers, crop_y, crop_x, axs = axs[1,1])
    axs[1,1].set_title('(d)', fontsize = fz)
    plt.tight_layout()

res_date = dir_imgs.split('\\')[-2]
if process:
# Retrieve angles and periods from Moire patterns
    cut_out = -2500
    t1 = time.time()
    angles_info = {}
    for i in range(nr_d):
        print(f'Angle extraction distance: {i} / {nr_d - 1}')
        for j in range(nr_grat):
            print(f'Grating: {j} / {nr_grat - 1}')
            for k in range(nr_rotangle):
                t2 = time.time()
                img = alts.load_image(dir_imgs, grat_d[i], lst_grat[j], rot_angle[j][k])
                img = ((img - di) / (flat - di))
                img_vi = alts.vignette_image(img, vi_kernel)
                img_ft = np.abs(np.fft.fftshift(np.fft.fft2(img_vi)))
                centers = alts.seg_peaks(img_ft, cut_out = cut_out, radius_1 = 50, radius_2 = 20)
                popt, abs_dev = alts.line_fit(centers)
                angle = np.arctan(popt[0]) # rad
                if len(centers) % 2 == 0:
                    abs_dev = 999
                    period = 999
                else:
                    centers -= 2047
                    freq_dis = np.sort(np.sqrt(centers[:,0]**2 + centers[:,1]**2))[::2]
                    freq = freq_dis * (1 / 4095)
                    period = 1 / freq
                    period = (period[1:] * np.arange(1, len(period[1:])+1)).mean()
                data_name = f'{grat_d[i]}_{lst_grat[j]}_{rot_angle[j][k]}'
                angles_info[data_name] = {}
                angles_info[data_name]['nr_pts'] = centers.shape[0]
                angles_info[data_name]['angle'] = angle
                angles_info[data_name]['abs_dev'] = abs_dev
                angles_info[data_name]['period'] = period
                t3 = time.time()
                print(f'slop: {angle:.3f}|center_nr: {centers.shape[0]}|diviation: {abs_dev:.2f}|'
                      + f'period: {period:.2f}|time: {t3-t2:.1f}')
    t4 = time.time()
    print(f'Time used: {t4 - t1}')
    
    with open(f'D:\Ph.D\Alignment\Gsense\\data_{res_date}.pickle', 'wb') as handle:
        pickle.dump(angles_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

fit_res = {}
with open(f'D:\Ph.D\Alignment\Gsense\\data_{res_date}.pickle', 'rb') as handle:
    input_dict = pickle.load(handle)
ori_dict = {}
ori_dict['G1_ori'] = G1_ori
ori_dict['G2_ori'] = G2_ori
ori_dict['d_ori'] = d_end

# construct fitting dataset
d_select = np.linspace(22500, 17500, 11).astype(np.int32)[::1]
grat_select = ['G1', 'G2']
rot_select = np.linspace(-400000, 400000, 11).astype(np.int32)[::1]
select_dict = alts.select_keys(input_dict, ori_dict, d_select, grat_select, rot_select)
filter_dict = alts.filter_data(select_dict, lim_dev = 1, lim_nr = 3)

# Data fitting
del_d, del_a1, del_a2, a_m, period = alts.cons_fitting_arr(filter_dict, ori_dict)
x0 = (0.5, 0.002, 0.1, 0.1)
popt, pcov = curve_fit(alts.lsq_fitting_angle, (del_d, del_a1, del_a2), a_m, p0 = x0)
r1, rg, a1, a2 = popt
paras = r1, rg, a1, a2
p_g = 1e-6
p_d = 16.4 * 1e-6
res = minimize(alts.cost_period, x0 = 1,
               args = (paras, p_g, p_d, period, del_d, del_a1, del_a2), 
               method = 'Nelder-Mead')
r2 = float(res.x)
p_m = period * p_d
a_m_est = alts.Moire_angle(r1, rg + del_d, a1 + del_a1, a2 + del_a2)
p_m_est = alts.Moire_period(r1, r2, rg + del_d, a1+ del_a1, a2 + del_a2, p_g)

print(f'\nAverage absolute deviation (angle): {np.abs(a_m_est - a_m).mean() * 180 / np.pi} deg')
print(f'Average absolute deviation (period): {np.abs(p_m_est - p_m).mean() * 1e6} um \n')

print(  f'r1 = {r1:.4f} m \n'
      + f'r2 = {r2:.4f} m \n'
      + f'rg = {rg*1e3:.4f} mm (d_ori = {d_end})\n'
      + f'G1 abs angle: {a1 * 180 / np.pi * 1e6} udeg\n'
      + f'G2 abs angle: {a2 * 180 / np.pi * 1e6} udeg\n'
      + f'G1 move to: {G1_ori - a1 * 180 / np.pi * 1e6} udeg\n'
      + f'G2 move to: {G2_ori - a2 * 180 / np.pi * 1e6} udeg\n')

# Results plots
fz = 14
fig, axs = plt.subplots(2, figsize = (16,8), sharex = True)
axs[0].plot(a_m * 180 / np.pi, 'ro-', label = 'Exp')
axs[0].plot(a_m_est * 180 / np.pi, 'b*', label = 'Est')
axs[0].set_ylabel('Moire angle (deg)', fontsize = fz)
axs[0].set_title('(a)', fontsize = fz)
axs[0].tick_params(axis='x', labelsize = fz)
axs[0].tick_params(axis='y', labelsize = fz)
axs[1].plot(p_m * 1e6, 'ro-')
axs[1].plot(p_m_est * 1e6, 'b*')
axs[1].set_xlabel('Data point index', fontsize = fz)
axs[1].set_ylabel('$p_{m}$ ($\mu m$)', fontsize = fz)
axs[1].set_title('(b)', fontsize = fz)
axs[1].tick_params(axis='x', labelsize = fz)
axs[1].tick_params(axis='y', labelsize = fz)
fig.legend(fontsize = fz, loc = 'right', frameon=True)



















