# -*- coding: utf-8 -*-
"""
Created on Wed May 31 09:44:09 2023

@author: ruizh
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy import ndimage
import random

def load_image(dir_imgs, dis, grat, angle):
    dis_s = f'\\D_{dis}'
    grat_s = f'\\{grat}'
    angle_s = f'\\Angle_{angle}.tif'
    path_img = dir_imgs + dis_s + grat_s + angle_s
    img = plt.imread(path_img)
    return img

def generate_vignette_filter(img_shape, percent, sigma):
    x_shape, y_shape = img_shape
    x_c = (x_shape - 1) // 2
    y_c = (y_shape - 1) // 2
    r = np.round(np.min([x_c, y_c]) * percent).astype(np.int32)
    x = np.linspace(0, x_shape, x_shape, endpoint = False)
    y = np.linspace(0, y_shape, y_shape, endpoint = False)
    xv, yv = np.meshgrid(x, y)
    kernel_inner = ((xv - x_c)**2 + (yv - y_c)**2 < r**2).astype(np.int32)
    kernel_outer = ((xv - x_c)**2 + (yv - y_c)**2 >= r**2).astype(np.int32)
    radius_kernel = np.sqrt((xv - x_c)**2 + (yv - y_c)**2) - r
    radius_kernel = radius_kernel * kernel_outer
    gauss_kernel = np.exp(-1/2 * (radius_kernel / sigma)**2)
    vi_kernel = gauss_kernel * kernel_outer + kernel_inner
    return vi_kernel

def vignette_image(img, vi_kernel, default = 0.35):
    image = img.copy()
    image_vi = image * vi_kernel
    image_vi = np.where(image_vi == -np.inf, default, image_vi)
    image_vi = np.where(image_vi == np.inf, default, image_vi)
    image_vi[np.isnan(image_vi)] = default
    return image_vi

def seg_peaks(img_ft, cut_out = -2000, radius_1 = 50, radius_2 = 20):
    shape_y, shape_x = img_ft.shape[0], img_ft.shape[1]
    threshold = np.sort(img_ft, axis = None)[cut_out]
    mask = img_ft >= threshold
    s = np.ones((3,3)).astype(np.int8)
    labels, num_label = ndimage.label(mask, structure = s)
    sel = img_ft * mask
    centers_raw = ndimage.center_of_mass(sel, labels = labels, 
                                         index = np.arange(1, num_label + 1))
    centers = []
    masks = np.zeros((shape_y, shape_x)).astype(np.uint8)
    for y, x in centers_raw:
        Y, X = np.ogrid[0:shape_y, 0:shape_x]
        circle_mask = np.sqrt((Y - y)**2 + (X - x)**2) <= radius_1
        # redirect to the highest center
        roi = img_ft * circle_mask
        y_idx, x_idx = np.where(roi == roi.max())[0][0], np.where(roi == roi.max())[1][0]
        circle_mask = np.sqrt((Y - y_idx)**2 + (X - x_idx)**2) <= radius_2
        roi = img_ft * circle_mask
        centers.append(ndimage.center_of_mass(roi)) # calculate center of mass for each spot
    masks = masks >= 1
    # remove the very close but separate labels
    centers = np.array(centers)
    centers = np.unique(centers, axis = 0)
    return centers

def linear_func(x, a, b):
    y = a*x + b
    return y

def line_fit(centers):
    x_pos, y_pos = centers[:,1], centers[:,0]
    popt, pcov = curve_fit(linear_func, x_pos, y_pos)
    y_fit = linear_func(x_pos, *popt)
    dev = np.abs(y_fit - y_pos).mean()
    return popt, dev

def plot_seg_peak(img_ft, centers, crop_y, crop_x, axs = False):
    ymin, ymax = crop_y
    xmin, xmax = crop_x
    shape_y, shape_x = img_ft.shape[0], img_ft.shape[1]
    Y, X = np.ogrid[0:shape_y, 0:shape_x]
    radius_2 = 20
    mask = np.zeros_like(img_ft)
    for y, x in centers:
        circle = np.sqrt((Y - y)**2 + (X - x)**2) <= radius_2
        mask = mask + circle
    img_plot = np.log(img_ft[ymin : ymax, xmin : xmax]) * mask[ymin : ymax, xmin : xmax]
    if axs == False:
        plt.figure()
        plt.imshow(img_plot, cmap = 'gray')
        plt.xticks([])
        plt.yticks([])
    else:
        axs.imshow(img_plot, cmap = 'gray')
        axs.set_xticks([])
        axs.set_yticks([])

def plot_fit_line(img_ft, popt, centers, crop_y, crop_x, axs = False):
    ymin, ymax = crop_y
    xmin, xmax = crop_x
    img_plot = np.log(img_ft[ymin : ymax, xmin : xmax])
    x_plot_temp = np.array(crop_x)
    y_plot_temp = linear_func(x_plot_temp, *popt)
    x_plot = x_plot_temp - crop_x[0]
    y_plot = y_plot_temp - crop_y[0]
    x_plot[0] += 5
    x_plot[1] -= 5
    if axs == False:
        plt.figure()
        plt.imshow(img_plot, cmap = 'gray')
        plt.plot(x_plot, y_plot, color="white", linewidth=0.3)
        for cm_y, cm_x in centers:
            cm_y -= crop_y[0]
            cm_x -= crop_x[0]
            plt.plot(cm_x, cm_y, 'kx', ms = 5)
        plt.xticks([])
        plt.yticks([])
    else:
        axs.imshow(img_plot, cmap = 'gray')
        axs.plot(x_plot, y_plot, color="white", linewidth=0.3)
        for cm_y, cm_x in centers:
            cm_y -= crop_y[0]
            cm_x -= crop_x[0]
            axs.plot(cm_x, cm_y, 'kx', ms = 5)
        axs.set_xticks([])
        axs.set_yticks([])      
    
def Moire_angle(r1, rg, a1, a2):
    numer = (r1 + rg) * np.sin(a2) - r1 * np.sin(a1)
    denom = np.sqrt((r1 + rg)**2 + r1**2 - 2 * r1 * (r1 + rg) * np.cos(a2 - a1))
    a_m = np.arcsin(numer / denom)
    return a_m

def Moire_period(r1, r2, rg, a1, a2, p_g):
    numer = (r1 + r2) * p_g
    denom = np.sqrt((r1 + rg)**2 + r1**2 - 2 * r1 * (r1 + rg) * np.cos(a2 - a1))
    p_m = numer / denom
    return p_m

def Moire_period_am(r1, r2, rg, a1, a2, a_m, p_g):
    numer = (r1 + r2) * p_g
    denom = (r1 + rg) * np.cos(a2) - r1 * np.cos(a1)
    p_m = numer / denom * np.cos(a_m)
    return p_m

def cost_angle(paras, a_m_arr, delta_d_arr, delta_a1_arr, delta_a2_arr):
    r1, rg, a1, a2 = paras
    dev = a_m_arr - Moire_angle(r1, rg + delta_d_arr, a1 + delta_a1_arr, a2 + delta_a2_arr)
    lost = np.abs(dev).sum()
    return lost

def cost_period(r2, paras, p_g, p_d, period_arr, delta_d_arr, delta_a1_arr, delta_a2_arr):
    r1, rg, a1, a2 = paras
    p_m = period_arr * p_d
    dev = p_m - Moire_period(r1, r2, rg + delta_d_arr, a1+ delta_a1_arr, 
                                      a2 + delta_a2_arr, p_g)
    lost = np.abs(dev).sum()
    return lost

def cost_period_am(r2, paras, p_g, a_m_arr, period_arr, delta_d_arr, delta_a1_arr, delta_a2_arr):
    r1, rg, a1, a2 = paras
    p_m = period_arr * 16.4 * 1e-6
    dev = p_m - Moire_period_am(r1, r2, rg + delta_d_arr, a1+ delta_a1_arr, 
                                      a2 + delta_a2_arr, a_m_arr, p_g)
    lost = np.abs(dev).sum()
    return lost

def filter_data(input_dict, lim_dev, lim_nr):
    keys_copy = tuple(input_dict.keys())
    filter_dict = {}
    num_ori = len(input_dict.keys())
    i = 0
    for key in keys_copy:
        if input_dict[key]['abs_dev'] <= lim_dev and input_dict[key]['nr_pts'] > lim_nr:
            filter_dict[key] = input_dict[key].copy()
            i += 1
    print(f'{i} / {num_ori} data points are kept')
    return filter_dict

def select_keys(input_dict, ori_dict, d_select, grat_select, rot_select):
    d_select = d_select.astype(np.int32)
    rot_select = rot_select.astype(np.int32)
    select_dict = {}
    for key in input_dict:
        d, grat, angle = tuple(key.split('_'))
        if grat in grat_select:
            if grat == 'G1':
                rot = int(angle) - ori_dict['G1_ori']
            else:
                rot = int(angle) - ori_dict['G2_ori']
            if int(d) in d_select:
                if rot in rot_select:
                    select_dict[key] = input_dict[key].copy()
    return select_dict
    
def cons_fitting_arr(input_dict, ori_dict):
    delta_d_arr = np.array([])
    delta_a1_arr = np.array([])
    delta_a2_arr = np.array([])
    a_m_arr = np.array([])
    period_arr = np.array([])
    for key in input_dict:
        d, grat, angle = tuple(key.split('_'))
        delta_d = (ori_dict['d_ori'] - float(d)) / 1e6 # unit: m
        if grat == 'G1':
            delta_a1 = (float(angle) - ori_dict['G1_ori']) / 1e6 * np.pi / 180
            delta_a2 = 0
        if grat == 'G2':
            delta_a2 = (float(angle) - ori_dict['G2_ori']) / 1e6 * np.pi / 180
            delta_a1 = 0
        delta_d_arr = np.append(delta_d_arr, delta_d)
        delta_a1_arr = np.append(delta_a1_arr, delta_a1)
        delta_a2_arr = np.append(delta_a2_arr, delta_a2)
        a_m_arr = np.append(a_m_arr, input_dict[key]['angle'])
        period_arr = np.append(period_arr, input_dict[key]['period'])
    return delta_d_arr, delta_a1_arr, delta_a2_arr, a_m_arr, period_arr

def lsq_fitting_angle(X, r1, rg, a1, a2):
    delta_d_arr, delta_a1_arr, delta_a2_arr = X
    angle = Moire_angle(r1, rg + delta_d_arr, a1 + delta_a1_arr, a2 + delta_a2_arr)
    return angle

def lsq_fitting_period(X, r2):
    delta_d_arr, delta_a1_arr, delta_a2_arr, r1, rg, a1, a2, p_g = X
    p_m = Moire_period(r1, r2, rg + delta_d_arr, a1 + delta_a1_arr, a2 + delta_a2_arr, p_g)
    return p_m

def lsq_fitting_period_onestep(X, r1, r2, rg, a1, a2):
    delta_d_arr, delta_a1_arr, delta_a2_arr, p_g = X
    p_m = Moire_period(r1, r2, rg + delta_d_arr, a1 + delta_a1_arr, a2 + delta_a2_arr, p_g)
    return p_m








