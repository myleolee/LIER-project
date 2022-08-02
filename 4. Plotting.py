# -*- coding: utf-8 -*-
"""
Created on Wed Apr 6 21:32:43 2022


@author: Leo Lee (modified from code created by Gerome Algodon)

LIER project data analysis pipeline 
4. Graph Plotting

Input: 3 bin_stack.fits files, 3 control_bin_stack.fits files, 3 bin.fits files
Output: 
    stacked_residual.fits
    X Figures(.png):
        Figures 4.1: Strong line spectum plotted with control line spectrum (left), raw residual spectrum (right)
        Figures 4.2: Residual spectra after Bspline correction
        Figures 4.3: Stacked residual spectra
        Figures 4.4: Regions of integration and background residual of the emission lines
        Figures 4.5: Plots of the auroral lines in the reduced spectrum
        Figures 4.6: Plots of log ratios between emission lines
"""
#%%
#   Importing required packages

from tabulate import tabulate
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splrep, BSpline
from matplotlib import rc
import scipy.stats
import os
import lier
from scipy.special import wofz
from scipy.optimize import curve_fit
import astropy.constants
import extinction as ext
from matplotlib import container
    
font = { 'size' : 10 , 'family' : 'serif'}  # size of normal font
fs = 14  # size of titles, must be manually inserted using ',fontsize=fs'
rc('font', **font)

c_vel = astropy.constants.c.to('km/s').value
#%%
#   Check whether the directories exist, if not create the directories
def check_dir(path):
   if (os.path.exists(path) == False):
       os.makedirs(path) 

dir_array = ['../Output/4. Plotting/Figures 4.1',  '../Output/4. Plotting/Figures 4.1/high',  '../Output/4. Plotting/Figures 4.1/mid',
             '../Output/4. Plotting/Figures 4.1/low', '../Output/4. Plotting/Figures 4.2', '../Output/4. Plotting/Figures 4.2/high',
             '../Output/4. Plotting/Figures 4.2/mid', '../Output/4. Plotting/Figures 4.2/low', '../Output/4. Plotting/Figures 4.3',
             '../Output/4. Plotting/Figures 4.4', '../Output/4. Plotting/Figures 4.4/high', '../Output/4. Plotting/Figures 4.4/mid',
             '../Output/4. Plotting/Figures 4.4/low', '../Output/4. Plotting/Figures 4.5', '../Output/4. Plotting/Figures 4.5/high',
             '../Output/4. Plotting/Figures 4.5/mid', '../Output/4. Plotting/Figures 4.5/low', '../Output/4. Plotting/Figures 4.6',
             '../Output/4. Plotting/Figures 4.6/high', '../Output/4. Plotting/Figures 4.6/mid', '../Output/4. Plotting/Figures 4.6/low',
             '../Output/4. Plotting/Figures 4.7']

for p in dir_array:
    check_dir(p)

#%% 
#   Import data on stacks (created in 3. stacking.py) and bins.fits file

spaxel_data_table = fits.open('../Data/spaxel_data_table.fits')

bin_1,bin_1_control = fits.open('../Data/Bin/Bin_1.fits'),fits.open('../Data/Bin/Bin_1_Control.fits')
bin_2,bin_2_control = fits.open('../Data/Bin/Bin_2.fits'),fits.open('../Data/Bin/Bin_2_Control.fits')
bin_3,bin_3_control = fits.open('../Data/Bin/Bin_3.fits'),fits.open('../Data/Bin/Bin_3_Control.fits')

stack_1 = fits.open('../Data/Stack/bin_1_stack.fits')
stack_2 = fits.open('../Data/Stack/bin_2_stack.fits')
stack_3 = fits.open('../Data/Stack/bin_3_stack.fits')

stack_1_control = fits.open('../Data/Stack/control_bin_1_stack.fits')
stack_2_control = fits.open('../Data/Stack/control_bin_2_stack.fits')
stack_3_control = fits.open('../Data/Stack/control_bin_3_stack.fits')

number_of_velocity_offset_bins = len(bin_1) - 1
wave = stack_1[0].data  

#%%
#   4.1 Bspline smoothing

def generate_plot(stack, stack_control, metal_bin):
    tot_bin_residual = np.zeros((number_of_velocity_offset_bins, len(wave)))    # For returning the residual spectra
    tot_var = np.zeros((number_of_velocity_offset_bins, len(wave)))    # For returning the residual spectra
    for s in range(number_of_velocity_offset_bins):     # s is the number of subbins
        
        #   1. Generate plots of the stacked spectrum and the residual spectrum of each bin (Figures 4.1)
        
        strong = stack[1].data[s]
        strong_var = stack[3].data[s]
        control = stack_control[1].data[s]
        control_var = stack_control[3].data[s]
        
        residual = strong - control
        
        plt.figure(figsize=(18,7))
        plt.suptitle('Subbin {}'.format(s+1))
           
        plt.subplot(1,2,1)
        plt.plot(wave,strong,label='Strong Line Spectum')
        plt.plot(wave,control,label='Control Line Spectrum')
        plt.xlim((wave[0],wave[-1]))
        plt.ylim(0, 2)
        plt.title('Strong and control spectra of {} metallicity bin'.format(metal_bin))
        plt.legend()
        
        plt.subplot(1,2,2)
        plt.plot(wave,residual)
        plt.xlim((wave[0],wave[-1]))
        plt.ylim(-0.6, 0.8)
        plt.title('Residual spectrum of {} metallicity bin'.format(metal_bin))

        plt.savefig('../Output/4. Plotting/Figures 4.1/{}/Subbin_{}.png'.format(metal_bin, s+1), format='png')
        plt.close()
        
        #   2. Define the emission line regions for excluding those region in BSpline smoothing
        
        lines = np.array([3728.6,3836.485,3869.86,3889.750,3890.158,3933, 3971.202,    # copy-pasted from Renbin's code
                          4070,4078,4102.899,4305,4341.692,4862.691,4959,5007,5200.317,
                          5756.19,5877.249, 6300, 6549.86, 6564.632,6585.27, 6718.294,6732.674,
                          7321.475, 7332.215, 9071.1, 9533.2])
        line_width = 12     # Define the width of the emission line region
        
        lineless = np.full(len(wave),True)      # Initialize array for lineless values
        width = np.full(len(lines),line_width)
        
        #   3. Construct lineless spectrum using the region defined above
       
        for l in range(len(lines)):
            lineless[((wave>(lines[l]-width[l]))&(wave<(lines[l]+width[l])))]=False
        
        #   4. Construct the spline for smoothing out the spectrum
        
        var = (strong/control)**2 * ((strong_var/strong**2)+ (control_var/control**2))
        isd = 1/(np.sqrt(var))
        sm_co = 0
        itm = np.arange(1, len(wave[lineless]), 50)
        tck = splrep(wave[lineless], strong[lineless]/control[lineless], k=3, w=isd[lineless], s=sm_co, t=wave[lineless][itm], per=0)
        spl = BSpline(tck[0], tck[1], tck[2])
        
        #   5. Construct and plot new flat residual after the Bspline correcting (Figures 4.2)
        
        residual_new = strong - control*spl(wave)
        
        plt.figure(figsize=(10,7))
        plt.plot(wave,residual_new)
        plt.xlim((wave[0],wave[-1]))
        plt.ylim((-0.6,0.8))
        plt.title('Bspline smoothed spectrum, {} metallicity bin - subbin {}'.format(metal_bin,s+1))

        plt.savefig('../Output/4. Plotting/Figures 4.2/{}/Subbin_{}.png'.format(metal_bin, s+1), format='png')
        plt.close()
        
        tot_bin_residual[s,:] = residual_new      # Save the spectrum into a single array
        
        #   6. Calculate the variance of each specrum in each velocity subbin
        total_var = strong_var + control_var*(spl(wave)**2)
        tot_var[s,:] = total_var
        
    #   6. Return every resudual spectrum
    
    tot_bin_residual[np.isnan(tot_bin_residual)] = 0    # Prevent Nah values ruining the stacking in next step
    tot_var[np.isnan(tot_var)] = 0 
    return tot_bin_residual, tot_var
        
bin_1_resid, bin_1_resid_var = generate_plot(stack_1, stack_1_control, 'high')
bin_2_resid, bin_2_resid_var = generate_plot(stack_2, stack_2_control, 'mid')
bin_3_resid, bin_3_resid_var = generate_plot(stack_3, stack_3_control, 'low')

#%%
#   Stack the spectra of the different velocity offset bins together and save the spectrum of different bins as a single fits file

#   1. Find the number of spaxels in every subbin

spx_num_1,spx_num_2,spx_num_3 = np.zeros(number_of_velocity_offset_bins), np.zeros(number_of_velocity_offset_bins), np.zeros(number_of_velocity_offset_bins)

for i in range(number_of_velocity_offset_bins):
    spx_num_1[i] = np.shape(bin_1[i].data)[0]
    spx_num_2[i] = np.shape(bin_2[i].data)[0]
    spx_num_3[i] = np.shape(bin_3[i].data)[0]

#   2. Stack the residuals together weighted by the number of subbin spaxels divided by total spaxels in the metallicty bin

bin1 = np.average(bin_1_resid, axis=0, weights = spx_num_1/(spx_num_1.sum()))
bin1_var = np.average(bin_1_resid_var, axis = 0 , weights = spx_num_1**2)*np.sum(spx_num_1**2)/(spx_num_1.sum()**2)
bin2 = np.average(bin_2_resid, axis=0, weights = spx_num_2/(spx_num_2.sum()))
bin2_var = np.average(bin_2_resid_var, axis = 0 , weights = spx_num_2**2)*np.sum(spx_num_2**2)/(spx_num_2.sum()**2)
bin3 = np.average(bin_3_resid, axis=0, weights = spx_num_3/(spx_num_3.sum()))
bin3_var = np.average(bin_3_resid_var, axis = 0 , weights = spx_num_3**2)*np.sum(spx_num_3**2)/(spx_num_3.sum()**2)

#   3. Plot out the stacked spectrum (Figures 4.3)

plt.figure(figsize=(20,7))
plt.plot(wave, bin1)
plt.axhline(y = 0, color = 'r', linestyle = '-')
plt.xlim((wave[0],wave[-1]))
plt.ylim([-0.05,0.4])
plt.title('Stacked spectra, High metalicity bin')
plt.savefig('../Output/4. Plotting/Figures 4.3/high.png', format='png')
plt.close()

plt.figure(figsize=(20,7))
plt.plot(wave, bin2)
plt.axhline(y = 0, color = 'r', linestyle = '-')
plt.xlim((wave[0],wave[-1]))
plt.ylim([-0.05,0.4])
plt.title('Stacked spectra, Mid metalicity bin')
plt.savefig('../Output/4. Plotting/Figures 4.3/mid.png', format='png')
plt.close()

plt.figure(figsize=(20,7))
plt.plot(wave, bin3)
plt.axhline(y = 0, color = 'r', linestyle = '-')
plt.xlim((wave[0],wave[-1]))
plt.ylim([-0.05,0.4])
plt.title('Stacked spectra, Low metalicity bin')
plt.savefig('../Output/4. Plotting/Figures 4.3/low.png', format='png')
plt.close()

#   4. Save the stacked spectra into a single fits file

fit = fits.HDUList()
fit.append(fits.PrimaryHDU(wave))
fit.append(fits.ImageHDU(bin1, name='high_stacked_spectrum'))
fit.append(fits.ImageHDU(bin1_var, name='high_stacked_spectrum_var'))
fit.append(fits.ImageHDU(bin2, name='mid_stacked_spectrum'))
fit.append(fits.ImageHDU(bin2_var, name='mid_stacked_spectrum_var'))
fit.append(fits.ImageHDU(bin3, name='low_stacked_spectrum'))
fit.append(fits.ImageHDU(bin3_var, name='low_stacked_spectrum_var'))
fit.writeto('../Data/Stack/stacked_spectra.fits', overwrite=True)

#%%
#   Read the stacked spectra

temp = fits.open('../Data/Stack/stacked_spectra.fits')
wave = temp[0].data
high = temp['HIGH_STACKED_SPECTRUM'].data
high_var = temp['HIGH_STACKED_SPECTRUM_VAR'].data
mid= temp['MID_STACKED_SPECTRUM'].data
mid_var = temp['MID_STACKED_SPECTRUM_VAR'].data
low = temp['LOW_STACKED_SPECTRUM'].data
low_var = temp['LOW_STACKED_SPECTRUM_VAR'].data
del temp

#%%
#   Define the emission lines


#   1. Strong emission lines

#   Doublet with [OII]3729, vacuum wavelength = 3727.092, 3729.875
OIId = lier.doublet(3727.092, 3729.875,[3697.36, 3717.36], [3739.36, 3759.36], 'OII 3727, 3729', 'strong')

#   Doublet with [NeIII] 3968, vacuum wavelength = 3869.86 (calculate only single line)
NeIII = lier.singlet(3869.86, [3839.85, 3858], [3879.85, 3886.85], 'NeIII 3869', 'strong')

#   Vacuum wavelength = 4862.721
H_beta = lier.singlet(4862.721, [4798.88, 4838.88], [4885.62, 4925.62], 'H beta', 'strong')

#   Vacuum wavelength = 5008.24
OIII = lier.singlet(5008.24, [4975.2, 4995.2], [5021.2, 5041.2], 'OIII 5007', 'strong')

#   Vacuum wavelength = 6302.046
OI = lier.singlet(6302.046, [6267.05,6287.05], [6317.05,6337.05], 'OI 6300', 'strong')

#   Vacuum wavelength = 6564.6
H_alpha = lier.singlet(6564.6, [6483,6538], [6598,6653], 'H alpha', 'strong')

#   Doublet with [NII] 6550, but proportion locked by quantum mechanics, so only include 6585, vacuum wavelength = 6549.86, 6585.27
NII = lier.singlet(6585.27, [6483, 6538], [6598, 6653], 'NII 6583', 'strong')

#   Vacuum wavelength = 6718.295, 6732.674
SIId = lier.doublet(6718.295, 6732.674, [6686.48, 6704.48], [6746.48, 6764.48], 'SIId 6716, 6731', 'strong')


#   2. Weak emission lines

#   Vacuum wavelength = 3890.166
H_zeta = lier.singlet(3890.166, [3835, 3860], [3894.08, 3919.08], 'H zeta', 'weak')

#   Vacuum wavelegnth = 3971.198
H_epsilon = lier.singlet(3971.198, [3952.2, 3960.2], [3979.2, 3987.2], 'H epsilon', 'weak')

#   Vacuum wavelength = 4069.75, 4077.5
SIIdf = lier.doublet(4069.75, 4077.5, [4000, 4060], [4140, 4200], 'SII 4068, 4076', 'weak')

#   Vacuum wavelength = 4102.892
H_delta = lier.singlet(4102.892, [4086.892,4094.892], [4110.892,4118.892], 'H delta', 'weak')

#   Vacuum wavelength = 4341.692
H_gamma = lier.singlet(4341.692, [4326.692,4334.692], [4348.692,4356.692], 'H gamma', 'weak')

#   Vacuum wavelength = 4364.436
OIIIf = lier.singlet(4364.436, [4248.435, 4328.435], [4374.435, 4454.435], 'OIII 4363', 'weak')

#   Vacuum wavelength = 5756.119
NIIf = lier.singlet(5756.119, [5666.2, 5746.2], [5766.2, 5846.2], 'NII 5755', 'weak')

#   Inaccurate gaussian fitting, use custom integration range instead, Vacuum wavelength = 7321.94, 7332.21 
OIIdf = lier.doublet(7321.94, 7332.21, [7252.08, 7312.08], [7342.08, 7402.08], 'OII 7320, 7330', 'weak')

#   Vacuum wavelength = 9071.1, 9533.2
SIIId = lier.doublet(9071.1, 9533.2, [9001.1, 9051.1], [9553.2, 9578.2], 'SIII 9069, 9532', 'weak')


#   3. Array of all strong lines and weak lines
strong_lines = [OIId, NeIII, H_beta, OIII, OI, H_alpha, NII, SIId]
weak_lines = [H_zeta, H_epsilon, SIIdf, H_delta, H_gamma, OIIIf, NIIf, OIIdf]
hlines = [H_alpha, H_beta, H_gamma, H_delta, H_epsilon, H_zeta]
 
        
#%%
#   Overplots different profile fitting different emission lines (Figures 4.4)
    
#   Function for fitting singlets
def overplot_singlet(line, wave, flux, path=''):
    continuum = np.where(((wave > line.left[0]) & (wave < line.left[1])) | ((wave > line.right[0]) & (wave < line.right[1])))[0]
    coeff = np.polyfit(wave[continuum], flux[continuum], 1)
    fitline = wave*coeff[0] + coeff[1]
    reduced_spectrum = flux - fitline
    centre = [line.w-(line.sigma(wave,flux)*6), line.w+(line.sigma(wave,flux)*6)]
    centre_region = np.where((wave >= centre[0]) & (wave <= centre[1]))[0]
    continuum = np.where(((wave > line.left[0]) & (wave < line.left[1])) | ((wave > line.right[0]) & (wave < line.right[1])))[0]
    def gauss(x, *p):
        A, mean, sigma = p
        return A*np.exp(-(x-mean)**2/(2.*sigma**2))
    def lorentz(x, *p):
        A, mean, gamma = p
        return A*((0.5*gamma)/((x-mean)**2+(0.5*gamma)**2))
    def voigt(x, *p):
        A, mean, sigma, gamma = p
        return A * np.real(wofz((x - mean + 1j*gamma)/sigma/np.sqrt(2))) / sigma /np.sqrt(2*np.pi)
    pg = [1, line.w, 1]
    pl = [1, line.w, 2.5]
    pv = [1, line.w, 1, 1]
    plt.figure(figsize=(7,5))
    plt.axvspan(line.left[0], line.left[1], color = 'red', alpha=0.1, linestyle = '-', label = 'continuum')
    plt.axvspan(line.right[0], line.right[1], color = 'red', alpha=0.1, linestyle = '-')
    plt.axvline(line.w, color = 'k', linestyle = '--')   
    coeff, var_matrix = curve_fit(gauss, wave, reduced_spectrum, p0=pg, method = 'lm')
    gau = coeff[0]*np.exp(-(wave-coeff[1])**2/(2.*coeff[2]**2))    
    plt.plot(wave[centre_region], gau[centre_region], label='Gaussian fit', linestyle = '--', color = 'orange')
    coeff, var_matrix = curve_fit(lorentz, wave, reduced_spectrum, p0=pl, method = 'lm')
    lor = coeff[0]*((0.5*coeff[2])/((wave-coeff[1])**2+(0.5*coeff[2])**2))       
    plt.plot(wave[centre_region], lor[centre_region], label='Lorentzian fit', linestyle = '--', color = 'green')
    coeff, var_matrix = curve_fit(voigt, wave, reduced_spectrum, p0=pv, method = 'lm')
    voi = coeff[0] * np.real(wofz((wave - coeff[1] + 1j*coeff[3])/coeff[2]/np.sqrt(2))) / coeff[2] /np.sqrt(2*np.pi)
    plt.plot(wave[centre_region], voi[centre_region], label='Voigt fit', linestyle = '-', color = 'red')
    plt.plot(wave, flux)
    plt.plot(wave[continuum], fitline[continuum], color = 'black')            
    #plt.xlim((x[continuum[0]-10],x[continuum[-1]+10]))
    plt.xlim(wave[centre_region[0]], wave[centre_region[-1]])
    y_temp = np.max(flux[centre_region])
    plt.ylim([-0.2*y_temp, 1.5*y_temp])
    plt.title('{}'.format(line.name))
    plt.legend()
    if (len(path)!=0):
        plt.savefig('{}/{}_overplot.png'.format(path, line.name), format='png', dpi = 1200)
        plt.close()
    else:
        plt.show()

for i in strong_lines[1:4]:
    overplot_singlet(i, wave, high, path = '../Output/4. Plotting/Figures 4.4/high')
    overplot_singlet(i, wave, mid, path = '../Output/4. Plotting/Figures 4.4/mid')
    overplot_singlet(i, wave, low, path = '../Output/4. Plotting/Figures 4.4/low')

#   Funtions for fitting doublets

def overplot_doublet(line, wave, flux, path=''):
    continuum = np.where(((wave > line.left[0]) & (wave < line.left[1])) | ((wave > line.right[0]) & (wave < line.right[1])))[0]
    coeff = np.polyfit(wave[continuum], flux[continuum], 1)
    fitline = wave*coeff[0] + coeff[1]
    reduced_spectrum = flux - fitline
    centre = [line.w1-(line.sigma(wave,flux)*6), line.w2+(line.sigma(wave,flux)*6)]
    centre_region = np.where((wave >= centre[0]) & (wave <= centre[1]))[0]
    continuum = np.where(((wave > line.left[0]) & (wave < line.left[1])) | ((wave > line.right[0]) & (wave < line.right[1])))[0]
    def gauss(x, *p):
        A, B, mean, sigma = p
        return A*np.exp(-(x-mean)**2/(2.*sigma**2)) + B*np.exp(-(x-(mean*line.w2/line.w1))**2/(2.*sigma**2))
    def lorentz(x, *p):
        A, B, mean, gamma = p
        return A*((0.5*gamma)/((x-mean)**2+(0.5*gamma)**2)) + B*((0.5*gamma)/((x-(mean*line.w2/line.w1))**2+(0.5*gamma)**2))
    def voigt(x, *p):
        A, B, mean, sigma, gamma= p
        return A * np.real(wofz((x - mean + 1j*gamma)/sigma/np.sqrt(2))) / sigma /np.sqrt(2*np.pi) + B * np.real(wofz((x - (mean*line.w2/line.w1) + 1j*gamma)/sigma/np.sqrt(2))) / sigma /np.sqrt(2*np.pi)
    pg = [1, 1, line.w1, 1]
    pl = [1, 1, line.w1, 2.5]
    pv = [1, 1, line.w1, 1, 1]
    plt.figure(figsize=(7,5))
    plt.axvspan(line.left[0], line.left[1], color = 'red', alpha=0.1, linestyle = '-', label = 'continuum')
    plt.axvspan(line.right[0], line.right[1], color = 'red', alpha=0.1, linestyle = '-')
    plt.axvline(line.w1, color = 'k', linestyle = '--')
    plt.axvline(line.w2, color = 'k', linestyle = '--')
    coeff, var_matrix = curve_fit(gauss, wave, reduced_spectrum, p0=pg, method = 'lm')
    gau = coeff[0]*np.exp(-(wave-coeff[2])**2/(2.*coeff[3]**2)) + coeff[1]*np.exp(-(wave-(coeff[2]*line.w2/line.w1))**2/(2.*coeff[3]**2))    
    plt.plot(wave[centre_region], gau[centre_region], label='Gaussian fit', linestyle = '--', color = 'orange')
    coeff, var_matrix = curve_fit(lorentz, wave, reduced_spectrum, p0=pl, method = 'lm')
    lor = coeff[0]*((0.5*coeff[3])/((wave-coeff[2])**2+(0.5*coeff[3])**2)) + coeff[1]*((0.5*coeff[3])/((wave-(coeff[2]*line.w2/line.w1))**2+(0.5*coeff[3])**2))        
    plt.plot(wave[centre_region], lor[centre_region], label='Lorentzian fit', linestyle = '--', color = 'green')
    coeff, var_matrix = curve_fit(voigt, wave, reduced_spectrum, p0=pv, method = 'lm')
    voi = coeff[0] * np.real(wofz((wave - coeff[2] + 1j*coeff[4])/coeff[3]/np.sqrt(2))) / coeff[3] /np.sqrt(2*np.pi) + (coeff[1] * np.real(wofz((wave - (coeff[2]*line.w2/line.w1) + 1j*coeff[4])/coeff[3]/np.sqrt(2))) / coeff[3] /np.sqrt(2*np.pi))
    plt.plot(wave[centre_region], voi[centre_region], label='Voigt fit', linestyle = '-', color = 'red')
    plt.plot(wave, flux)
    plt.plot(wave[continuum], fitline[continuum], color = 'black')            
    #plt.xlim((x[continuum[0]-10],x[continuum[-1]+10]))
    plt.xlim(wave[centre_region[0]], wave[centre_region[-1]])
    y_temp = np.max(flux[centre_region])
    plt.ylim([-0.2*y_temp, 1.5*y_temp])
    plt.title('{}'.format(line.name))
    plt.legend()
    if (len(path)!=0):
        plt.savefig('{}/{}_overplot.png'.format(path, line.name), format='png', dpi = 1200)
        plt.close()
    else:
        plt.show()

overplot_doublet(OIId, wave, high, path = '../Output/4. Plotting/Figures 4.4/high')
overplot_doublet(OIId, wave, mid, path = '../Output/4. Plotting/Figures 4.4/mid')
overplot_doublet(OIId, wave, low, path = '../Output/4. Plotting/Figures 4.4/low')
overplot_doublet(SIId, wave, high, path = '../Output/4. Plotting/Figures 4.4/high')
overplot_doublet(SIId, wave, mid, path = '../Output/4. Plotting/Figures 4.4/mid')
overplot_doublet(SIId, wave, low, path = '../Output/4. Plotting/Figures 4.4/low')

#   Funtions for fittng NII Ha part of the spectrum

def overplot_NII_Ha(NII, Ha, wave, flux, path=''):
    NII_other = 6549.86
    continuum = np.where(((wave > NII.left[0]) & (wave < NII.left[1])) | ((wave > NII.right[0]) & (wave < NII.right[1])))[0]
    coeff = np.polyfit(wave[continuum], flux[continuum], 1)
    fitline = wave*coeff[0] + coeff[1]
    reduced_spectrum = flux - fitline
    centre = [Ha.w-(Ha.sigma(wave,flux)*15), Ha.w+(Ha.sigma(wave,flux)*15)]
    centre_region = np.where((wave >= centre[0]) & (wave <= centre[1]))[0]
    continuum = np.where(((wave > NII.left[0]) & (wave < NII.left[1])) | ((wave > NII.right[0]) & (wave < NII.right[1])))[0]
    def gauss(x, *p):
        NII_A, Ha_A, NII_mean, Ha_mean, NII_sigma, Ha_sigma = p
        return NII_A*np.exp(-(x-NII_mean)**2/(2.*NII_sigma**2)) + 0.3256*NII_A*np.exp(-(x-(NII_mean*NII_other/NII.w))**2/(2.*NII_sigma**2)) + Ha_A*np.exp(-(x-Ha_mean)**2/(2.*Ha_sigma**2))
    def lorentz(x, *p):
        NII_A, Ha_A, NII_mean, Ha_mean, NII_gamma, Ha_gamma = p
        return NII_A*((0.5*NII_gamma)/((x-NII_mean)**2+(0.5*NII_gamma)**2)) + 0.3256*NII_A*((0.5*NII_gamma)/((x-(NII_mean*NII_other/NII.w))**2+(0.5*NII_gamma)**2)) + Ha_A*((0.5*Ha_gamma)/((x-Ha_mean)**2+(0.5*Ha_gamma)**2))
    def voigt(x, *p):
        NII_A, Ha_A, NII_mean, Ha_mean, NII_sigma, Ha_sigma, NII_gamma, Ha_gamma = p
        return NII_A*np.real(wofz((x-NII_mean+1j*NII_gamma)/NII_sigma/np.sqrt(2)))/NII_sigma/np.sqrt(2*np.pi) + 0.3256*NII_A*np.real(wofz((x-(NII_mean*NII_other/NII.w)+1j*NII_gamma)/NII_sigma/np.sqrt(2)))/NII_sigma/np.sqrt(2*np.pi) + Ha_A*np.real(wofz((x-(Ha_mean)+1j*Ha_gamma)/Ha_sigma/np.sqrt(2)))/Ha_sigma/np.sqrt(2*np.pi)
    pg = [1, 1, NII.w, Ha.w, 1, 1]
    pl = [1, 1, NII.w, Ha.w, 2.5, 2.5]
    pv = [1, 1, NII.w, Ha.w, 1, 1, 1, 1]
    plt.figure(figsize=(7,5))
    plt.axvspan(NII.left[0], NII.left[1], color = 'red', alpha=0.1, linestyle = '-', label = 'continuum')
    plt.axvspan(NII.right[0], NII.right[1], color = 'red', alpha=0.1, linestyle = '-')
    plt.axvline(NII.w, color = 'k', linestyle = '--')
    plt.axvline(NII_other, color = 'k', linestyle = '--')
    plt.axvline(Ha.w, color = 'k', linestyle = '--')
    coeff, var_matrix = curve_fit(gauss, wave, reduced_spectrum, p0=pg, method = 'lm')
    gau = coeff[0]*np.exp(-(wave-coeff[2])**2/(2.*coeff[4]**2)) + 0.3256*coeff[0]*np.exp(-(wave-(coeff[2]*NII_other/NII.w))**2/(2.*coeff[4]**2)) + coeff[1]*np.exp(-(wave-coeff[3])**2/(2.*coeff[5]**2))    
    plt.plot(wave[centre_region], gau[centre_region], label='Gaussian fit', linestyle = '--', color = 'orange')
    coeff, var_matrix = curve_fit(lorentz, wave, reduced_spectrum, p0=pl, method = 'lm')
    lor = coeff[0]*((0.5*coeff[4])/((wave-coeff[2])**2+(0.5*coeff[4])**2)) + 0.3256*coeff[0]*((0.5*coeff[4])/((wave-(coeff[2]*NII_other/NII.w))**2+(0.5*coeff[4])**2)) + coeff[1]*((0.5*coeff[5])/((wave-coeff[3])**2+(0.5*coeff[5])**2))   
    plt.plot(wave[centre_region], lor[centre_region], label='Lorentzian fit', linestyle = '--', color = 'green')
    coeff, var_matrix = curve_fit(voigt, wave, reduced_spectrum, p0=pv, method = 'lm')
    voi = coeff[0]*np.real(wofz((wave-coeff[2]+1j*coeff[6])/coeff[4]/np.sqrt(2)))/coeff[4]/np.sqrt(2*np.pi) + 0.3256*coeff[0]*np.real(wofz((wave-(coeff[2]*NII_other/NII.w)+1j*coeff[6])/coeff[4]/np.sqrt(2)))/coeff[4]/np.sqrt(2*np.pi) + coeff[1]*np.real(wofz((wave-(coeff[3])+1j*coeff[7])/coeff[5]/np.sqrt(2)))/coeff[5]/np.sqrt(2*np.pi)
    plt.plot(wave[centre_region], voi[centre_region], label='Voigt fit', linestyle = '-', color = 'red')
    plt.plot(wave, flux)
    plt.plot(wave[continuum], fitline[continuum], color = 'black')            
    #plt.xlim((x[continuum[0]-10],x[continuum[-1]+10]))
    plt.xlim(wave[centre_region[0]], wave[centre_region[-1]])
    y_temp = np.max(flux[centre_region])
    plt.ylim([-0.2*y_temp, 1.5*y_temp])
    plt.title('NII_Ha')
    plt.legend()
    if (len(path)!=0):
        plt.savefig('{}/NII_Ha_overplot.png'.format(path), format='png', dpi = 1200)
        plt.close()
    else:
        plt.show()

overplot_NII_Ha(NII, H_alpha, wave, high, path = '../Output/4. Plotting/Figures 4.4/high')
overplot_NII_Ha(NII, H_alpha, wave, mid, path = '../Output/4. Plotting/Figures 4.4/mid')
overplot_NII_Ha(NII, H_alpha, wave, low, path = '../Output/4. Plotting/Figures 4.4/low')


#%%
#   Calculate the voigt flux and the 2 FWHM fluxes of the strong lines

def get_p(line, wave, flux):
    continuum = np.where(((wave > line.left[0]) & (wave < line.left[1])) | ((wave > line.right[0]) & (wave < line.right[1])))[0]
    coeff = np.polyfit(wave[continuum], flux[continuum], 1)
    fitline = wave*coeff[0] + coeff[1]
    reduced_spectrum = flux - fitline
    continuum = np.where(((wave > line.left[0]) & (wave < line.left[1])) | ((wave > line.right[0]) & (wave < line.right[1])))[0]
    if ((line.name == 'H alpha') | (line.name == 'NII 6583')):
         NII_other = 6549.86
         def voigt(x, *p):
             NII_A, Ha_A, NII_mean, Ha_mean, NII_sigma, Ha_sigma, NII_gamma, Ha_gamma = p
             return NII_A*np.real(wofz((x-NII_mean+1j*NII_gamma)/NII_sigma/np.sqrt(2)))/NII_sigma/np.sqrt(2*np.pi) + 0.3256*NII_A*np.real(wofz((x-(NII_mean*NII_other/NII.w)+1j*NII_gamma)/NII_sigma/np.sqrt(2)))/NII_sigma/np.sqrt(2*np.pi) + Ha_A*np.real(wofz((x-(Ha_mean)+1j*Ha_gamma)/Ha_sigma/np.sqrt(2)))/Ha_sigma/np.sqrt(2*np.pi)
         pv = [1, 1, NII.w, H_alpha.w, 1, 1, 1, 1]
    elif (type(line) == lier.singlet):
        def voigt(x, *p):
            A, mean, sigma, gamma = p
            return A * np.real(wofz((x - mean + 1j*gamma)/sigma/np.sqrt(2))) / sigma /np.sqrt(2*np.pi)
        pv = [1, line.w, 1, 1]
    else:
        def voigt(x, *p):
            A, B, mean, sigma, gamma= p
            return A * np.real(wofz((x - mean + 1j*gamma)/sigma/np.sqrt(2))) / sigma /np.sqrt(2*np.pi) + B * np.real(wofz((x - (mean*line.w2/line.w1) + 1j*gamma)/sigma/np.sqrt(2))) / sigma /np.sqrt(2*np.pi)
        pv = [1, 1, line.w1, 1, 1]    
    coeff, var_matrix = curve_fit(voigt, wave, reduced_spectrum, p0=pv, method = 'lm')
    return coeff

def voigt_flux(line, wave, flux):
    coeff = get_p(line, wave, flux)
    if (line.name == 'H alpha'):
         return coeff[1]
    elif (line.name == 'NII 6583'):
         return coeff[0]
    elif (type(line) == lier.singlet):
         return coeff[0]
    else:
         return (coeff[0] + coeff[1])

        
def FWHM_temp(line, flux):
    coeff = get_p(line, wave, flux)
    t = 2*np.sqrt(2*np.log(2)) 
    if (line.name == 'H alpha'):
        fg = t*coeff[5]
        fl = 2*coeff[7]
    elif (line.name == 'NII 6583'):
        fg = t*coeff[4]
        fl = 2*coeff[6]
    elif (type(line)==lier.singlet):
        fg = t*coeff[2] # The FWHM of the gaussian component
        fl = 2*coeff[3] # The FWHM of the lorenzian component
    else:
        fg = t*coeff[3] # The FWHM of the gaussian component
        fl = 2*coeff[4] # The FWHM of the lorenzian component
    return 0.5346*fl + np.sqrt(0.2166*fl**2 + fg**2)
    

def spaxel_data(bin_data, bin_control_data):
    result = []
    for i in range(len(bin_data)-1):
        for j in range(len(bin_data[i].data)):
            result.append(bin_data[i].data[j])
        for j in range(len(bin_control_data[i].data)):
            result.append(bin_control_data[i].data[j])
    return np.array(result)

def LSF(line, spaxels):
    if (line.name == 'OII 3727, 3729'):
        kms = lier.round_down(np.median(spaxel_data_table[1].data['OIId-3727_LSF'][spaxels]),3)
    elif (line.name == 'H beta'):
        kms = lier.round_down(np.median(spaxel_data_table[1].data['H_beta_LSF'][spaxels]),3)
    elif (line.name == 'OIII 5007'):
        kms = lier.round_down(np.median(spaxel_data_table[1].data['OIII-5007_LSF'][spaxels]),3)
    elif (line.name == 'OI 6300'):
        kms = lier.round_down(np.median(spaxel_data_table[1].data['OI-6300_LSF'][spaxels]),3)
    elif (line.name == 'H alpha'):
        kms = lier.round_down(np.median(spaxel_data_table[1].data['H_alpha_LSF'][spaxels]),3)
    elif (line.name == 'NII 6583'):
        kms = lier.round_down(np.median(spaxel_data_table[1].data['NII-6583_LSF'][spaxels]),3)
    elif (line.name == 'SIId 6716, 6731'):
        kms = lier.round_down(np.median(spaxel_data_table[1].data['SIId-6716_LSF'][spaxels]),3)
    elif (line.name == 'SII 4068, 4076'):
        kms = lier.round_down(np.median(spaxel_data_table[1].data['SIId-4068_LSF'][spaxels]),3)
    elif (line.name == 'OIII 4363'):
        kms = lier.round_down(np.median(spaxel_data_table[1].data['OIII-4363_LSF'][spaxels]),3)
    elif (line.name == 'NII 5755'):
        kms = lier.round_down(np.median(spaxel_data_table[1].data['NII-5755_LSF'][spaxels]),3)
    elif (line.name == 'OII 7320, 7330'):
        kms = lier.round_down(np.median(spaxel_data_table[1].data['OIId-7320_LSF'][spaxels]),3)
    elif ((line.name == 'H zeta') | (line.name == 'H delta') | (line.name == 'H gamma') | (line.name == 'H epsilon')):
        kms = lier.round_down(np.median(spaxel_data_table[1].data['SIId-4068_LSF'][spaxels]),3)
    else:
        return 0
    if (type(line)==lier.singlet):
        return 2.35*kms*line.w/c_vel
    else:
        return 2.35*kms*((line.w1+line.w2)/2)/c_vel

high_LSF = []
mid_LSF = []
low_LSF = []
for i in strong_lines:
    high_LSF.append(LSF(i, spaxel_data(bin_1, bin_1_control)))    
    mid_LSF.append(LSF(i, spaxel_data(bin_2, bin_2_control)))    
    low_LSF.append(LSF(i, spaxel_data(bin_3, bin_3_control)))    

def velocity(line, n):
    if (type(line)==lier.singlet):
        return (n/line.w)*c_vel
    else:
        return (n/((line.w1+line.w2)/2))*c_vel
    
table = [[],[],[],[]]
table[0].append('Bin')
table[1].append('High')
table[2].append('Mid')
table[3].append('Low')

for i in range(len(strong_lines)):
    table[0].append(strong_lines[i].name)
    table[1].append(round(velocity(strong_lines[i], np.sqrt(FWHM_temp(strong_lines[i], high)**2-high_LSF[i]**2)),3))
    table[2].append(round(velocity(strong_lines[i], np.sqrt(FWHM_temp(strong_lines[i], mid)**2-mid_LSF[i]**2)),3))
    table[3].append(round(velocity(strong_lines[i], np.sqrt(FWHM_temp(strong_lines[i], low)**2-low_LSF[i]**2)),3))
print(tabulate(table, headers = 'firstrow'))


OI_velo = []
OII_velo = []
OIII_velo = []
H_velo = [] 
N_velo = []
S_velo = [] 

for i in range(1,4):
    OII_velo.append(table[i][1])
    OIII_velo.append(table[i][3])
    OI_velo.append(table[i][4])
    H_velo.append((table[i][2]+table[i][5])/2)
    N_velo.append(table[i][6])
    S_velo.append(table[i][7])

def FWHM(line, flux):
    if (line.type == 'strong'):
        coeff = get_p(line, wave, flux)
        t = 2*np.sqrt(2*np.log(2)) 
        if (line.name == 'H alpha'):
            fg = t*coeff[5]
            fl = 2*coeff[7]
        elif (line.name == 'NII 6583'):
            fg = t*coeff[4]
            fl = 2*coeff[6]
        elif (type(line)==lier.singlet):
            fg = t*coeff[2] # The FWHM of the gaussian component
            fl = 2*coeff[3] # The FWHM of the lorenzian component
        else:
            fg = t*coeff[3] # The FWHM of the gaussian component
            fl = 2*coeff[4] # The FWHM of the lorenzian component
        return 0.5346*fl + np.sqrt(0.2166*fl**2 + fg**2)
    
    elif (line.type == 'weak'):
        m = 0 
        if (np.equal(flux, high).all()):
            m = 0
            b = bin_1
            bc = bin_1_control
        elif (np.equal(flux, mid).all()):
            m = 1
            b = bin_2
            bc = bin_2_control
        elif (np.equal(flux, low).all()):
            m = 2
            b = bin_3
            bc = bin_3_control
        else:
            m = 3
        w = line.name[0]
        vel = 0
        if (w == 'H'):
            vel = H_velo[m]
        elif (w == 'N'):
            vel = N_velo[m]
        elif (w == 'S'):
            vel = S_velo[m]
        elif (w == 'O'):
            w2 = line.name[0:4]
            if (w2 == 'OIII'):
                vel = OIII_velo[m]
            elif (w2 == 'OII '):
                vel = OII_velo[m]
        if (type(line) == lier.doublet):
            return np.sqrt(((vel*(line.w1+line.w2)/2/c_vel)**2) + (LSF(line,spaxel_data(b, bc))**2))
        else:
            return np.sqrt(((vel*line.w/c_vel)**2)+(LSF(line,spaxel_data(b, bc))**2))

#%%

def summed_flux(line, wave, flux, FWHM, graph = False, path=''):
        #   1. Get the actual wavelength and the SD of the emission lines
        if (type(line) == lier.singlet):
            centre = [line.w - 1.5*FWHM, line.w + 1.5*FWHM]
        else:
            centre = [line.w1 - 1.5*FWHM, line.w2 + 1.5*FWHM]
        
        #   2. Get the regions of the emission line on the wave array
        centre_region = np.where(((wave >= centre[0]) & (wave <= centre[1])))[0]
        continuum = np.where(((wave > line.left[0]) & (wave < line.left[1])) | ((wave > line.right[0]) & (wave < line.right[1])))[0]
        
        #   3. Construct a fitline from continuum and subtract it to get the reduced spectrum
        coeff = np.polyfit(wave[continuum], flux[continuum], 1)
        fitline = wave*coeff[0] + coeff[1]
        reduced_spectrum = flux - fitline
        
        #   4. Integrate the centre region (default = 3 sigma) and get the summed flux
        sflux = np.trapz(reduced_spectrum[centre_region], wave[centre_region])
        
        #   5. Show the regions of integration
        if (graph == True):
            plt.figure(figsize=(7,5))
            plt.axvspan(wave[centre_region][0], wave[centre_region][-1], color = 'grey', linestyle = '-', label = 'integrated region')
            plt.axvspan(line.left[0], line.left[1], color = 'red', alpha=0.1, linestyle = '-', label = 'continuum')
            plt.axvspan(line.right[0], line.right[1], color = 'red', alpha=0.1, linestyle = '-')
            if (line.type == 'weak'):
                if (type(line) == lier.singlet):
                    plt.axvline(line.w, color = 'k', linestyle = '--')
                else:    
                    plt.axvline(line.w1, color = 'k', linestyle = '--')
                    plt.axvline(line.w2, color = 'k', linestyle = '--')    
            else:
                coeff = get_p(line, wave, flux)
                if ((line.name == 'H alpha') | (line.name == 'NII 6583')):
                    plt.axvline(line.w, color = 'k', linestyle = '--')
                    NII_other = 6549.86
                    voi = coeff[0]*np.real(wofz((wave-coeff[2]+1j*coeff[6])/coeff[4]/np.sqrt(2)))/coeff[4]/np.sqrt(2*np.pi) + 0.3256*coeff[0]*np.real(wofz((wave-(coeff[2]*NII_other/NII.w)+1j*coeff[6])/coeff[4]/np.sqrt(2)))/coeff[4]/np.sqrt(2*np.pi) + coeff[1]*np.real(wofz((wave-(coeff[3])+1j*coeff[7])/coeff[5]/np.sqrt(2)))/coeff[5]/np.sqrt(2*np.pi)
                    n = coeff[0]*np.real(wofz((wave-coeff[2]+1j*coeff[6])/coeff[4]/np.sqrt(2)))/coeff[4]/np.sqrt(2*np.pi) + 0.3256*coeff[0]*np.real(wofz((wave-(coeff[2]*NII_other/NII.w)+1j*coeff[6])/coeff[4]/np.sqrt(2)))/coeff[4]/np.sqrt(2*np.pi)
                    ha = coeff[1]*np.real(wofz((wave-(coeff[3])+1j*coeff[7])/coeff[5]/np.sqrt(2)))/coeff[5]/np.sqrt(2*np.pi)
                    plt.plot(wave, n, color = 'purple', label = 'NII')
                    plt.plot(wave, ha, color = 'green', label = 'Ha')
                elif (type(line) == lier.singlet):
                    plt.axvline(line.w, color = 'k', linestyle = '--')
                    voi = coeff[0] * np.real(wofz((wave - coeff[1] + 1j*coeff[3])/coeff[2]/np.sqrt(2))) / coeff[2] /np.sqrt(2*np.pi)
                else:    
                    plt.axvline(line.w1, color = 'k', linestyle = '--')
                    plt.axvline(line.w2, color = 'k', linestyle = '--')
                    voi = coeff[0] * np.real(wofz((wave - coeff[2] + 1j*coeff[4])/coeff[3]/np.sqrt(2))) / coeff[3] /np.sqrt(2*np.pi) + (coeff[1] * np.real(wofz((wave - (coeff[2]*line.w2/line.w1) + 1j*coeff[4])/coeff[3]/np.sqrt(2))) / coeff[3] /np.sqrt(2*np.pi))
                    first = coeff[0] * np.real(wofz((wave - coeff[2] + 1j*coeff[4])/coeff[3]/np.sqrt(2))) / coeff[3] /np.sqrt(2*np.pi)
                    second = coeff[1] * np.real(wofz((wave - (coeff[2]*line.w2/line.w1) + 1j*coeff[4])/coeff[3]/np.sqrt(2))) / coeff[3] /np.sqrt(2*np.pi)
                    plt.plot(wave, first, color = 'purple')
                    plt.plot(wave, second, color = 'green')
                plt.plot(wave, voi, label='Voigt fit', linestyle = '-', color = 'red')
            plt.plot(wave, reduced_spectrum)
            plt.xlim((wave[continuum[0]-10],wave[continuum[-1]+10]))
            y_temp = np.max(flux[centre_region])
            plt.ylim([-0.2*y_temp, 1.5*y_temp])
            plt.legend()
            plt.title('Summed flux of {}'.format(line.name))
            if (len(path)!=0):
                plt.savefig('{}/{}_summed.png'.format(path, line.name), dpi = 600)
                plt.close()
            else:
                plt.show()
        return sflux

def table_flux(line_array, wave, flux, name):    
    first_row = [name]
    second_row = ['Summed']
    third_row = ['Voigt']
    fourth_row = ['Flux loss (%)']
    for l in line_array:
        first_row.append(l.name)
        s = round(summed_flux(l, wave, flux, FWHM(l, flux)),3)
        v = round(voigt_flux(l, wave, flux),3)
        second_row.append(s)
        third_row.append(v)
        fourth_row.append(round(((v - s)/v*100), 3))
    return [first_row, second_row, third_row, fourth_row]

print('\n')
print(tabulate(table_flux(strong_lines, wave, high, 'high'), headers='firstrow'), '\n')
print(tabulate(table_flux(strong_lines, wave, mid, 'mid'), headers='firstrow'), '\n')
print(tabulate(table_flux(strong_lines, wave, low, 'low'), headers='firstrow'), '\n')

for l in strong_lines:
    summed_flux(l, wave, high, FWHM(l, high), graph = True, path = '../Output/4. Plotting/Figures 4.5/high')
    summed_flux(l, wave, mid, FWHM(l, mid), graph = True, path = '../Output/4. Plotting/Figures 4.5/mid')
    summed_flux(l, wave, low, FWHM(l, low), graph = True, path = '../Output/4. Plotting/Figures 4.5/low')


#%%

for l in weak_lines:
    summed_flux(l, wave, high, FWHM(l, high), graph = True, path = '../Output/4. Plotting/Figures 4.6/high')
    summed_flux(l, wave, mid, FWHM(l, mid), graph = True, path = '../Output/4. Plotting/Figures 4.6/mid')
    summed_flux(l, wave, low, FWHM(l, low), graph = True, path = '../Output/4. Plotting/Figures 4.6/low')



#%%
#   Function to calculate the log ratio between different lines

#   Input:
    #   x: The x axis of plot (wave array)
    #   y: The y axis of plot (residual spectrum)    
    #   line_x: The dividend line
    #   line_y: The divisor line
    #   Av = The extinction magnitude
#   Output:
    #   output: The log ratio of the emission line
def log_ratio(wave, flux, line_x, line_y, Av = 0):
    if (Av != 0):
        x_index = np.where(line_x.w < wave)[0][0]
        x_flux = summed_flux(line_x, wave, flux, FWHM(line_x, flux)) / (10**(-0.4*ext.fitzpatrick99(wave, Av)))[x_index] 
        y_index = np.where(line_y.w < wave)[0][0]
        y_flux = summed_flux(line_y, wave, flux, FWHM(line_y, flux)) / (10**(-0.4*ext.fitzpatrick99(wave, Av)))[y_index]
        ratio = x_flux/y_flux
    else:
        ratio = summed_flux(line_x, wave, flux, FWHM(line_x, flux)) / summed_flux(line_y, wave, flux, FWHM(line_y, flux))    
    return np.log10(ratio)
    
def variance(wave, flux, var, line):
    temp = FWHM(line, flux)
    if (type(line) == lier.singlet):
        centre = [line.w - 1.5*temp, line.w + 1.5*temp]
    else:
        centre = [line.w1 - 1.5*temp, line.w2 + 1.5*temp]
    centre_region = np.where(((wave >= centre[0]) & (wave <= centre[1])))[0]
    delta_x = np.median(wave[centre_region] - wave[centre_region-1])
    sflux = np.trapz(var[centre_region], wave[centre_region]) * delta_x
    return sflux

def table_sigma(line_array, wave, flux, var, name):    
    first_row = [name]
    second_row = ['Summed']
    third_row = ['SD']
    for l in line_array:
        first_row.append(l.name)
        s = round(summed_flux(l, wave, flux, FWHM(l, flux)),3)
        second_row.append(s)
        sd = round(np.sqrt(variance(wave, flux, var, l)),3)
        third_row.append(sd)
    return [first_row, second_row, third_row]

print('\n')
print(tabulate(table_sigma(strong_lines, wave, high, high_var, 'high'), headers='firstrow'), '\n')
print(tabulate(table_sigma(strong_lines, wave, mid, mid_var, 'mid'), headers='firstrow'), '\n')
print(tabulate(table_sigma(strong_lines, wave, low, low_var, 'low'), headers='firstrow'), '\n')
print(tabulate(table_sigma(weak_lines, wave, high, high_var, 'high'), headers='firstrow'), '\n')
print(tabulate(table_sigma(weak_lines, wave, mid, mid_var, 'mid'), headers='firstrow'), '\n')
print(tabulate(table_sigma(weak_lines, wave, low, low_var, 'low'), headers='firstrow'), '\n')




#%%
def sliding_box(wave, flux, line):
    output = []
    temp = 3*FWHM(line, flux)
    l_index = np.where((wave>line.left[0]) & (wave<line.left[1]))[0]
    r_index = np.where((wave>line.right[0]) & (wave<line.right[1]))[0]
    delta = wave[r_index[-1]] - wave[r_index[-2]]
    d_spaxels = int((temp//delta)+1)
    for i in range(len(l_index)-d_spaxels):
        error_sum = np.trapz(flux[l_index[i:i+d_spaxels]], wave[l_index[i:i+d_spaxels]])
        output.append(error_sum)
        
    for j in range(d_spaxels):
        spaxel_left = d_spaxels-j-1
        error_sum_left = np.trapz(flux[l_index[-spaxel_left:]], wave[l_index[-spaxel_left:]])
        error_sum_right = np.trapz(flux[r_index[0:j+1]], wave[r_index[0:j+1]])
        error_sum = error_sum_left + error_sum_right
        output.append(error_sum)
        
    for i in range(len(r_index)-d_spaxels):
        error_sum = np.trapz(flux[r_index[i:i+d_spaxels]], wave[r_index[i:i+d_spaxels]])
        output.append(error_sum)
        
    out = np.array(output)
    rms = np.sqrt(np.mean(out**2))
    return len(output), rms

k = sliding_box(wave, high, H_beta)

def sliding_box_new(wave, flux, line):
    output = []
    temp = 3*FWHM(line, flux)
    l_index = np.where((wave>line.left[0]) & (wave<line.left[1]))[0]
    r_index = np.where((wave>line.right[0]) & (wave<line.right[1]))[0]
    delta = wave[r_index[-1]] - wave[r_index[-2]]
    d_spaxels = int((temp//delta)+1)
    space = 5
    for i in range((len(l_index)-d_spaxels)//space):
        error_sum = np.trapz(flux[l_index[space*i:space*i+d_spaxels]], wave[l_index[space*i:space*i+d_spaxels]])
        output.append(error_sum)
        
    for i in range((len(r_index)-d_spaxels)//space):
        error_sum = np.trapz(flux[r_index[space*i:space*i+d_spaxels]], wave[r_index[space*i:space*i+d_spaxels]])
        output.append(error_sum)
        
    out = np.array(output)
    rms = np.sqrt(np.mean(out**2))
    return len(output), rms

aroural_lines = [SIIdf, OIIIf, NIIf, OIIdf]

def table_box(line_array, wave, flux, name):    
    first_row = [name]
    second_row = ['Summed']
    third_row = ['SD']
    fourth_row = ['# of boxes']
    for l in line_array:
        first_row.append(l.name)
        s = round(summed_flux(l, wave, flux, FWHM(l, flux)),3)
        second_row.append(s)
        temp, sdt = sliding_box(wave, flux, l)
        sd = round(sdt, 3)
        third_row.append(sd)
        fourth_row.append(temp)
    return [first_row, second_row, third_row, fourth_row]
"""
print(tabulate(table_box(strong_lines, wave, high, 'high'), headers='firstrow'), '\n')
print(tabulate(table_box(strong_lines, wave, mid, 'mid'), headers='firstrow'), '\n')
print(tabulate(table_box(strong_lines, wave, low, 'low'), headers='firstrow'), '\n')
print(tabulate(table_box(weak_lines, wave, high, 'high'), headers='firstrow'), '\n')
print(tabulate(table_box(weak_lines, wave, mid, 'mid'), headers='firstrow'), '\n')
print(tabulate(table_box(weak_lines, wave, low, 'low'), headers='firstrow'), '\n')
"""
def table_box_weak(line_array, wave, flux, name):    
    first_row = [name]
    second_row = ['Summed']
    third_row = ['SD_1']
    fourth_row = ['# of boxes']
    fifth_row= ['SD_5']
    sixth_row = ['# of boxes']
    seventh_row = ['Difference']
    for l in line_array:
        first_row.append(l.name)
        
        s = round(summed_flux(l, wave, flux, FWHM(l, flux)),3)
        second_row.append(s)
        temp, sdt = sliding_box(wave, flux, l)
        sd = round(sdt, 3)
        third_row.append(sd)
        fourth_row.append(temp)
        temp5, sdt5 = sliding_box_new(wave, flux, l)
        sd5 = round(sdt5, 3)
        fifth_row.append(sd5)
        sixth_row.append(temp5)
        seventh_row.append(sdt - sdt5)
    return [first_row, second_row, third_row, fourth_row, fifth_row, sixth_row, seventh_row]

print(tabulate(table_box_weak(aroural_lines, wave, high, 'high'), headers='firstrow'), '\n')
print(tabulate(table_box_weak(aroural_lines, wave, mid, 'mid'), headers='firstrow'), '\n')
print(tabulate(table_box_weak(aroural_lines, wave, low, 'low'), headers='firstrow'), '\n')


#%%
def ratio(wave, flux, hline, hb):
    sflux_hline = voigt_flux(hline, wave, flux)
    sflux_hb = voigt_flux(hb, wave, flux)
    return sflux_hline/sflux_hb

hlines = [H_alpha, H_beta, H_gamma, H_delta, H_epsilon, H_zeta]
ref_value = [2.85, 1, 0.469, 0.26, 0.159, 0.105, 0.0786]

def table_h_ratio(line_array, wave, ref):    
    row1 = ['Bin', 'High', 'Mid', 'Low', 'Ref']
    row2 = ['H alpha']
    row3 = ['H beta']
    row4 = ['H gamma']
    row5= ['H delta']
    row6= ['H epsilon']
    row7 = ['H zeta']
    temp = [row2, row3, row4, row5, row6, row7]
    for i in range(len(line_array)):
        temp[i].append(round(ratio(wave, high, line_array[i], line_array[1]),3))
        temp[i].append(round(ratio(wave, mid, line_array[i], line_array[1]),3))
        temp[i].append(round(ratio(wave, low, line_array[i], line_array[1]),3))
        temp[i].append(ref[i])
    return [row1, row2, row3, row4, row5, row6, row7]

print(tabulate(table_h_ratio(hlines, wave, ref_value), headers='firstrow'), '\n')

def cal_Av(wave, flux, hline, hb, value):
    r = ratio(wave, flux, hline, hb)    
    d_A = -2.5*np.log10(r/value)
    Al_Av = ext.fitzpatrick99(np.array([hline.w]),1)[0] - ext.fitzpatrick99(np.array([hb.w]),1)[0]
    return d_A/Al_Av

def table_Av_ratio(line_array, wave):
    row1 = ['Bin', 'High', 'Mid', 'Low']
    row2 = ['H alpha']
    row3 = ['H beta']
    row4 = ['H gamma']
    row5= ['H delta']
    row6= ['H epsilon']
    row7 = ['H zeta']
    temp = [row2, row3, row4, row5, row6, row7]
    for i in range(len(line_array)):
        temp[i].append(round(cal_Av(wave, high, line_array[i], line_array[1], ref_value[i]),3))
        temp[i].append(round(cal_Av(wave, mid, line_array[i], line_array[1], ref_value[i]),3))
        temp[i].append(round(cal_Av(wave, low, line_array[i], line_array[1], ref_value[i]),3))
    return [row1, row2, row3, row4, row5, row6, row7]

print(tabulate(table_Av_ratio(hlines, wave), headers='firstrow'), '\n')

#%%

def get_value(wave, Av, line, ref_value):
    x = 10**(-0.4*ext.fitzpatrick99(wave,Av))
    ind_b = np.where(wave > H_beta.w)[0][0]
    ind = np.where(wave > line.w)[0][0]
    return ref_value*x[ind]/x[ind_b]

ha = voigt_flux(H_alpha, wave, high)    
hb = voigt_flux(H_beta, wave, high)

def balmer_all(wave, high, mid, low):    
    fig, [axes1, axes2, axes3] = plt.subplots(3 , 1 ,figsize = [7, 18])
    def balmer_decrement(wave, flux, label, axes):
        Av = cal_Av(wave, flux, H_alpha, H_beta, 2.85)
        h_ind = []
        real = []
        error = []
        extinct = 10**(-0.4*ext.fitzpatrick99(wave, Av))
        ind_b = np.where(wave > H_beta.w)[0][0]
        hb_ext = extinct[ind_b]
        for i in range(len(hlines)):
            h_ind.append(np.where(wave > hlines[i].w)[0][0])
            if (hlines[i].name == 'H epsilon'):
                real.append((voigt_flux(H_epsilon, wave, flux) - 0.31*voigt_flux(NeIII, wave, flux))/voigt_flux(H_beta, wave, flux))
            else:
                real.append(ratio(wave, flux, hlines[i], H_beta))
            if (hlines[i].name == 'H beta'):
                error.append(sliding_box(wave, flux, H_beta)[1])
            elif (hlines[i].name == 'H epsilon'):
                ne_err = sliding_box(wave, flux, NeIII)[1] * 0.31
                h_err = sliding_box(wave, flux, H_epsilon)[1] + ne_err
                h_flux = voigt_flux(H_epsilon, wave, flux) - 0.31 * voigt_flux(NeIII, wave, flux)
                error.append(np.sqrt((h_err/h_flux)**2 + (sliding_box(wave, flux, H_beta)[1]/voigt_flux(H_beta, wave, flux))**2)*real[-1]/ref_value[i])
            else:
                error.append(np.sqrt((sliding_box(wave, flux, hlines[i])[1]/voigt_flux(hlines[i], wave, flux))**2 + (sliding_box(wave, flux, H_beta)[1]/voigt_flux(H_beta, wave, flux))**2)*real[-1]/ref_value[i])
        axes.plot(wave, extinct/hb_ext, color = 'k')
        axes.axhline(y = 1, color = 'k', linestyle = '--')
        for i in range(len(h_ind)):
            axes.errorbar(wave[h_ind[i]], real[i]/ref_value[i], yerr = error[i], fmt = 'r.')
        axes.set_xlim(3500, 7000)
        axes.set_ylim(0.3,1.7)
        axes.set_title(label)
    
    balmer_decrement(wave, high, 'High', axes1)
    balmer_decrement(wave, mid, 'Mid', axes2)
    balmer_decrement(wave, low, 'Low', axes3)
    fig.show()

balmer_all(wave, high, mid ,low)

#%%
grid = fits.open('../sh_liner_metal.fits')

Ha = grid[1].data['HA']
Hb = grid[1].data['HB']
N2 = grid[1].data['N2B']
NT2 = grid[1].data['NT2']
O1 = grid[1].data['O1']
O2 = grid[1].data['O2']
OT2 = grid[1].data['OT2']
O3 = grid[1].data['O3B']
OT3 = grid[1].data['OT3']
S2 = grid[1].data['S2']
ST2 = grid[1].data['ST2']
M = grid[1].data['METAL']
I = grid[1].data['IONIZATION']

H_alpha.old = [439.2, 393.6, 390.6]
H_beta.old = [100, 100, 100]
OIII.old = [229, 200.9, 191.7]
NII.old = [687.7, 459.4, 312.1]
SIId.old = [466.9, 394.8, 363.1]
OIId.old = [583.6, 620.9, 686]
OIIIf.old = [10.9, 4.1, 2.9]
NIIf.old = [7.7, 9.3, 8.2]
SIIdf.old = [24.3, 25.4, 19.7]
OIIdf.old = [24.6, 20.6, 14.0]
OI.old = [100000, 100000, 100000]

def plot_arrows(wave, flux, x1, x2, y1, y2):
    non_detected = [NIIf, OIIIf]
    output = [0, 0]
    if (x1 in non_detected):
        output[0] = -1
    elif (x2 in non_detected):
        output[0] = 1
    if (y1 in non_detected):
        output[1] = -1
    elif (y2 in non_detected):
        output[1] = 1
    return output
        
#%%
#   Plot the log line ratio with each other (Figures 4.6)
def extinction_vector(wave, line1, line2):
    e = 10**(ext.fitzpatrick99(wave, 1, 3.1)/-2.512)
    def w(line):
        if (type(line) == lier.doublet):
            return (line.w1+line.w2)/2
        else:
            return line.w
    pos1 = np.where(wave>w(line1))[0][0]
    pos2 = np.where(wave>w(line2))[0][0]
    frac = e[pos1]/e[pos2]
    return np.log10(frac)

#%%
#   Plot SOT in another paper of Renbin in 2018

def plot_log_line_ratio(x1, x2, y1, y2, x1g, x2g, y1g, y2g, xlim, ylim):
    
    fig, axes = plt.subplots(figsize=(7,7))

    #   Plot the error bars of the three points (If y = a/c, dy^2 = ((da/a)^2 + (dc/c)^2) y^2), d (log10x) = 1/ln10 * dx / x 
    def plot_error_bars(wave, flux, var, x1, x2, y1, y2, form, form_corr, ecolor, label):
        ln10 = np.log(10)
        
        x1_flux = summed_flux(x1, wave, flux, FWHM(x1, flux))
        x2_flux = summed_flux(x2, wave, flux, FWHM(x2, flux))
        x1_sd = sliding_box(wave, flux, x1)[1]
        x2_sd = sliding_box(wave, flux, x2)[1]
        x_frac_err = np.sqrt((x1_sd/x1_flux)**2 + (x2_sd/x2_flux)**2)
        log_x_err = x_frac_err / ln10
        x = log_ratio(wave, flux, x1, x2)
        y1_flux = summed_flux(y1, wave, flux, FWHM(y1, flux))
        y2_flux = summed_flux(y2, wave, flux, FWHM(y2, flux))
        y1_sd = sliding_box(wave, flux, y1)[1]
        y2_sd = sliding_box(wave, flux, y2)[1]
        y_frac_err = np.sqrt((y1_sd/y1_flux)**2 + (y2_sd/y2_flux)**2)
        log_y_err = y_frac_err / ln10
        y = log_ratio(wave, flux, y1, y2)
       
        arrows = plot_arrows(wave, flux, x1, x2, y1, y2)
        
        Av = cal_Av(wave, flux, H_alpha, H_beta, 2.85)
        ext_arr = ext.fitzpatrick99(wave, Av)
        
        if (arrows[0] == -1):
            x1_flux = 2 * sliding_box(wave, flux, x1)[1]
            x = np.log10(x1_flux/summed_flux(x2, wave, flux, FWHM(x2, flux)))
        elif (arrows[0] == 1):
            x2_flux = 2 * sliding_box(wave, flux, x2)[1]
            x = np.log10(summed_flux(x1, wave, flux, FWHM(x1, flux))/x2_flux)
        x1_index = np.where(x1.w < wave)[0][0]
        x2_index = np.where(x2.w < wave)[0][0]
        x_corr = x - 0.4 * (ext_arr[x2_index] - ext_arr[x1_index])
        
        if (arrows[1] == -1):
            y1_flux = 2 * sliding_box(wave, flux, y1)[1]
            y = np.log10(y1_flux/summed_flux(y2, wave, flux, FWHM(y2, flux)))
        elif (arrows[1] == 1):
            y2_flux = 2 * sliding_box(wave, flux, y2)[1]
            y = np.log10(summed_flux(y1, wave, flux, FWHM(y1, flux))/y2_flux)
        y1_index = np.where(y1.w < wave)[0][0]
        y2_index = np.where(y2.w < wave)[0][0]
        y_corr = y - 0.4 * (ext_arr[y2_index] - ext_arr[y1_index])
        
        if ((arrows[0] != 0) | (arrows[1] != 0)):
            axes.plot(x, y, form, label = label)
            axes.plot(x_corr, y_corr, form_corr, label = label + '_corr')
            
            axes.arrow(x, y, arrows[0]*0.5, arrows[1]*0.5, color = ecolor, head_width = 0.03, head_length = 0.02)        
            axes.arrow(x_corr, y_corr, arrows[0]*0.5, arrows[1]*0.5, color = ecolor, head_width = 0.03, head_length = 0.02)            
            
            if (arrows[0] == 0):
                axes.errorbar(x, y, xerr = log_x_err, yerr = 0, fmt=form, ecolor = ecolor)
                axes.errorbar(x_corr, y_corr, xerr = log_x_err, yerr = 0, fmt=form, ecolor = ecolor)         
                
            elif (arrows[1] == 0):
                axes.errorbar(x, y, xerr = 0, yerr = log_y_err, fmt=form, ecolor = ecolor)    
                axes.errorbar(x_corr, y_corr, xerr = 0, yerr = log_y_err, fmt=form, ecolor = ecolor)    
        else:
            axes.errorbar(x, y, xerr = log_x_err, yerr = log_y_err, fmt=form, ecolor = ecolor, label = label)
            axes.errorbar(x_corr, y_corr, xerr = log_x_err, yerr = log_y_err, fmt=form_corr, ecolor = ecolor, label = label + '_corr')
        
        #   Plot the data points of 2018
        index = 0
        if (label == 'High'):
            index = 0
        elif (label == 'Mid'):
            index = 1
        elif (label == 'Low'):
            index = 2
        
        x1_old = x1.old[index]
        x2_old = x2.old[index]
        y1_old = y1.old[index]
        y2_old = y2.old[index]
        x_old = np.log10(x1_old/x2_old)
        y_old = np.log10(y1_old/y2_old)
        axes.plot(x_old, y_old, 's', color = ecolor, label = label + '_old')
        
       
    plot_error_bars(wave, high, high_var, x1, x2, y1, y2, '.r','r*' , 'r', 'High')
    plot_error_bars(wave, mid, mid_var, x1, x2, y1, y2, '.b','b*', 'b', 'Mid')
    plot_error_bars(wave, low, low_var, x1, x2, y1, y2, '.k','k*', 'k', 'Low')
    
    #   Plot the grid of photoionization model
    gridx = (np.log10(x1g) - np.log10(x2g))
    gridy = (np.log10(y1g) - np.log10(y2g))
    gx = np.reshape(gridx, [7,6])
    gy = np.reshape(gridy, [7,6])
    Mx = np.reshape(M, [7,6])
    Iy = np.reshape(I, [7,6])
    
    for i in range(gy.shape[0]):
        axes.plot(gx[i],gy[i], '-', color = 'grey', lw = 1.5)    
        axes.annotate(round(Mx[i][0],2), axes.lines[-1].get_xydata()[0])
    for j in range(gy.shape[1]):
        axes.plot(gx[:,j],gy[:,j], '-', color = 'grey', lw = 1.5)    
        axes.annotate(Iy[:,j][0], axes.lines[-1].get_xydata()[-1])
    
    #   Plot the extinction vector
    xpos = xlim[0] + 0.1*(xlim[1] - xlim[0])
    ypos = ylim[0]+ 0.9*(ylim[1] - ylim[0])
    axes.arrow(xpos, ypos, extinction_vector(wave, x1, x2), extinction_vector(wave, y1, y2), length_includes_head=True, color='green', head_width = 0.03, head_length = 0.02)
    
    
    axes.set_xlabel('log {}/{}'.format(x1.name, x2.name))
    axes.set_ylabel('log {}/{}'.format(y1.name, y2.name))
    axes.set_xlim(xlim[0], xlim[1])
    axes.set_ylim(ylim[0], ylim[1])
    
    #   To prevent showing the error bars in the legend
    handles, labels = axes.get_legend_handles_labels()
    new_handles = []
    for h in handles:
        if isinstance(h, container.ErrorbarContainer):
            new_handles.append(h[0])
        else:
            new_handles.append(h)
    axes.legend(new_handles, labels)
    
    fig.savefig('../Output/4. Plotting/Figures 4.7/{}_{}_{}_{}.png'.format(x1.name, x2.name, y1.name, y2.name), format='png')
    plt.close(fig)
    
def plot_log_ratio_SOT(x1, x2, SOT1_up, SOT1_low, SOT2_up, SOT2_low, x1g, x2g, SOT1_upg, SOT1_lowg, SOT2_upg, SOT2_lowg, xlim, ylim):
    fig, axes = plt.subplots(figsize=(7,7))
    def plot_error_bars_SOT(wave, flux, var, x1, x2, SOT1_up, SOT1_low, SOT2_up, SOT2_low, form, form_corr, ecolor, label):
        ln10 = np.log(10)
        x1_flux = summed_flux(x1, wave, flux, FWHM(x1, flux))
        x2_flux = summed_flux(x2, wave, flux, FWHM(x2, flux))
        x1_sd = sliding_box(wave, flux, x1)[1]
        x2_sd = sliding_box(wave, flux, x2)[1]
        x_frac_err = np.sqrt((x1_sd/x1_flux)**2 + (x2_sd/x2_flux)**2)
        log_x_err = x_frac_err / ln10
        x = log_ratio(wave, flux, x1, x2)
        
        SOT1_up_flux = summed_flux(SOT1_up, wave, flux, FWHM(SOT1_up, flux))
        SOT1_low_flux = summed_flux(SOT1_low, wave, flux, FWHM(SOT1_low, flux))
        SOT2_up_flux = summed_flux(SOT2_up, wave, flux, FWHM(SOT2_up, flux))
        SOT2_low_flux = summed_flux(SOT2_low, wave, flux, FWHM(SOT2_low, flux))
        SOT1_up_sd = sliding_box(wave, flux, SOT1_up)[1]
        SOT1_low_sd = sliding_box(wave, flux, SOT1_low)[1]
        SOT2_up_sd = sliding_box(wave, flux, SOT2_up)[1]
        SOT2_low_sd = sliding_box(wave, flux, SOT2_low)[1]
        SOT = log_ratio(wave, flux, SOT1_up, SOT1_low) + 1.3 * log_ratio(wave, flux, SOT2_up, SOT2_low)
        SOT1_frac_err = np.sqrt((SOT1_up_sd/SOT1_up_flux)**2 + (SOT1_low_sd/SOT1_low_flux)**2)
        SOT2_frac_err = np.sqrt((SOT2_up_sd/SOT2_up_flux)**2 + (SOT2_low_sd/SOT2_low_flux)**2)
        log_SOT1_err = SOT1_frac_err / ln10
        log_SOT2_err = SOT2_frac_err / ln10
        log_SOT_err = log_SOT1_err + 1.3*log_SOT2_err
        
        Av = cal_Av(wave, flux, H_alpha, H_beta, 2.85)
        x_corr = log_ratio(wave, flux, x1, x2, Av)
        SOT_corr = log_ratio(wave, flux, SOT1_up, SOT1_low, Av) + 1.3 * log_ratio(wave, flux, SOT2_up, SOT2_low, Av)
        
        axes.errorbar(x, SOT, xerr = log_x_err, yerr = log_SOT_err, fmt=form, ecolor = ecolor, label = label)
        axes.errorbar(x_corr, SOT_corr, xerr = log_x_err, yerr = log_SOT_err, fmt=form_corr, ecolor = ecolor, label = label+'_corr')
        
        
        index = 0
        if (label == 'High'):
            index = 0
        elif (label == 'Mid'):
            index = 1
        elif (label == 'Low'):
            index = 2
        
        x1_old = x1.old[index]
        x2_old = x2.old[index]
        x_old = np.log10(x1_old/x2_old)
        SOT1_up_old = SOT1_up.old[index]
        SOT1_low_old = SOT1_low.old[index]
        SOT2_up_old = SOT2_up.old[index]
        SOT2_low_old = SOT2_low.old[index]
        SOT_old = np.log10(SOT1_up_old/SOT1_low_old) + 1.3*np.log10(SOT2_up_old/SOT2_low_old)
        axes.plot(x_old, SOT_old, 's', color = ecolor, label = label + '_old')
        
    plot_error_bars_SOT(wave, high, high_var, x1, x2, SOT1_up, SOT1_low, SOT2_up, SOT2_low, 'r.','r*', 'r', 'High')
    plot_error_bars_SOT(wave, mid, mid_var, x1, x2, SOT1_up, SOT1_low, SOT2_up, SOT2_low, 'b.','b*', 'b', 'Mid')
    plot_error_bars_SOT(wave, low, low_var, x1, x2, SOT1_up, SOT1_low, SOT2_up, SOT2_low, 'k.','k*', 'k', 'Low')
        
    #   Plot the grid of photoionization model
    gridx = (np.log10(x1g) - np.log10(x2g))
    gridSOT1 = (np.log10(SOT1_upg) - np.log10(SOT1_lowg))
    gridSOT2 = (np.log10(SOT2_upg) - np.log10(SOT2_lowg))
    gridSOT = gridSOT1 + 1.3*gridSOT2 
    gx = np.reshape(gridx, [7,6])
    gSOT = np.reshape(gridSOT, [7,6])
    Mx = np.reshape(M, [7,6])
    Iy = np.reshape(I, [7,6])

    for i in range(gSOT.shape[0]):
        axes.plot(gx[i],gSOT[i], '-', color = 'grey', lw = 1.5)
        axes.annotate(round(Mx[i][0],2), axes.lines[-1].get_xydata()[0])
    for j in range(gSOT.shape[1]):
        axes.plot(gx[:,j],gSOT[:,j], '-', color = 'grey', lw = 1.5)    
        axes.annotate(Iy[:,j][0], axes.lines[-1].get_xydata()[-1])
    
    #   Plot the extinction vector
    xpos = xlim[0] + 0.1*(xlim[1] - xlim[0])
    ypos = ylim[0]+ 0.9*(ylim[1] - ylim[0])
    axes.arrow(xpos, ypos, extinction_vector(wave, x1, x2), extinction_vector(wave, SOT1_up, SOT1_low) + 1.3*extinction_vector(wave, SOT2_up, SOT2_low), length_includes_head=True, color='green', head_width = 0.03, head_length = 0.02)
    
    axes.set_xlabel('log {}/{}'.format(x1.name, x2.name))
    axes.set_ylabel('SOT')
    axes.set_xlim(xlim[0], xlim[1])
    axes.set_ylim(ylim[0], ylim[1])
    
    #   To prevent showing the error bars in the legend
    handles, labels = axes.get_legend_handles_labels()
    new_handles = []
    for h in handles:
        if isinstance(h, container.ErrorbarContainer):
            new_handles.append(h[0])
        else:
            new_handles.append(h)
    axes.legend(new_handles, labels)
    fig.savefig('../Output/4. Plotting/Figures 4.7/{}_{}_SOT.png'.format(x1.name, x2.name), format='png')  
    plt.close(fig)


plot_log_line_ratio(OIIdf, OIId, SIIdf, SIId, OT2, O2, ST2, S2, [-2.3,-1.2], [-1.7,-0.9])
plot_log_line_ratio(OIII, OIId, SIIdf, SIId, O3, O2, ST2, S2, [-2,0.9], [-1.7,-0.9])
plot_log_line_ratio(NIIf, NII, OIIIf, OIII, NT2, N2, OT3, O3, [-2.5,-1.2], [-3.5,-0.7])   
plot_log_line_ratio(NII, OIId, OIII, OIId, N2, O2, O3, O2, [-1.5,0.7], [-2,1])
plot_log_line_ratio(NII, OIId, NIIf, NII, N2, O2, NT2, N2, [-1.5,0.7], [-2.48,-1.2])
plot_log_line_ratio(NII, OIId, OIIIf, OIII, N2, O2, OT3, O3, [-1.5,0.7], [-3.5,-0.7])

plot_log_ratio_SOT(NII, OIId, OIId, OIIdf, SIId, SIIdf, N2, O2, O2, OT2, S2, ST2, [-2,1.5], [2.4,4.2])
plot_log_ratio_SOT(NII, H_alpha, OIId, OIIdf, SIId, SIIdf, N2, Ha, O2, OT2, S2, ST2, [-2,1.5], [2.4,4.2])
plot_log_ratio_SOT(NII, SIId, OIId, OIIdf, SIId, SIIdf, N2, S2, O2, OT2, S2, ST2, [-2,1.5], [2.4,4.2])

plot_log_ratio_SOT(SIId, SIIdf, OIId, OIIdf, SIId, SIIdf, S2, ST2, O2, OT2, S2, ST2, [0.9,1.7], [2.4,4.2])
plot_log_ratio_SOT(OIId, OIIdf, OIId, OIIdf, SIId, SIIdf, O2, OT2, O2, OT2, S2, ST2, [1.2,2.3], [2.4,4.2])

#log_line_ratio_plot(NII, OIId, OIId, OIIdf, N2, O2, O2, OT20, xlim, ylim)
#log_

# 13Gyr star population
# GASS
# N/O
# Stellar m and gas m same
    # High metallicty gas -> high metallicity stellar population model
# BC03 to generate stellar population

# Balmer decrement for dust estimation (Take Halpha and Hbeta)
# Check consistancy of different Balmer lines

# Plot 3d SOT　plots
# compare the numbers of SOT to the previous paper

# next: Improve continuum subtraction 

#%%

def plot_BPT():
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))
    axes[0].set_box_aspect(1)
    axes[1].set_box_aspect(1)
    axes[2].set_box_aspect(1)
    
    axes[1].set_xlabel('log([SII]6716,6731/H$\\alpha $)')
    X1 = np.linspace(-3.5,.3,100)
    X2 = np.linspace(-.305,.4,80)
    Y1 = 0.72/(X1-0.32)+1.3
    Y2 = 1.89*X2+0.76
    bound = axes[1].plot(X1, Y1, 'k', X2, Y2, 'k', lw=2.5)
    abound = axes[1].axis([-1.5,.5,-1.5,1.5])
    axes[1].text(-0.7, 1.3, 'Seyfert', fontsize=10, color='k')
    axes[1].text(-.1, -.2, 'LI(N)ER', fontsize=10, color='k')
    axes[1].text(-1.4, -1. ,'SF', fontsize=10, color='k')
    
    axes[0].set_xlabel('log([NII]6583/H$\\alpha $)')
    axes[0].set_ylabel('log([OIII]5007/H$\\beta $)')
    X1 = np.linspace(-3.5,0.4,100)
    X2 = np.linspace(-3.5,0.04,80)
    Y1 = 0.61/(X1-0.47)+1.19
    Y2 = 0.61/(X2-0.05)+1.3
    bound = axes[0].plot(X1, Y1, 'k', X2, Y2, 'k--')
    abound = axes[0].axis([-1.5,0.5,-1.5,1.5])
    axes[0].text(-0.5,1.2,'AGN', fontsize=10, color='k')
    axes[0].text(-0.1,-1.2,'Com', fontsize=10, color='k')
    axes[0].text(-.9,-0.75,'HII', fontsize=10, color='k')
    
    axes[0].tick_params(labelsize=10)
    axes[1].tick_params(labelsize=10)
    axes[2].tick_params(labelsize=10)
    axes[2].set_xlabel('log([OI]6300/H$\\alpha $)')
    axes[2].text(-1.5, 1.3, 'Seyfert', fontsize=10, color='k')
    axes[2].text(-0.4, -.2, 'LI(N)ER', fontsize=10, color='k')
    axes[2].text(-2.5, -1.3 ,'SF', fontsize=10, color='k')
    X1 = np.linspace(-3.5,-.8,100)
    X2 = np.linspace(-1.13,.4,80)
    Y1 = 0.73/(X1+0.59)+1.33
    Y2 = 1.18*X2+1.3
    bound = axes[2].plot(X1, Y1, 'k', X2, Y2, 'k', lw=2.5)
    
    axes[0].set_xlim(-1.5, 0.28)
    axes[1].set_xlim(-1.5, 0.28)
    axes[2].set_xlim(-2.8, 0.5)
    axes[0].set_ylim(-1.5, 1.5)
    axes[1].set_ylim(-1.5, 1.5)
    axes[2].set_ylim(-1.5, 1.5)
    
    l = [N2, S2, O1]
    for k in range(len(l)):
        gridx = (np.log10(l[k]) - np.log10(Ha))
        gridy = (np.log10(O3) - np.log10(Hb))
        gx = np.reshape(gridx, [7,6])
        gy = np.reshape(gridy, [7,6])
        Mx = np.reshape(M, [7,6])
        Iy = np.reshape(I, [7,6])
        for i in range(gy.shape[0]):
            axes[k].plot(gx[i],gy[i], '-', color = 'grey', lw = 1.5)    
            #axes[k].annotate(round(Mx[i][0],2), axes[k].lines[-1].get_xydata()[0])
        for j in range(gy.shape[1]):
            axes[k].plot(gx[:,j],gy[:,j], '-', color = 'grey', lw = 1.5)    
            #axes[k].annotate(Iy[:,j][0], axes[k].lines[-1].get_xydata()[-1])
            
    def plot_error_bars(wave, flux, var, x1, x2, y1, y2, form, form_corr, ecolor, label, axes):
        ln10 = np.log(10)
        
        x1_flux = summed_flux(x1, wave, flux, FWHM(x1, flux))
        x2_flux = summed_flux(x2, wave, flux, FWHM(x2, flux))
        x1_sd = sliding_box(wave, flux, x1)[1]
        x2_sd = sliding_box(wave, flux, x2)[1]
        x_frac_err = np.sqrt((x1_sd/x1_flux)**2 + (x2_sd/x2_flux)**2)
        log_x_err = x_frac_err / ln10
        x = log_ratio(wave, flux, x1, x2)
        
        y1_flux = summed_flux(y1, wave, flux, FWHM(y1, flux))
        y2_flux = summed_flux(y2, wave, flux, FWHM(y2, flux))
        y1_sd = sliding_box(wave, flux, y1)[1]
        y2_sd = sliding_box(wave, flux, y2)[1]
        y_frac_err = np.sqrt((y1_sd/y1_flux)**2 + (y2_sd/y2_flux)**2)
        log_y_err = y_frac_err / ln10
        y = log_ratio(wave, flux, y1, y2)
       
        arrows = plot_arrows(wave, flux, x1, x2, y1, y2)
        
        Av = cal_Av(wave, flux, H_alpha, H_beta, 2.85)
        ext_arr = ext.fitzpatrick99(wave, Av)
        
        if (arrows[0] == -1):
            x1_flux = 2 * sliding_box(wave, flux, x1)[1]
            x = np.log10(x1_flux/summed_flux(x2, wave, flux, FWHM(x2, flux)))
        elif (arrows[0] == 1):
            x2_flux = 2 * sliding_box(wave, flux, x2)[1]
            x = np.log10(summed_flux(x1, wave, flux, FWHM(x1, flux))/x2_flux)
        x1_index = np.where(x1.w < wave)[0][0]
        x2_index = np.where(x2.w < wave)[0][0]
        x_corr = x - 0.4 * (ext_arr[x2_index] - ext_arr[x1_index])
        
        if (arrows[1] == -1):
            y1_flux = 2 * sliding_box(wave, flux, y1)[1]
            y = np.log10(y1_flux/summed_flux(y2, wave, flux, FWHM(y2, flux)))
        elif (arrows[1] == 1):
            y2_flux = 2 * sliding_box(wave, flux, y2)[1]
            y = np.log10(summed_flux(y1, wave, flux, FWHM(y1, flux))/y2_flux)
        y1_index = np.where(y1.w < wave)[0][0]
        y2_index = np.where(y2.w < wave)[0][0]
        y_corr = y - 0.4 * (ext_arr[y2_index] - ext_arr[y1_index])
        
        if ((arrows[0] != 0) | (arrows[1] != 0)):
            axes.plot(x, y, form, label = label)
            axes.plot(x_corr, y_corr, form_corr, label = label + '_corr')
            
            axes.arrow(x, y, arrows[0]*0.5, arrows[1]*0.5, color = ecolor, head_width = 0.03, head_length = 0.02)        
            axes.arrow(x_corr, y_corr, arrows[0]*0.5, arrows[1]*0.5, color = ecolor, head_width = 0.03, head_length = 0.02)            
            
            if (arrows[0] == 0):
                axes.errorbar(x, y, xerr = log_x_err, yerr = 0, fmt=form, ecolor = ecolor)
                axes.errorbar(x_corr, y_corr, xerr = log_x_err, yerr = 0, fmt=form, ecolor = ecolor)         
                
            elif (arrows[1] == 0):
                axes.errorbar(x, y, xerr = 0, yerr = log_y_err, fmt=form, ecolor = ecolor)    
                axes.errorbar(x_corr, y_corr, xerr = 0, yerr = log_y_err, fmt=form, ecolor = ecolor)    
        else:
            axes.errorbar(x, y, xerr = log_x_err, yerr = log_y_err, fmt=form, ecolor = ecolor, label = label)
            axes.errorbar(x_corr, y_corr, xerr = log_x_err, yerr = log_y_err, fmt=form_corr, ecolor = ecolor, label = label + '_corr')
        
        #   Plot the data points of 2018
        index = 0
        if (label == 'High'):
            index = 0
        elif (label == 'Mid'):
            index = 1
        elif (label == 'Low'):
            index = 2
        
        x1_old = x1.old[index]
        x2_old = x2.old[index]
        y1_old = y1.old[index]
        y2_old = y2.old[index]
        x_old = np.log10(x1_old/x2_old)
        y_old = np.log10(y1_old/y2_old)
        axes.plot(x_old, y_old, 's', color = ecolor, label = label + '_old')
    
    l_o = [NII, SIId, OI]
    for k in range(len(l_o)):
        plot_error_bars(wave, high, high_var, l_o[k], H_alpha, OIII, H_beta, '.r','r*' , 'r', 'High', axes[k])
        plot_error_bars(wave, mid, mid_var, l_o[k], H_alpha, OIII, H_beta,'.b','b*', 'b', 'Mid', axes[k])
        plot_error_bars(wave, low, low_var, l_o[k], H_alpha, OIII, H_beta, '.k','k*', 'k', 'Low', axes[k])
    
    handles, labels = axes[0].get_legend_handles_labels()
    new_handles = []
    for h in handles:
        if isinstance(h, container.ErrorbarContainer):
            new_handles.append(h[0])
        else:
            new_handles.append(h)
    axes[0].legend(new_handles, labels)
    
    fig.savefig('../Output/4. Plotting/BPT diagrams.png', format='png')  
    plt.close(fig)
    

plot_BPT()
#%%

def plot_log_ratio_SOT_3d(x1, x2, SOT1_up, SOT1_low, SOT2_up, SOT2_low, z1, z2, x1g, x2g, SOT1_upg, SOT1_lowg, SOT2_upg, SOT2_lowg, z1g, z2g, xlim, ylim, zlim):
    fig = plt.figure(figsize = (14,14))
    axes = fig.add_subplot(111, projection = '3d')

    def plot_error_bars_SOT(wave, flux, var, x1, x2, SOT1_up, SOT1_low, SOT2_up, SOT2_low, z1, z2, form, form_corr, ecolor, label):
        ln10 = np.log(10)
        x1_flux = summed_flux(x1, wave, flux, FWHM(x1, flux))
        x2_flux = summed_flux(x2, wave, flux, FWHM(x2, flux))
        x1_sd = sliding_box(wave, flux, x1)[1]
        x2_sd = sliding_box(wave, flux, x2)[1]
        x_frac_err = np.sqrt((x1_sd/x1_flux)**2 + (x2_sd/x2_flux)**2)
        log_x_err = x_frac_err / ln10
        x = log_ratio(wave, flux, x1, x2)
        
        SOT1_up_flux = summed_flux(SOT1_up, wave, flux, FWHM(SOT1_up, flux))
        SOT1_low_flux = summed_flux(SOT1_low, wave, flux, FWHM(SOT1_low, flux))
        SOT2_up_flux = summed_flux(SOT2_up, wave, flux, FWHM(SOT2_up, flux))
        SOT2_low_flux = summed_flux(SOT2_low, wave, flux, FWHM(SOT2_low, flux))
        SOT1_up_sd = sliding_box(wave, flux, SOT1_up)[1]
        SOT1_low_sd = sliding_box(wave, flux, SOT1_low)[1]
        SOT2_up_sd = sliding_box(wave, flux, SOT2_up)[1]
        SOT2_low_sd = sliding_box(wave, flux, SOT2_low)[1]
        SOT = log_ratio(wave, flux, SOT1_up, SOT1_low) + 1.3 * log_ratio(wave, flux, SOT2_up, SOT2_low)
        SOT1_frac_err = np.sqrt((SOT1_up_sd/SOT1_up_flux)**2 + (SOT1_low_sd/SOT1_low_flux)**2)
        SOT2_frac_err = np.sqrt((SOT2_up_sd/SOT2_up_flux)**2 + (SOT2_low_sd/SOT2_low_flux)**2)
        log_SOT1_err = SOT1_frac_err / ln10
        log_SOT2_err = SOT2_frac_err / ln10
        log_SOT_err = log_SOT1_err + 1.3*log_SOT2_err
        
        z1_flux = summed_flux(z1, wave, flux, FWHM(z1, flux))
        z2_flux = summed_flux(z2, wave, flux, FWHM(z2, flux))
        z1_sd = sliding_box(wave, flux, z1)[1]
        z2_sd = sliding_box(wave, flux, z2)[1]
        z_frac_err = np.sqrt((z1_sd/z1_flux)**2 + (z2_sd/z2_flux)**2)
        log_z_err = z_frac_err / ln10
        z = log_ratio(wave, flux, z1, z2)
        
        axes.scatter(x, SOT, z, marker = 'o', color = ecolor, label = label)
        
        Av = cal_Av(wave, flux, H_alpha, H_beta, 2.85)
        x_corr = log_ratio(wave, flux, x1, x2, Av)
        SOT_corr = log_ratio(wave, flux, SOT1_up, SOT1_low, Av) + 1.3 * log_ratio(wave, flux, SOT2_up, SOT2_low, Av)
        z_corr = log_ratio(wave, flux, z1, z2, Av)
        
        axes.scatter(x_corr, SOT_corr, z_corr, marker = '*', color = ecolor, label = label + '_corr')
        
        #plot error bars
        axes.plot([x_corr + log_x_err, x_corr - log_x_err], [SOT_corr, SOT_corr], [z_corr, z_corr], marker = '_', color = ecolor)
        axes.plot([x_corr, x_corr], [SOT_corr + log_SOT_err, SOT_corr - log_SOT_err], [z_corr, z_corr], marker = '_', color = ecolor)
        axes.plot([x_corr, x_corr], [SOT_corr, SOT_corr], [z_corr + log_z_err, z_corr - log_z_err], marker = '_', color = ecolor)
        
        index = 0
        if (label == 'High'):
            index = 0
        elif (label == 'Mid'):
            index = 1
        elif (label == 'Low'):
            index = 2
        
        x1_old = x1.old[index]
        x2_old = x2.old[index]
        x_old = np.log10(x1_old/x2_old)
        SOT1_up_old = SOT1_up.old[index]
        SOT1_low_old = SOT1_low.old[index]
        SOT2_up_old = SOT2_up.old[index]
        SOT2_low_old = SOT2_low.old[index]
        SOT_old = np.log10(SOT1_up_old/SOT1_low_old) + 1.3*np.log10(SOT2_up_old/SOT2_low_old)
        z1_old = z1.old[index]
        z2_old = z2.old[index]
        z_old = np.log10(z1_old/z2_old)

        axes.scatter(x_old, SOT_old, z_old, marker = 's', color = ecolor, label = label + '_old')
    
    plot_error_bars_SOT(wave, high, high_var, x1, x2, SOT1_up, SOT1_low, SOT2_up, SOT2_low, z1, z2, 'r.', 'r^', 'r', 'High')
    plot_error_bars_SOT(wave, mid, mid_var, x1, x2, SOT1_up, SOT1_low, SOT2_up, SOT2_low, z1, z2, 'b.', 'b^', 'b', 'Mid')
    plot_error_bars_SOT(wave, low, low_var, x1, x2, SOT1_up, SOT1_low, SOT2_up, SOT2_low, z1, z2, 'k.', 'k^', 'k', 'Low')
    
    #   Plot the grid of photoionization model
    gridx = (np.log10(x1g) - np.log10(x2g))
    gridSOT1 = (np.log10(SOT1_upg) - np.log10(SOT1_lowg))
    gridSOT2 = (np.log10(SOT2_upg) - np.log10(SOT2_lowg))
    gridSOT = gridSOT1 + 1.3*gridSOT2
    gridz = (np.log10(z1g) - np.log10(z2g))
    gx = np.reshape(gridx, [7,6])
    gSOT = np.reshape(gridSOT, [7,6])
    gz = np.reshape(gridz, [7,6])
    
    axes.plot_wireframe(gx, gSOT, gz, color = 'grey')
    #axes.plot_surface(gx, gSOT, gz)

    axes.set_xlabel('log {}/{}'.format(x1.name, x2.name))
    axes.set_ylabel('SOT')
    axes.set_zlabel('log {}/{}'.format(z1.name, z2.name))
    axes.set_xlim(xlim[0], xlim[1])
    axes.set_ylim(ylim[0], ylim[1])
    axes.set_zlim(zlim[0], zlim[1])
    axes.legend()
    fig.show()


plot_log_ratio_SOT_3d(NII, SIId, OIId, OIIdf, SIId, SIIdf, NII, H_alpha, N2, S2, O2, OT2, S2, ST2, N2, Ha, [-2,1.5], [2.4,4.2],[-1.5, 0.28])
plot_log_ratio_SOT_3d(OIII, H_beta, OIId, OIIdf, SIId, SIIdf, NII, H_alpha, O3, Hb, O2, OT2, S2, ST2, N2, Ha, [-1.5,1.5], [2.4,4.2],[-1.5, 0.28])


def plot_log_ratio_3d(x1, x2, y1, y2, z1, z2, x1g, x2g, y1g, y2g, z1g, z2g, xlim, ylim, zlim):
    fig = plt.figure(figsize = (14,14))
    axes = fig.add_subplot(111, projection = '3d')

    def plot_error_bars(wave, flux, var, x1, x2, y1, y2, z1, z2, form, form_corr, ecolor, label):
        ln10 = np.log(10)
        x1_flux = summed_flux(x1, wave, flux, FWHM(x1, flux))
        x2_flux = summed_flux(x2, wave, flux, FWHM(x2, flux))
        x1_sd = sliding_box(wave, flux, x1)[1]
        x2_sd = sliding_box(wave, flux, x2)[1]
        x_frac_err = np.sqrt((x1_sd/x1_flux)**2 + (x2_sd/x2_flux)**2)
        log_x_err = x_frac_err / ln10
        x = log_ratio(wave, flux, x1, x2)
        
        y1_flux = summed_flux(y1, wave, flux, FWHM(y1, flux))
        y2_flux = summed_flux(y2, wave, flux, FWHM(y2, flux))
        y1_sd = sliding_box(wave, flux, y1)[1]
        y2_sd = sliding_box(wave, flux, y2)[1]
        y_frac_err = np.sqrt((y1_sd/y1_flux)**2 + (y2_sd/y2_flux)**2)
        log_y_err = y_frac_err / ln10
        y = log_ratio(wave, flux, y1, y2)

        z1_flux = summed_flux(z1, wave, flux, FWHM(z1, flux))
        z2_flux = summed_flux(z2, wave, flux, FWHM(z2, flux))
        z1_sd = sliding_box(wave, flux, z1)[1]
        z2_sd = sliding_box(wave, flux, z2)[1]
        z_frac_err = np.sqrt((z1_sd/z1_flux)**2 + (z2_sd/z2_flux)**2)
        log_z_err = z_frac_err / ln10
        z = log_ratio(wave, flux, z1, z2)
        
        axes.scatter(x, y, z, marker = 'o', color = ecolor, label = label)
        
        Av = cal_Av(wave, flux, H_alpha, H_beta, 2.85)
        x_corr = log_ratio(wave, flux, x1, x2, Av)
        y_corr = log_ratio(wave, flux, y1, y2, Av)
        z_corr = log_ratio(wave, flux, z1, z2, Av)
        
        axes.scatter(x_corr, y_corr, z_corr, marker = '*', color = ecolor, label = label + '_corr')
        
        #plot error bars
        axes.plot([x_corr + log_x_err, x_corr - log_x_err], [y_corr, y_corr], [z_corr, z_corr], marker = '_', color = ecolor)
        axes.plot([x_corr, x_corr], [y_corr + log_y_err, y_corr - log_y_err], [z_corr, z_corr], marker = '_', color = ecolor)
        axes.plot([x_corr, x_corr], [y_corr, y_corr], [z_corr + log_z_err, z_corr - log_z_err], marker = '_', color = ecolor)
        
        index = 0
        if (label == 'High'):
            index = 0
        elif (label == 'Mid'):
            index = 1
        elif (label == 'Low'):
            index = 2
        
        x1_old = x1.old[index]
        x2_old = x2.old[index]
        x_old = np.log10(x1_old/x2_old)
        y1_old = y1.old[index]
        y2_old = y2.old[index]
        y_old = np.log10(y1_old/y2_old)
        z1_old = z1.old[index]
        z2_old = z2.old[index]
        z_old = np.log10(z1_old/z2_old)

        axes.scatter(x_old, y_old, z_old, marker = 's', color = ecolor, label = label + '_old')
    
    plot_error_bars(wave, high, high_var, x1, x2, y1, y2, z1, z2, 'r.', 'r^', 'r', 'High')
    plot_error_bars(wave, mid, mid_var, x1, x2, y1, y2, z1, z2, 'b.', 'b^', 'b', 'Mid')
    plot_error_bars(wave, low, low_var, x1, x2, y1, y2, z1, z2, 'k.', 'k^', 'k', 'Low')
    
    #   Plot the grid of photoionization model
    gridx = (np.log10(x1g) - np.log10(x2g))
    gridy = (np.log10(y1g) - np.log10(y2g))
    gridz = (np.log10(z1g) - np.log10(z2g))
    gx = np.reshape(gridx, [7,6])
    gy = np.reshape(gridy, [7,6])
    gz = np.reshape(gridz, [7,6])
    
    axes.plot_wireframe(gx, gy, gz, color = 'grey')
    #axes.plot_surface(gx, gy, gz, color = 'grey')

    axes.set_xlabel('log {}/{}'.format(x1.name, x2.name))
    axes.set_ylabel('log {}/{}'.format(y1.name, y2.name))
    axes.set_zlabel('log {}/{}'.format(z1.name, z2.name))
    axes.set_xlim(xlim[0], xlim[1])
    axes.set_ylim(ylim[0], ylim[1])
    axes.set_zlim(zlim[0], zlim[1])
    axes.legend()
    fig.show()
    
plot_log_ratio_3d(OIIdf, OIId, NII, H_alpha, OIII, H_beta, OT2, O2, N2, Ha, O3, Hb, [-2.3,-1.2], [-1.5, 0.28], [-1.5, 1.5])
plot_log_ratio_3d(SIIdf, SIId, NII, H_alpha, OIII, H_beta, ST2, S2, N2, Ha, O3, Hb, [-1.7,-0.9], [-1.5, 0.28], [-1.5, 1.5])

#axis: 1 temp, 1 metal, 1 ionization parameter


#%%
def get_line(x1, x2, y1, y2, x1g, x2g, y1g, y2g, xlim, ylim):
    
    fig, axes = plt.subplots(figsize=(7,7))

    #   Plot the error bars of the three points (If y = a/c, dy^2 = ((da/a)^2 + (dc/c)^2) y^2), d (log10x) = 1/ln10 * dx / x 
    def plot_error_bars(wave, flux, var, x1, x2, y1, y2, form, form_corr, ecolor, label):
        ln10 = np.log(10)
        
        x1_flux = summed_flux(x1, wave, flux, FWHM(x1, flux))
        x2_flux = summed_flux(x2, wave, flux, FWHM(x2, flux))
        x1_sd = sliding_box(wave, flux, x1)[1]
        x2_sd = sliding_box(wave, flux, x2)[1]
        x_frac_err = np.sqrt((x1_sd/x1_flux)**2 + (x2_sd/x2_flux)**2)
        log_x_err = x_frac_err / ln10
        x = log_ratio(wave, flux, x1, x2)
        
        y1_flux = summed_flux(y1, wave, flux, FWHM(y1, flux))
        y2_flux = summed_flux(y2, wave, flux, FWHM(y2, flux))
        y1_sd = sliding_box(wave, flux, y1)[1]
        y2_sd = sliding_box(wave, flux, y2)[1]
        y_frac_err = np.sqrt((y1_sd/y1_flux)**2 + (y2_sd/y2_flux)**2)
        log_y_err = y_frac_err / ln10
        y = log_ratio(wave, flux, y1, y2)
       
        arrows = plot_arrows(wave, flux, x1, x2, y1, y2)
        
        if (label == 'High'):
            Av = 1.0119931437150211
        elif (label == 'Mid'):
            Av = 0.45041753046755334
        else:
            Av = 0.13117073723062803
        ext_arr = ext.fitzpatrick99(wave, Av)
        
        if (arrows[0] == -1):
            x1_flux = 2 * sliding_box(wave, flux, x1)[1]
            x = np.log10(x1_flux/summed_flux(x2, wave, flux, FWHM(x2, flux)))
        elif (arrows[0] == 1):
            x2_flux = 2 * sliding_box(wave, flux, x2)[1]
            x = np.log10(summed_flux(x1, wave, flux, FWHM(x1, flux))/x2_flux)
        x1_index = np.where(x1.w < wave)[0][0]
        x2_index = np.where(x2.w < wave)[0][0]
        x_corr = x - 0.4 * (ext_arr[x2_index] - ext_arr[x1_index])
        
        if (arrows[1] == -1):
            y1_flux = 2 * sliding_box(wave, flux, y1)[1]
            y = np.log10(y1_flux/summed_flux(y2, wave, flux, FWHM(y2, flux)))
        elif (arrows[1] == 1):
            y2_flux = 2 * sliding_box(wave, flux, y2)[1]
            y = np.log10(summed_flux(y1, wave, flux, FWHM(y1, flux))/y2_flux)
        y1_index = np.where(y1.w < wave)[0][0]
        y2_index = np.where(y2.w < wave)[0][0]
        y_corr = y - 0.4 * (ext_arr[y2_index] - ext_arr[y1_index])
        
        if ((arrows[0] != 0) | (arrows[1] != 0)):
            axes.plot(x, y, form, label = label)
            axes.plot(x_corr, y_corr, form_corr, label = label + '_corr')
            
            axes.arrow(x, y, arrows[0]*0.5, arrows[1]*0.5, color = ecolor, head_width = 0.03, head_length = 0.02)        
            axes.arrow(x_corr, y_corr, arrows[0]*0.5, arrows[1]*0.5, color = ecolor, head_width = 0.03, head_length = 0.02)            
            
            if (arrows[0] == 0):
                axes.errorbar(x, y, xerr = log_x_err, yerr = 0, fmt=form, ecolor = ecolor)
                axes.errorbar(x_corr, y_corr, xerr = log_x_err, yerr = 0, fmt=form, ecolor = ecolor)         
                
            elif (arrows[1] == 0):
                axes.errorbar(x, y, xerr = 0, yerr = log_y_err, fmt=form, ecolor = ecolor)    
                axes.errorbar(x_corr, y_corr, xerr = 0, yerr = log_y_err, fmt=form, ecolor = ecolor)    
        else:
            axes.errorbar(x, y, xerr = log_x_err, yerr = log_y_err, fmt=form, ecolor = ecolor, label = label)
            axes.errorbar(x_corr, y_corr, xerr = log_x_err, yerr = log_y_err, fmt=form_corr, ecolor = ecolor, label = label + '_corr')
        
        #   Plot the data points of 2018
        index = 0
        if (label == 'High'):
            index = 0
        elif (label == 'Mid'):
            index = 1
        elif (label == 'Low'):
            index = 2
        
        x1_old = x1.old[index]
        x2_old = x2.old[index]
        y1_old = y1.old[index]
        y2_old = y2.old[index]
        x_old = np.log10(x1_old/x2_old)
        y_old = np.log10(y1_old/y2_old)
        axes.plot(x_old, y_old, 's', color = ecolor, label = label + '_old')
        
        #print(x, y)
        
       
    plot_error_bars(wave, high, high_var, x1, x2, y1, y2, '.r','r*' , 'r', 'High')
    plot_error_bars(wave, mid, mid_var, x1, x2, y1, y2, '.b','b*', 'b', 'Mid')
    plot_error_bars(wave, low, low_var, x1, x2, y1, y2, '.k','k*', 'k', 'Low')
    
    #   Plot the grid of photoionization model
    gridx = (np.log10(x1g) - np.log10(x2g))
    gridy = (np.log10(y1g) - np.log10(y2g))
    gx = np.reshape(gridx, [7,6])
    gy = np.reshape(gridy, [7,6])
    gx = gx[:,:3].flatten()
    gy = gy[:,:3].flatten()
    
    axes.scatter(gx, gy, color = 'grey', alpha = 0.5)
    m, c = np.polyfit(gx, gy, 1)
    axes.plot(gx, m*gx+c)
    #print('m:', m, 'c: ', c)
    
    #   Plot the extinction vector
    xpos = xlim[0] + 0.1*(xlim[1] - xlim[0])
    ypos = ylim[0]+ 0.9*(ylim[1] - ylim[0])
    axes.arrow(xpos, ypos, extinction_vector(wave, x1, x2), extinction_vector(wave, y1, y2), length_includes_head=True, color='green', head_width = 0.03, head_length = 0.02)
    #print(extinction_vector(wave, x1, x2), extinction_vector(wave, y1, y2))
    
    axes.set_xlabel('log {}/{}'.format(x1.name, x2.name))
    axes.set_ylabel('log {}/{}'.format(y1.name, y2.name))
    axes.set_xlim(xlim[0], xlim[1])
    axes.set_ylim(ylim[0], ylim[1])
    
    #   To prevent showing the error bars in the legend
    handles, labels = axes.get_legend_handles_labels()
    new_handles = []
    for h in handles:
        if isinstance(h, container.ErrorbarContainer):
            new_handles.append(h[0])
        else:
            new_handles.append(h)
    axes.legend(new_handles, labels)
    fig.show()

get_line(OIIdf, OIId, SIIdf, SIId, OT2, O2, ST2, S2, [-2.3,-1.2], [-1.7,-0.9])
 

y_high = -1.5882855063508743
x_high = -1.3996191867895946
y_mid = -1.4583151388435962
x_mid = -1.6225867565140095
y_low = -1.3319987202100125
x_low = -1.670431784696089
m = 0.6644448281197751
c = -0.15712672409884582
cx = -0.3441158759035662
cy = 0.2666034403533839

#print('A_high:', (y_high-m*x_high-c)/(m*cx-cy))
#print('A_mid:', (y_mid-m*x_mid-c)/(m*cx-cy))
#print('A_low:', (y_low-m*x_low-c)/(m*cx-cy))

def ext_new_SOT(x1, x2, SOT1_up, SOT1_low, SOT2_up, SOT2_low, x1g, x2g, SOT1_upg, SOT1_lowg, SOT2_upg, SOT2_lowg, xlim, ylim):
    fig, axes = plt.subplots(figsize=(7,7))
    def plot_error_bars_SOT(wave, flux, var, x1, x2, SOT1_up, SOT1_low, SOT2_up, SOT2_low, form, form_corr, ecolor, label):
        ln10 = np.log(10)
        x1_flux = summed_flux(x1, wave, flux, FWHM(x1, flux))
        x2_flux = summed_flux(x2, wave, flux, FWHM(x2, flux))
        x1_sd = sliding_box(wave, flux, x1)[1]
        x2_sd = sliding_box(wave, flux, x2)[1]
        x_frac_err = np.sqrt((x1_sd/x1_flux)**2 + (x2_sd/x2_flux)**2)
        log_x_err = x_frac_err / ln10
        x = log_ratio(wave, flux, x1, x2)
        
        SOT1_up_flux = summed_flux(SOT1_up, wave, flux, FWHM(SOT1_up, flux))
        SOT1_low_flux = summed_flux(SOT1_low, wave, flux, FWHM(SOT1_low, flux))
        SOT2_up_flux = summed_flux(SOT2_up, wave, flux, FWHM(SOT2_up, flux))
        SOT2_low_flux = summed_flux(SOT2_low, wave, flux, FWHM(SOT2_low, flux))
        SOT1_up_sd = sliding_box(wave, flux, SOT1_up)[1]
        SOT1_low_sd = sliding_box(wave, flux, SOT1_low)[1]
        SOT2_up_sd = sliding_box(wave, flux, SOT2_up)[1]
        SOT2_low_sd = sliding_box(wave, flux, SOT2_low)[1]
        SOT = log_ratio(wave, flux, SOT1_up, SOT1_low) + 1.3 * log_ratio(wave, flux, SOT2_up, SOT2_low)
        SOT1_frac_err = np.sqrt((SOT1_up_sd/SOT1_up_flux)**2 + (SOT1_low_sd/SOT1_low_flux)**2)
        SOT2_frac_err = np.sqrt((SOT2_up_sd/SOT2_up_flux)**2 + (SOT2_low_sd/SOT2_low_flux)**2)
        log_SOT1_err = SOT1_frac_err / ln10
        log_SOT2_err = SOT2_frac_err / ln10
        log_SOT_err = log_SOT1_err + 1.3*log_SOT2_err
        
        if (label == 'High'):
            Av = 1.0119931437150211
        elif (label == 'Mid'):
            Av = 0.45041753046755334
        else:
            Av = 0.13117073723062803
        
        x_corr = log_ratio(wave, flux, x1, x2, Av)
        SOT_corr = log_ratio(wave, flux, SOT1_up, SOT1_low, Av) + 1.3 * log_ratio(wave, flux, SOT2_up, SOT2_low, Av)
        
        #axes.errorbar(x, SOT, xerr = log_x_err, yerr = log_SOT_err, fmt=form, ecolor = ecolor, label = label)
        axes.errorbar(x_corr, SOT_corr, xerr = log_x_err, yerr = log_SOT_err, fmt=form_corr, ecolor = ecolor, label = label+'_corr')
        
        
        index = 0
        if (label == 'High'):
            index = 0
        elif (label == 'Mid'):
            index = 1
        elif (label == 'Low'):
            index = 2
        
        x1_old = x1.old[index]
        x2_old = x2.old[index]
        x_old = np.log10(x1_old/x2_old)
        SOT1_up_old = SOT1_up.old[index]
        SOT1_low_old = SOT1_low.old[index]
        SOT2_up_old = SOT2_up.old[index]
        SOT2_low_old = SOT2_low.old[index]
        SOT_old = np.log10(SOT1_up_old/SOT1_low_old) + 1.3*np.log10(SOT2_up_old/SOT2_low_old)
        #axes.plot(x_old, SOT_old, 's', color = ecolor, label = label + '_old')
        
    plot_error_bars_SOT(wave, high, high_var, x1, x2, SOT1_up, SOT1_low, SOT2_up, SOT2_low, 'r.','r*', 'r', 'High')
    plot_error_bars_SOT(wave, mid, mid_var, x1, x2, SOT1_up, SOT1_low, SOT2_up, SOT2_low, 'b.','b*', 'b', 'Mid')
    plot_error_bars_SOT(wave, low, low_var, x1, x2, SOT1_up, SOT1_low, SOT2_up, SOT2_low, 'k.','k*', 'k', 'Low')
        
    #   Plot the grid of photoionization model
    gridx = (np.log10(x1g) - np.log10(x2g))
    gridSOT1 = (np.log10(SOT1_upg) - np.log10(SOT1_lowg))
    gridSOT2 = (np.log10(SOT2_upg) - np.log10(SOT2_lowg))
    gridSOT = gridSOT1 + 1.3*gridSOT2 
    gx = np.reshape(gridx, [7,6])
    gSOT = np.reshape(gridSOT, [7,6])
    Mx = np.reshape(M, [7,6])
    Iy = np.reshape(I, [7,6])

    for i in range(gSOT.shape[0]):
        axes.plot(gx[i],gSOT[i], '-', color = 'grey', lw = 1.5)
        axes.annotate(round(Mx[i][0],2), axes.lines[-1].get_xydata()[0])
    for j in range(gSOT.shape[1]):
        axes.plot(gx[:,j],gSOT[:,j], '-', color = 'grey', lw = 1.5)    
        axes.annotate(Iy[:,j][0], axes.lines[-1].get_xydata()[-1])
    
    #   Plot the extinction vector
    xpos = xlim[0] + 0.1*(xlim[1] - xlim[0])
    ypos = ylim[0]+ 0.9*(ylim[1] - ylim[0])
    axes.arrow(xpos, ypos, extinction_vector(wave, x1, x2), extinction_vector(wave, SOT1_up, SOT1_low) + 1.3*extinction_vector(wave, SOT2_up, SOT2_low), length_includes_head=True, color='green', head_width = 0.03, head_length = 0.02)
    
    axes.set_xlabel('log {}/{}'.format(x1.name, x2.name))
    axes.set_ylabel('SOT')
    axes.set_xlim(xlim[0], xlim[1])
    axes.set_ylim(ylim[0], ylim[1])
    
    #   To prevent showing the error bars in the legend
    handles, labels = axes.get_legend_handles_labels()
    new_handles = []
    for h in handles:
        if isinstance(h, container.ErrorbarContainer):
            new_handles.append(h[0])
        else:
            new_handles.append(h)
    axes.legend(new_handles, labels)
    fig.savefig('../Output/4. Plotting/Figures 4.8/{}_{}_SOT.png'.format(x1.name, x2.name), format='png')  
    plt.close(fig)
    
def ext_new(x1, x2, y1, y2, x1g, x2g, y1g, y2g, xlim, ylim):
    
    fig, axes = plt.subplots(figsize=(7,7))

    #   Plot the error bars of the three points (If y = a/c, dy^2 = ((da/a)^2 + (dc/c)^2) y^2), d (log10x) = 1/ln10 * dx / x 
    def plot_error_bars(wave, flux, var, x1, x2, y1, y2, form, form_corr, ecolor, label):
        ln10 = np.log(10)
        
        x1_flux = summed_flux(x1, wave, flux, FWHM(x1, flux))
        x2_flux = summed_flux(x2, wave, flux, FWHM(x2, flux))
        x1_sd = sliding_box(wave, flux, x1)[1]
        x2_sd = sliding_box(wave, flux, x2)[1]
        x_frac_err = np.sqrt((x1_sd/x1_flux)**2 + (x2_sd/x2_flux)**2)
        log_x_err = x_frac_err / ln10
        x = log_ratio(wave, flux, x1, x2)
        
        y1_flux = summed_flux(y1, wave, flux, FWHM(y1, flux))
        y2_flux = summed_flux(y2, wave, flux, FWHM(y2, flux))
        y1_sd = sliding_box(wave, flux, y1)[1]
        y2_sd = sliding_box(wave, flux, y2)[1]
        y_frac_err = np.sqrt((y1_sd/y1_flux)**2 + (y2_sd/y2_flux)**2)
        log_y_err = y_frac_err / ln10
        y = log_ratio(wave, flux, y1, y2)
       
        arrows = plot_arrows(wave, flux, x1, x2, y1, y2)
        
        if (label == 'High'):
            Av = 1.0119931437150211
        elif (label == 'Mid'):
            Av = 0.45041753046755334
        else:
            Av = 0.13117073723062803
        ext_arr = ext.fitzpatrick99(wave, Av)
        
        if (arrows[0] == -1):
            x1_flux = 2 * sliding_box(wave, flux, x1)[1]
            x = np.log10(x1_flux/summed_flux(x2, wave, flux, FWHM(x2, flux)))
        elif (arrows[0] == 1):
            x2_flux = 2 * sliding_box(wave, flux, x2)[1]
            x = np.log10(summed_flux(x1, wave, flux, FWHM(x1, flux))/x2_flux)
        x1_index = np.where(x1.w < wave)[0][0]
        x2_index = np.where(x2.w < wave)[0][0]
        x_corr = x - 0.4 * (ext_arr[x2_index] - ext_arr[x1_index])
        
        if (arrows[1] == -1):
            y1_flux = 2 * sliding_box(wave, flux, y1)[1]
            y = np.log10(y1_flux/summed_flux(y2, wave, flux, FWHM(y2, flux)))
        elif (arrows[1] == 1):
            y2_flux = 2 * sliding_box(wave, flux, y2)[1]
            y = np.log10(summed_flux(y1, wave, flux, FWHM(y1, flux))/y2_flux)
        y1_index = np.where(y1.w < wave)[0][0]
        y2_index = np.where(y2.w < wave)[0][0]
        y_corr = y - 0.4 * (ext_arr[y2_index] - ext_arr[y1_index])
        
        if ((arrows[0] != 0) | (arrows[1] != 0)):
            #axes.plot(x, y, form, label = label)
            axes.plot(x_corr, y_corr, form_corr, label = label + '_corr')
            
            #axes.arrow(x, y, arrows[0]*0.5, arrows[1]*0.5, color = ecolor, head_width = 0.03, head_length = 0.02)        
            axes.arrow(x_corr, y_corr, arrows[0]*0.5, arrows[1]*0.5, color = ecolor, head_width = 0.03, head_length = 0.02)            
            
            if (arrows[0] == 0):
                #axes.errorbar(x, y, xerr = log_x_err, yerr = 0, fmt=form, ecolor = ecolor)
                axes.errorbar(x_corr, y_corr, xerr = log_x_err, yerr = 0, fmt=form, ecolor = ecolor)         
                
            elif (arrows[1] == 0):
                #axes.errorbar(x, y, xerr = 0, yerr = log_y_err, fmt=form, ecolor = ecolor)    
                axes.errorbar(x_corr, y_corr, xerr = 0, yerr = log_y_err, fmt=form, ecolor = ecolor)    
        else:
            #axes.errorbar(x, y, xerr = log_x_err, yerr = log_y_err, fmt=form, ecolor = ecolor, label = label)
            axes.errorbar(x_corr, y_corr, xerr = log_x_err, yerr = log_y_err, fmt=form_corr, ecolor = ecolor, label = label + '_corr')
        
        #   Plot the data points of 2018
        index = 0
        if (label == 'High'):
            index = 0
        elif (label == 'Mid'):
            index = 1
        elif (label == 'Low'):
            index = 2
        
        x1_old = x1.old[index]
        x2_old = x2.old[index]
        y1_old = y1.old[index]
        y2_old = y2.old[index]
        x_old = np.log10(x1_old/x2_old)
        y_old = np.log10(y1_old/y2_old)
        #axes.plot(x_old, y_old, 's', color = ecolor, label = label + '_old')
        
       
    plot_error_bars(wave, high, high_var, x1, x2, y1, y2, '.r','r*' , 'r', 'High')
    plot_error_bars(wave, mid, mid_var, x1, x2, y1, y2, '.b','b*', 'b', 'Mid')
    plot_error_bars(wave, low, low_var, x1, x2, y1, y2, '.k','k*', 'k', 'Low')
    
    #   Plot the grid of photoionization model
    gridx = (np.log10(x1g) - np.log10(x2g))
    gridy = (np.log10(y1g) - np.log10(y2g))
    gx = np.reshape(gridx, [7,6])
    gy = np.reshape(gridy, [7,6])
    Mx = np.reshape(M, [7,6])
    Iy = np.reshape(I, [7,6])
    
    for i in range(gy.shape[0]):
        axes.plot(gx[i],gy[i], '-', color = 'grey', lw = 1.5)    
        axes.annotate(round(Mx[i][0],2), axes.lines[-1].get_xydata()[0])
    for j in range(gy.shape[1]):
        axes.plot(gx[:,j],gy[:,j], '-', color = 'grey', lw = 1.5)    
        axes.annotate(Iy[:,j][0], axes.lines[-1].get_xydata()[-1])
    
    #   Plot the extinction vector
    xpos = xlim[0] + 0.1*(xlim[1] - xlim[0])
    ypos = ylim[0]+ 0.9*(ylim[1] - ylim[0])
    axes.arrow(xpos, ypos, extinction_vector(wave, x1, x2), extinction_vector(wave, y1, y2), length_includes_head=True, color='green', head_width = 0.03, head_length = 0.02)
    
    
    axes.set_xlabel('log {}/{}'.format(x1.name, x2.name))
    axes.set_ylabel('log {}/{}'.format(y1.name, y2.name))
    axes.set_xlim(xlim[0], xlim[1])
    axes.set_ylim(ylim[0], ylim[1])
    
    #   To prevent showing the error bars in the legend
    handles, labels = axes.get_legend_handles_labels()
    new_handles = []
    for h in handles:
        if isinstance(h, container.ErrorbarContainer):
            new_handles.append(h[0])
        else:
            new_handles.append(h)
    axes.legend(new_handles, labels)
    
    fig.savefig('../Output/4. Plotting/Figures 4.8/{}_{}_{}_{}.png'.format(x1.name, x2.name, y1.name, y2.name), format='png')
    plt.close(fig)
    
    
ext_new(OIIdf, OIId, SIIdf, SIId, OT2, O2, ST2, S2, [-2.3,-1.2], [-1.7,-0.9])
ext_new(OIII, OIId, SIIdf, SIId, O3, O2, ST2, S2, [-2,0.9], [-1.7,-0.9])
ext_new(NIIf, NII, OIIIf, OIII, NT2, N2, OT3, O3, [-2.5,-1.2], [-3.5,-0.7])   
ext_new(NII, OIId, OIII, OIId, N2, O2, O3, O2, [-1.5,0.7], [-2,1])
ext_new(NII, OIId, NIIf, NII, N2, O2, NT2, N2, [-1.5,0.7], [-2.48,-1.2])
ext_new(NII, OIId, OIIIf, OIII, N2, O2, OT3, O3, [-1.5,0.7], [-3.5,-0.7])

ext_new_SOT(NII, OIId, OIId, OIIdf, SIId, SIIdf, N2, O2, O2, OT2, S2, ST2, [-2,1.5], [2.4,4.2])
ext_new_SOT(NII, H_alpha, OIId, OIIdf, SIId, SIIdf, N2, Ha, O2, OT2, S2, ST2, [-2,1.5], [2.4,4.2])
ext_new_SOT(NII, SIId, OIId, OIIdf, SIId, SIIdf, N2, S2, O2, OT2, S2, ST2, [-2,1.5], [2.4,4.2])

ext_new_SOT(SIId, SIIdf, OIId, OIIdf, SIId, SIIdf, S2, ST2, O2, OT2, S2, ST2, [0.9,1.7], [2.4,4.2])
ext_new_SOT(OIId, OIIdf, OIId, OIIdf, SIId, SIIdf, O2, OT2, O2, OT2, S2, ST2, [1.2,2.3], [2.4,4.2])
