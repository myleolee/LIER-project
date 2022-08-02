# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 01:32:56 2022

@author: Leo Lee

LIER project data analysis pipeline 
4. Graph Plotting - 4.3. Line ratio diagrams

Input: stacked_spectra.fits, 3 bins.fits, 3 bins_control.fits

Output: 
    3 Figures(.png):
        Figure 4.8: Extinction predicted from the Ha Vs Hb ratio with the reference of other Balmer lines
        Figures 4.9: Line ratio diagram of different elements against each other (Figures 4.9)
        Figures 4.10: Line ratio diagram of different elements against SOT (Figures 4.10)
"""
#%%
#   To control whether the figures will be generated
plot_4_8 = False
plot_4_9 = False
plot_4_10 = False

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

dir_array = ['../Output/4. Plotting/Figures 4.9', '../Output/4. Plotting/Figures 4.10']

for p in dir_array:
    check_dir(p)

#%%
#   Read the stacked spectra and spaxels data
spaxel_data_table = fits.open('../Data/spaxel_data_table.fits')

bin_1,bin_1_control = fits.open('../Data/Bin/Bin_1.fits'),fits.open('../Data/Bin/Bin_1_Control.fits')
bin_2,bin_2_control = fits.open('../Data/Bin/Bin_2.fits'),fits.open('../Data/Bin/Bin_2_Control.fits')
bin_3,bin_3_control = fits.open('../Data/Bin/Bin_3.fits'),fits.open('../Data/Bin/Bin_3_Control.fits')

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
#   Define the emission lines (Format: wavelength(s), [left window], [right window], 'name', 'type', 'ion')

#   1. Strong emission lines

#   Doublet with [OII]3729, vacuum wavelength = 3727.092, 3729.875
OIId = lier.doublet(3727.092, 3729.875, [3660, 3710], [3745, 3795], 'OII 3727, 3729', 'strong', 'O+')

#   Doublet with [NeIII] 3968, vacuum wavelength = 3869.86 (calculate only single line)
NeIII = lier.singlet(3869.86, [3805, 3855], [3900, 3950], 'NeIII 3869', 'strong', 'Ne++')
Ne3968 = lier.singlet(3968.59, [3948, 3964], [3978, 3994], 'Ne 3968', 'weak', 'Ne++')

#   Vacuum wavelength = 4862.721
H_beta = lier.singlet(4862.721, [4800, 4850], [4875, 4925], 'H beta', 'strong', 'H+')

#   Doublet with [OIII] 4960, but proportion locked by quantum mechanics, Vacuum wavelength = 5008.24
OIII = lier.singlet(5008.24, [4969, 4975], [5020, 5070], 'OIII 5007', 'strong', 'O++')

#   Doublet with [OI] 6365, but proportion locked by quantum mechanics, Vacuum wavelength = 6302.046
OI = lier.singlet(6302.046, [6238, 6288], [6315, 6355], 'OI 6300', 'strong', 'O')

#   Vacuum wavelength = 6564.6
H_alpha = lier.singlet(6564.6, [6485,6535], [6600,6650], 'H alpha', 'strong', 'H+')

#   Doublet with [NII] 6549, but proportion locked by quantum mechanics, so only include 6585, vacuum wavelength = 6549.86, 6585.27
NII = lier.singlet(6585.27, [6485, 6535], [6600, 6650], 'NII 6583', 'strong', 'N+')

#   Vacuum wavelength = 6718.295, 6732.674
SIId = lier.doublet(6718.295, 6732.674, [6685, 6705], [6745, 6795], 'SIId 6716, 6731', 'strong', 'S+')


#   2. Weak emission lines

#   Vacuum wavelength = 3890.166
H_zeta = lier.singlet(3890.166, [3880, 3885], [3895, 3900], 'H zeta', 'weak', 'H+')

#   Vacuum wavelegnth = 3971.198
H_epsilon = lier.singlet(3971.198, [3948, 3964], [3978, 3994], 'H epsilon', 'weak', 'H+')

#   Vacuum wavelength = 4069.75, 4077.5
SIIdf = lier.doublet(4069.75, 4077.5, [3980, 4060], [4110, 4190], 'SII 4068, 4076', 'weak', 'S+')

#   Vacuum wavelength = 4102.892
H_delta = lier.singlet(4102.892, [4089, 4098], [4108, 4116], 'H delta', 'weak', 'H+')

#   Vacuum wavelength = 4341.692
H_gamma = lier.singlet(4341.692, [4325, 4335], [4348, 4358], 'H gamma', 'weak', 'H+')

#   Vacuum wavelength = 4364.436
OIIIf = lier.singlet(4364.436, [4348, 4355], [4373, 4423], 'OIII 4363', 'weak', 'O++')

#   Vacuum wavelength = 5756.119
NIIf = lier.singlet(5756.119, [5665, 5745], [5766, 5846], 'NII 5755', 'weak', 'N+')

#   Inaccurate gaussian fitting, use custom integration range instead, Vacuum wavelength = 7321.94, 7332.21 
OIIdf = lier.doublet(7321.94, 7332.21, [7228, 7308], [7345, 7425], 'OII 7320, 7330', 'weak', 'O+')

#   Doublet with [SIII] 9533, but proportion locked by quantum mechanics, Vacuum wavelength = 9071.1
SIII = lier.singlet(9071.1, [8975, 9055], [9085, 9165], 'SIII 9071', 'weak', 'S++')


#   3. Array of all strong lines and weak lines
lines = [OIId, NeIII, H_beta, OIII, OI, H_alpha, NII, SIId, H_zeta, H_epsilon, SIIdf, H_delta, H_gamma, OIIIf, NIIf, OIIdf, SIII]
strong_lines = [OIId, NeIII, H_beta, OIII, OI, H_alpha, NII, SIId]
weak_lines = [H_zeta, H_epsilon, SIIdf, H_delta, H_gamma, OIIIf, NIIf, OIIdf, SIII]
hlines = [H_alpha, H_beta, H_gamma, H_delta, H_epsilon, H_zeta]

#%%
#   A function to get the coefficient of the corresponding voigt fit and gaussian fit

def get_p(flux, line, second = None, fit = 'voigt'):
    
    #   1. Get the continuum and reduce the spectrum by the fitline
        
    continuum = np.where(((wave > line.left[0]) & (wave < line.left[1])) | ((wave > line.right[0]) & (wave < line.right[1])))[0]
    coeff = np.polyfit(wave[continuum], flux[continuum], 1)
    fitline = wave*coeff[0] + coeff[1]
    reduced_spectrum = flux - fitline
        
    #   2. Fit the spectrum with a voigt profile return the coefficient   
    if (second == None):
        if (type(line) == lier.singlet):
            def voigt(x, *p):
                A, mean, sigma, gamma = p
                return A * np.real(wofz((x - mean + 1j*gamma)/sigma/np.sqrt(2))) / sigma /np.sqrt(2*np.pi)
            def gauss(x, *p):
                A, mean, sigma = p
                return A*np.exp(-(x-mean)**2/(2.*sigma**2))
            pg = [1, line.w, 1]
            pv = [1, line.w, 1, 1]
           
        elif (type(line) == lier.doublet):
            def voigt(x, *p):
                A, B, mean, sigma, gamma= p
                return A * np.real(wofz((x - mean + 1j*gamma)/sigma/np.sqrt(2))) / sigma /np.sqrt(2*np.pi) + B * np.real(wofz((x - (mean*line.w2/line.w1) + 1j*gamma)/sigma/np.sqrt(2))) / sigma /np.sqrt(2*np.pi)
            def gauss(x, *p):
                A, B, mean, sigma = p
                return A*np.exp(-(x-mean)**2/(2.*sigma**2)) + B*np.exp(-(x-(mean*line.w2/line.w1))**2/(2.*sigma**2))
            pg = [1, 1, line.w, 1]
            pv = [1, 1, line.w1, 1, 1]            

        else:
            print('Error: Input not lier.line')
            return
        
    elif (type(second) == lier.singlet):
        if (line.name == 'NII 6583'):
            NII_other = 6549.86
            def gauss(x, *p):
                NII_A, NII_mean, NII_sigma, Ha_A, Ha_mean, Ha_sigma = p
                return NII_A*np.exp(-(x-NII_mean)**2/(2.*NII_sigma**2)) + 0.3256*NII_A*np.exp(-(x-(NII_mean*NII_other/NII.w))**2/(2.*NII_sigma**2)) + Ha_A*np.exp(-(x-Ha_mean)**2/(2.*Ha_sigma**2))
            def voigt(x, *p):
                NII_A, NII_mean, NII_sigma, NII_gamma, Ha_A, Ha_mean, Ha_sigma, Ha_gamma = p
                return NII_A*np.real(wofz((x-NII_mean+1j*NII_gamma)/NII_sigma/np.sqrt(2)))/NII_sigma/np.sqrt(2*np.pi) + 0.3256*NII_A*np.real(wofz((x-(NII_mean*NII_other/NII.w)+1j*NII_gamma)/NII_sigma/np.sqrt(2)))/NII_sigma/np.sqrt(2*np.pi) + Ha_A*np.real(wofz((x-(Ha_mean)+1j*Ha_gamma)/Ha_sigma/np.sqrt(2)))/Ha_sigma/np.sqrt(2*np.pi)
            pg = [1, NII.w, 1, 1, second.w, 1]
            pv = [1, NII.w, 1, 1, 1, second.w, 1, 1]
                
        elif (type(line) == lier.singlet):
            def gauss(x, *p):
                 A, mean, sigma, B, mean_b, sigma_b = p
                 return A*np.exp(-(x-mean)**2/(2.*sigma**2)) + B*np.exp(-(x-mean_b)**2/(2.*sigma_b**2))
            def voigt(x, *p):
                A, mean, sigma, gamma, B, mean_b, sigma_b, gamma_b = p
                return A * np.real(wofz((x - mean + 1j*gamma)/sigma/np.sqrt(2))) / sigma /np.sqrt(2*np.pi) + B * np.real(wofz((x - mean_b + 1j*gamma_b)/sigma_b/np.sqrt(2))) / sigma_b /np.sqrt(2*np.pi)
            pg = [1, line.w, 1, 1, second.w, 1]
            pv = [1, line.w, 1, 1, 1, second.w, 1, 1]
        else:
            print('Error: Double line, single line input not yet supported')
            return
            
    else:
        print('Error: Input type of second line incorrect (must be singlet)')
        return
  
    if (fit == 'voigt'):
        coeff, var_matrix = curve_fit(voigt, wave, reduced_spectrum, p0=pv, method = 'lm')
    else:
        coeff, var_matrix = curve_fit(gauss, wave, reduced_spectrum, p0=pg, method = 'lm')
    return coeff

#%%
#   Define a function to get the voigt flux and gaussian flux

def voigt_flux(flux, line):
    if (line.name == 'H alpha'):
        coeff = get_p(flux, NII, H_alpha) 
        return coeff[4]
    elif (line.name == 'NII 6583'):
        coeff = get_p(flux, NII, H_alpha) 
        return coeff[0]
    elif (type(line) == lier.singlet):
        coeff = get_p(flux, line)
        return coeff[0]
    else:
        coeff = get_p(flux, line)
        return (coeff[0] + coeff[1])
    
def gauss_flux(flux, line):
    if (line.name == 'H alpha'):
        coeff = get_p(flux, NII, H_alpha, fit = 'gaussian') 
        return np.sqrt(2*np.pi)*coeff[3]*coeff[5]
    elif (line.name == 'NII 6583'):
        coeff = get_p(flux, NII, H_alpha, fit = 'gaussian') 
        return np.sqrt(2*np.pi)*coeff[0]*coeff[2]
    elif (line.name == 'H epsilon'):
        coeff = get_p(flux, H_epsilon, fit = 'gaussian') 
        coeff_Ne = get_p(flux, NeIII, fit = 'gaussian') 
        return np.sqrt(2*np.pi)*coeff[0]*coeff[2] - 0.31*(np.sqrt(2*np.pi)*coeff_Ne[0]*coeff_Ne[2])     # H epsilon is overlapping with Ne 3968, which is proportion to Ne 3869
    elif (type(line) == lier.singlet):
        coeff = get_p(flux, line, fit = 'gaussian')
        return np.sqrt(2*np.pi)*coeff[0]*coeff[2]
    else:
        coeff = get_p(flux, line, fit = 'gaussian')
        return np.sqrt(2*np.pi)*coeff[0]*coeff[3] + np.sqrt(2*np.pi)*coeff[1]*coeff[3]

#%%    #LSF related
#   Get the strong line FWHM from the voigt fit
def FWHM_strong(flux, line):
    t = 2*np.sqrt(2*np.log(2)) 
    if (line.name == 'H alpha'):
        coeff = get_p(flux, NII, H_alpha)
        fg = t*coeff[6]
        fl = 2*coeff[7]
    elif (line.name == 'NII 6583'):
        coeff = get_p(flux, NII, H_alpha)
        fg = t*coeff[2]
        fl = 2*coeff[3]
    elif (type(line)==lier.singlet):
        coeff = get_p(flux, line)
        fg = t*coeff[2] # The FWHM of the gaussian component
        fl = 2*coeff[3] # The FWHM of the lorenzian component
    else:
        coeff = get_p(flux, line)
        fg = t*coeff[3] # The FWHM of the gaussian component
        fl = 2*coeff[4] # The FWHM of the lorenzian component
    return 0.5346*fl + np.sqrt(0.2166*fl**2 + fg**2)


#   Get the spaxel numbers of the spaxels used in each metallicity bin
spaxels_used = []
def spaxel_data(bin_data, bin_control_data):
    result = []
    for i in range(len(bin_data)-1):
        for j in range(len(bin_data[i].data)):
            result.append(bin_data[i].data[j])
        for j in range(len(bin_control_data[i].data)):
            result.append(bin_control_data[i].data[j])
    spaxels_used.append(result)
spaxel_data(bin_1, bin_1_control)
spaxel_data(bin_2, bin_2_control)
spaxel_data(bin_2, bin_3_control)

#   Save the LSF data of each lines
LSF = {}
def get_LSF_from_data(l, name):
     # This are the LSFs expressed in terms of km s-1
    kms_high = round(np.median(spaxel_data_table[1].data[name][spaxels_used[0]]),3)  
    kms_mid = round(np.median(spaxel_data_table[1].data[name][spaxels_used[1]]),3)   
    kms_low = round(np.median(spaxel_data_table[1].data[name][spaxels_used[2]]),3)   
    output = np.array([kms_high, kms_mid, kms_low])
    return 2.354*output*l.w/c_vel   # 2.354 as the FWHM of gaussian is 2*sqrt(2*ln2)*sd, 2*sqrt(2*ln2) = 2.354

#   Note that the weak hydrogen lines LSF are from lines nearby
LSF[OIId.name] = get_LSF_from_data(OIId, 'OIId-3727_LSF')
LSF[NeIII.name] = get_LSF_from_data(NeIII, 'OIId-3727_LSF')
LSF[H_beta.name] = get_LSF_from_data(H_beta, 'H_beta_LSF')
LSF[OIII.name] = get_LSF_from_data(OIII,'OIII-5007_LSF')
LSF[OI.name] = get_LSF_from_data(OI, 'OI-6300_LSF')
LSF[H_alpha.name] = get_LSF_from_data(H_alpha, 'H_alpha_LSF')
LSF[NII.name] = get_LSF_from_data(NII, 'NII-6583_LSF')
LSF[SIId.name] = get_LSF_from_data(SIId, 'SIId-6716_LSF')
LSF[H_zeta.name] = get_LSF_from_data(H_zeta, 'SIId-4068_LSF')
LSF[H_epsilon.name] = get_LSF_from_data(H_epsilon, 'SIId-4068_LSF')
LSF[SIIdf.name] = get_LSF_from_data(SIIdf, 'SIId-4068_LSF')
LSF[H_delta.name] = get_LSF_from_data(H_delta,'SIId-4068_LSF')
LSF[H_gamma.name] = get_LSF_from_data(H_gamma, 'OIII-4363_LSF')
LSF[OIIIf.name] = get_LSF_from_data(OIIIf, 'OIII-4363_LSF')
LSF[NIIf.name] = get_LSF_from_data(NIIf, 'NII-5755_LSF')
LSF[OIIdf.name] = get_LSF_from_data(OIIdf, 'OIId-7320_LSF')
LSF[SIII.name] = get_LSF_from_data(SIII, 'SIIId-9071_LSF')

#   From the strong line, get the velocity of the ions in the three metallicity bin
vel = {} 
m_bins = [high, mid, low]
for l in strong_lines:
    if (l.name == 'H beta'):
        continue
    elif (l.name == 'H alpha'):
        l2 = H_beta
        FWHM_vel = np.zeros(len(m_bins))
        FWHM_vel2 = np.zeros(len(m_bins))
        for i in range(len(m_bins)):
            FWHM_vel[i] = np.sqrt(FWHM_strong(m_bins[i], l)**2 - LSF[l.name][i]**2)
            FWHM_vel2[i] = np.sqrt(FWHM_strong(m_bins[i], l2)**2 - LSF[l2.name][i]**2)
        vel[l.ion] = ((FWHM_vel/l.w)*c_vel + (FWHM_vel2/l2.w)*c_vel)/2      # Average of the velocity calculate from H alpha and H beta
    else:
        FWHM_vel = np.zeros(len(m_bins))
        for i in range(len(m_bins)):
            LSF_FWHM = 2.354*LSF[l.name][i]*l.w/c_vel
            FWHM_vel[i] = np.sqrt(FWHM_strong(m_bins[i], l)**2 - LSF[l.name][i]**2)   # 2.354 as the FWHM of gaussian is 2*sqrt(2*ln2)*sd, 2*sqrt(2*ln2) = 2.354
        vel[l.ion] = (FWHM_vel/l.w)*c_vel
            
vel['S++'] = vel['S+']      # For SIII 9071

#   Define an overall function for all of the lines

def FWHM(flux, line):
    m = 0 
    if (np.equal(flux, high).all()):
        m = 0
    elif (np.equal(flux, mid).all()):
        m = 1
    elif (np.equal(flux, low).all()):
        m = 2
    else:
        return 'error'
    FWHM = np.sqrt( (vel[line.ion][m] * line.w / c_vel)**2 + LSF[line.name][m]**2)
    return FWHM

#%%
#   Define the function to calculate the summed flux of the lines and plot the summed flux of the line(Figures 4.7)

def summed_flux(flux, line, FWHM, graph = False, path=''):
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
                if ((line.name == 'H alpha') | (line.name == 'NII 6583')):
                    coeff = get_p(flux, NII, H_alpha)
                    plt.axvline(line.w, color = 'k', linestyle = '--')
                    NII_other = 6549.86
                    voi = coeff[0]*np.real(wofz((wave-coeff[1]+1j*coeff[3])/coeff[2]/np.sqrt(2)))/coeff[2]/np.sqrt(2*np.pi) + 0.3256*coeff[0]*np.real(wofz((wave-(coeff[1]*NII_other/NII.w)+1j*coeff[3])/coeff[2]/np.sqrt(2)))/coeff[2]/np.sqrt(2*np.pi) + coeff[4]*np.real(wofz((wave-(coeff[5])+1j*coeff[7])/coeff[6]/np.sqrt(2)))/coeff[6]/np.sqrt(2*np.pi)
                    n = coeff[0]*np.real(wofz((wave-coeff[1]+1j*coeff[3])/coeff[2]/np.sqrt(2)))/coeff[2]/np.sqrt(2*np.pi) + 0.3256*coeff[0]*np.real(wofz((wave-(coeff[1]*NII_other/NII.w)+1j*coeff[3])/coeff[2]/np.sqrt(2)))/coeff[2]/np.sqrt(2*np.pi)
                    ha = coeff[4]*np.real(wofz((wave-(coeff[5])+1j*coeff[7])/coeff[6]/np.sqrt(2)))/coeff[6]/np.sqrt(2*np.pi)
                    plt.plot(wave, n, color = 'purple', label = 'NII')
                    plt.plot(wave, ha, color = 'green', label = 'Ha')
                elif (type(line) == lier.singlet):
                    coeff = get_p(flux, line)
                    plt.axvline(line.w, color = 'k', linestyle = '--')
                    voi = coeff[0] * np.real(wofz((wave - coeff[1] + 1j*coeff[3])/coeff[2]/np.sqrt(2))) / coeff[2] /np.sqrt(2*np.pi)
                else:    
                    coeff = get_p(flux, line)
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
    
#%%
#   The variance propagated from the spaxels file
def variance(flux, var, line):
    temp = FWHM(flux, line)
    if (type(line) == lier.singlet):
        centre = [line.w - 1.5*temp, line.w + 1.5*temp]
    else:
        centre = [line.w1 - 1.5*temp, line.w2 + 1.5*temp]
    centre_region = np.where(((wave >= centre[0]) & (wave <= centre[1])))[0]
    delta_x = np.median(wave[centre_region] - wave[centre_region-1])
    sflux = np.trapz(var[centre_region], wave[centre_region]) * delta_x
    return sflux

#   Sliding box for uncertainty

def sliding_box(flux, line):
    output = []
    
    #   Get the FWHM size of and check how many sliding box will be used
    temp = 3*FWHM(flux, line)
    l_index = np.where((wave>line.left[0]) & (wave<line.left[1]))[0]
    r_index = np.where((wave>line.right[0]) & (wave<line.right[1]))[0]
    delta = wave[r_index[-1]] - wave[r_index[-2]]
    d_spaxels = int((temp//delta)+1)
    
    for i in range(len(l_index)-d_spaxels):
        error_sum = np.trapz(flux[l_index[i:i+d_spaxels]], wave[l_index[i:i+d_spaxels]])
        output.append(error_sum)
    
    #   This is the middle part of the two sidebands
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

#%%
#   Find the extinction derived from the Balmer decrement and plot the other Balmer lines with respect to the one from Ha (Figure 4.8)

ref_h_ratio = [2.85, 1, 0.469, 0.26, 0.159, 0.105, 0.0786]      # The reference value of balmer line ratio from Osterbrock

#   Plot a table showing the observed ratio of Balmer lines and Hb, and the reference value

def table_h_ratio():    
    row1 = ['Bin', 'High', 'Mid', 'Low', 'Ref']
    row2 = ['H alpha']
    row3 = ['H beta']
    row4 = ['H gamma']
    row5= ['H delta']
    row6= ['H epsilon']
    row7 = ['H zeta']
    temp = [row2, row3, row4, row5, row6, row7]
    for i in range(len(hlines)):
        temp[i].append(round((gauss_flux(high, hlines[i])/gauss_flux(high, H_beta)),3))
        temp[i].append(round((gauss_flux(mid, hlines[i])/gauss_flux(mid, H_beta)),3))
        temp[i].append(round((gauss_flux(low, hlines[i])/gauss_flux(low, H_beta)),3))
        temp[i].append(ref_h_ratio[i])
    return [row1, row2, row3, row4, row5, row6, row7]

print(tabulate(table_h_ratio(), headers='firstrow'), '\n')

#   A function to get the Av calculated from the expected ratio (value) and the gaussian flux of the hline

def cal_Av(flux, hline, value):
    r = gauss_flux(flux, hline)/gauss_flux(flux, H_beta)
    d_A = -2.5*np.log10(r/value)
    Al_Av = ext.fitzpatrick99(np.array([hline.w]),1)[0] - ext.fitzpatrick99(np.array([H_beta.w]),1)[0]
    return d_A/Al_Av

#   Plot the Balmer lines ratio with hb after applying the Av calculated from from Ha/Hb ratio (Figure 4.8)
def balmer_all(wave, high, mid, low):    
    fig, [axes1, axes2, axes3] = plt.subplots(3 , 1 ,figsize = [7, 18])
    def balmer_decrement(wave, flux, label, axes):
        Av = cal_Av(flux, H_alpha, 2.85)
        h_ind = []
        real = []
        error = []
        extinct = 10**(-0.4*ext.fitzpatrick99(wave, Av))
        ind_b = np.where(wave > H_beta.w)[0][0]
        hb_ext = extinct[ind_b]
        for i in range(len(hlines)):
            h_ind.append(np.where(wave > hlines[i].w)[0][0])
            real.append(gauss_flux(flux, hlines[i])/gauss_flux(flux, H_beta))
            if (hlines[i].name == 'H beta'):
                error.append(sliding_box(flux, H_beta)[1])
            elif (hlines[i].name == 'H epsilon'):
                ne_err = sliding_box(flux, NeIII)[1] * 0.31
                h_err = sliding_box(flux, H_epsilon)[1] + ne_err
                h_flux = gauss_flux(flux, H_epsilon)
                error.append(np.sqrt((h_err/h_flux)**2 + (sliding_box(flux, H_beta)[1]/gauss_flux(flux, H_beta))**2)*real[-1]/ref_h_ratio[i])
            else:
                error.append(np.sqrt((sliding_box(flux, hlines[i])[1]/gauss_flux(flux, hlines[i]))**2 + (sliding_box(flux, H_beta)[1]/gauss_flux(flux, H_beta))**2)*real[-1]/ref_h_ratio[i])
        axes.plot(wave, extinct/hb_ext, color = 'k')
        axes.axhline(y = 1, color = 'k', linestyle = '--')
        for i in range(len(h_ind)):
            axes.errorbar(wave[h_ind[i]], real[i]/ref_h_ratio[i], yerr = error[i], fmt = 'r.')
        axes.set_xlim(3500, 7000)
        axes.set_ylim(0.3,1.7)
        axes.set_title(label)
    
    balmer_decrement(wave, high, 'High', axes1)
    balmer_decrement(wave, mid, 'Mid', axes2)
    balmer_decrement(wave, low, 'Low', axes3)
    fig.savefig('../Output/4. Plotting/Figures 4.8.png', format='png', dpi = 1200)

if (plot_4_8 == True):
    balmer_all(wave, high, mid ,low)

#%%
#   Define functions for plotting 


#   The data points in Renbin's 2018 paper

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

#   The model grid from CLOUDY, which the ionization source are old stars
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

#   Get the extinction vector of the diagram
def extinction_vector(line1, line2):
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

#   Fuction for plotting upper bound limit (2 sigma limit)
def plot_arrows(flux, x1, x2, y1, y2):
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

#   Function for getting the line ratio under a certain Av
def log_ratio(flux, line_x, line_y, Av = 0):
    if (Av != 0):
        x_index = np.where(line_x.w < wave)[0][0]
        x_flux = summed_flux(flux, line_x, FWHM(flux, line_x)) / (10**(-0.4*ext.fitzpatrick99(wave, Av)))[x_index] 
        y_index = np.where(line_y.w < wave)[0][0]
        y_flux = summed_flux(flux, line_y, FWHM(flux, line_y)) / (10**(-0.4*ext.fitzpatrick99(wave, Av)))[y_index]
        ratio = x_flux/y_flux
    else:
        ratio = summed_flux(flux, line_x, FWHM(flux, line_x)) / summed_flux(flux, line_y, FWHM(flux, line_y))    
    return np.log10(ratio)
#%%
#   Plot the line ratio diagram of different elements (Figures 4.9)

def plot_log_line_ratio(x1, x2, y1, y2, x1g, x2g, y1g, y2g, xlim, ylim):
    
    fig, axes = plt.subplots(figsize=(7,7))

    #   Plot the error bars of the three points (If y = a/c, dy^2 = ((da/a)^2 + (dc/c)^2) y^2), d (log10x) = 1/ln10 * dx / x 
    def plot_error_bars(flux, var, x1, x2, y1, y2, form, form_corr, ecolor, label):
        ln10 = np.log(10)
        
        x1_flux = summed_flux(flux, x1, FWHM(flux, x1))
        x2_flux = summed_flux(flux, x2, FWHM(flux, x2))
        x1_sd = sliding_box(flux, x1)[1]
        x2_sd = sliding_box(flux, x2)[1]
        x_frac_err = np.sqrt((x1_sd/x1_flux)**2 + (x2_sd/x2_flux)**2)
        log_x_err = x_frac_err / ln10
        x = log_ratio(flux, x1, x2)
        y1_flux = summed_flux(flux, y1, FWHM(flux, y1))
        y2_flux = summed_flux(flux, y2, FWHM(flux, y2))
        y1_sd = sliding_box(flux, y1)[1]
        y2_sd = sliding_box(flux, y2)[1]
        y_frac_err = np.sqrt((y1_sd/y1_flux)**2 + (y2_sd/y2_flux)**2)
        log_y_err = y_frac_err / ln10
        y = log_ratio(flux, y1, y2)
       
        arrows = plot_arrows(flux, x1, x2, y1, y2)
        
        Av = cal_Av(flux, H_alpha, 2.85)
        ext_arr = ext.fitzpatrick99(wave, Av)
        
        if (arrows[0] == -1):
            x1_flux = 2 * sliding_box(flux, x1)[1]
            x = np.log10(x1_flux/summed_flux(flux, x2, FWHM(flux, x2)))
        elif (arrows[0] == 1):
            x2_flux = 2 * sliding_box(flux, x2)[1]
            x = np.log10(summed_flux(flux, x1, FWHM(flux, x1))/x2_flux)
        x1_index = np.where(x1.w < wave)[0][0]
        x2_index = np.where(x2.w < wave)[0][0]
        x_corr = x - 0.4 * (ext_arr[x2_index] - ext_arr[x1_index])
        
        if (arrows[1] == -1):
            y1_flux = 2 * sliding_box(flux, y1)[1]
            y = np.log10(y1_flux/summed_flux(flux, y2, FWHM(flux, y2)))
        elif (arrows[1] == 1):
            y2_flux = 2 * sliding_box(flux, y2)[1]
            y = np.log10(summed_flux(flux, y1, FWHM(flux, y1))/y2_flux)
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
        
       
    plot_error_bars(high, high_var, x1, x2, y1, y2, '.r','r^' , 'r', 'High')
    plot_error_bars(mid, mid_var, x1, x2, y1, y2, '.b','b^', 'b', 'Mid')
    plot_error_bars(low, low_var, x1, x2, y1, y2, '.k','k^', 'k', 'Low')
    
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
    axes.arrow(xpos, ypos, extinction_vector(x1, x2), extinction_vector(y1, y2), length_includes_head=True, color='green', head_width = 0.03, head_length = 0.02)
    
    
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
    
    fig.savefig('../Output/4. Plotting/Figures 4.9/{}_{}_{}_{}.png'.format(x1.name, x2.name, y1.name, y2.name), format='png')
    plt.close(fig)
    
if (plot_4_9 == True):        
    plot_log_line_ratio(OIIdf, OIId, SIIdf, SIId, OT2, O2, ST2, S2, [-2.3,-1.2], [-1.7,-0.9])
    plot_log_line_ratio(OIII, OIId, SIIdf, SIId, O3, O2, ST2, S2, [-2,0.9], [-1.7,-0.9])
    plot_log_line_ratio(NIIf, NII, OIIIf, OIII, NT2, N2, OT3, O3, [-2.5,-1.2], [-3.5,-0.7])   
    plot_log_line_ratio(NII, OIId, OIII, OIId, N2, O2, O3, O2, [-1.5,0.7], [-2,1])
    plot_log_line_ratio(NII, OIId, NIIf, NII, N2, O2, NT2, N2, [-1.5,0.7], [-2.48,-1.2])
    plot_log_line_ratio(NII, OIId, OIIIf, OIII, N2, O2, OT3, O3, [-1.5,0.7], [-3.5,-0.7])

#%%
#    Plot the line ratio diagram of different elements against SOT (Figures 4.10)

def plot_log_ratio_SOT(x1, x2, SOT1_up, SOT1_low, SOT2_up, SOT2_low, x1g, x2g, SOT1_upg, SOT1_lowg, SOT2_upg, SOT2_lowg, xlim, ylim):
    fig, axes = plt.subplots(figsize=(7,7))
    
    def plot_error_bars_SOT(flux, var, x1, x2, SOT1_up, SOT1_low, SOT2_up, SOT2_low, form, form_corr, ecolor, label):
        ln10 = np.log(10)
        x1_flux = summed_flux(flux, x1, FWHM(flux, x1))
        x2_flux = summed_flux(flux, x2, FWHM(flux, x2))
        x1_sd = sliding_box(flux, x1)[1]
        x2_sd = sliding_box(flux, x2)[1]
        x_frac_err = np.sqrt((x1_sd/x1_flux)**2 + (x2_sd/x2_flux)**2)
        log_x_err = x_frac_err / ln10
        x = log_ratio(flux, x1, x2)
        
        SOT1_up_flux = summed_flux(flux, SOT1_up, FWHM(flux, SOT1_up))
        SOT1_low_flux = summed_flux(flux, SOT1_low, FWHM(flux, SOT1_low))
        SOT2_up_flux = summed_flux(flux, SOT2_up, FWHM(flux, SOT2_up))
        SOT2_low_flux = summed_flux(flux, SOT2_low, FWHM(flux, SOT2_low))
        SOT1_up_sd = sliding_box(flux, SOT1_up)[1]
        SOT1_low_sd = sliding_box(flux, SOT1_low)[1]
        SOT2_up_sd = sliding_box(flux, SOT2_up)[1]
        SOT2_low_sd = sliding_box(flux, SOT2_low)[1]
        SOT = log_ratio(flux, SOT1_up, SOT1_low) + 1.3 * log_ratio(flux, SOT2_up, SOT2_low)
        SOT1_frac_err = np.sqrt((SOT1_up_sd/SOT1_up_flux)**2 + (SOT1_low_sd/SOT1_low_flux)**2)
        SOT2_frac_err = np.sqrt((SOT2_up_sd/SOT2_up_flux)**2 + (SOT2_low_sd/SOT2_low_flux)**2)
        log_SOT1_err = SOT1_frac_err / ln10
        log_SOT2_err = SOT2_frac_err / ln10
        log_SOT_err = log_SOT1_err + 1.3*log_SOT2_err
        
        Av = cal_Av(flux, H_alpha, 2.85)
        x_corr = log_ratio(flux, x1, x2, Av)
        SOT_corr = log_ratio(flux, SOT1_up, SOT1_low, Av) + 1.3 * log_ratio(flux, SOT2_up, SOT2_low, Av)
        
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
        
    plot_error_bars_SOT(high, high_var, x1, x2, SOT1_up, SOT1_low, SOT2_up, SOT2_low, 'r.','r^', 'r', 'High')
    plot_error_bars_SOT(mid, mid_var, x1, x2, SOT1_up, SOT1_low, SOT2_up, SOT2_low, 'b.','b^', 'b', 'Mid')
    plot_error_bars_SOT(low, low_var, x1, x2, SOT1_up, SOT1_low, SOT2_up, SOT2_low, 'k.','k^', 'k', 'Low')
        
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
    axes.arrow(xpos, ypos, extinction_vector(x1, x2), extinction_vector(SOT1_up, SOT1_low) + 1.3*extinction_vector(SOT2_up, SOT2_low), length_includes_head=True, color='green', head_width = 0.03, head_length = 0.02)
    
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
    fig.savefig('../Output/4. Plotting/Figures 4.10/{}_{}_SOT.png'.format(x1.name, x2.name), format='png')  
    plt.close(fig)

if (plot_4_10 == True):    
    plot_log_ratio_SOT(NII, OIId, OIId, OIIdf, SIId, SIIdf, N2, O2, O2, OT2, S2, ST2, [-2,1.5], [2.4,4.2])
    plot_log_ratio_SOT(NII, H_alpha, OIId, OIIdf, SIId, SIIdf, N2, Ha, O2, OT2, S2, ST2, [-2,1.5], [2.4,4.2])
    plot_log_ratio_SOT(NII, SIId, OIId, OIIdf, SIId, SIIdf, N2, S2, O2, OT2, S2, ST2, [-2,1.5], [2.4,4.2])
    plot_log_ratio_SOT(SIId, SIIdf, OIId, OIIdf, SIId, SIIdf, S2, ST2, O2, OT2, S2, ST2, [0.9,1.7], [2.4,4.2])
    plot_log_ratio_SOT(OIId, OIIdf, OIId, OIIdf, SIId, SIIdf, O2, OT2, O2, OT2, S2, ST2, [1.2,2.3], [2.4,4.2])
