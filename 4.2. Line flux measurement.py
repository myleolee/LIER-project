# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 18:36:19 2022

@author: Leo Lee

LIER project data analysis pipeline 
4. Graph Plotting - 4.2. Line flux measurement

Input: stacked_spectra.fits, 3 bins.fits, 3 bins_control.fits

Output: 
    3 Figures(.png):
        Figures 4.4: The spectrum and continuum around each line in the three metallicity bin
        Figures 4.5: Overplots of the three profile around strong lines and hydrogen lines
        Figures 4.6: Plots of component voigts for doublets (and H epsilon)
        Figures 4.7: Plots of the integrated region of each lines for summed flux
"""
#%%
#   To control whether the figures will be generated
plot_4_4 = False
plot_4_5 = False
plot_4_6 = False
plot_4_7 = False

#%%
#   Importing required packages

from tabulate import tabulate
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
import os
import lier
from scipy.special import wofz
from scipy.optimize import curve_fit
import astropy.constants
    
font = { 'size' : 10 , 'family' : 'serif'}  # size of normal font
fs = 14  # size of titles, must be manually inserted using ',fontsize=fs'
rc('font', **font)

c_vel = astropy.constants.c.to('km/s').value
#%%
#   Check whether the directories exist, if not create the directories
def check_dir(path):
   if (os.path.exists(path) == False):
       os.makedirs(path) 

dir_array = ['../Output/4. Plotting/Figures 4.4', '../Output/4. Plotting/Figures 4.5', '../Output/4. Plotting/Figures 4.5/high',
             '../Output/4. Plotting/Figures 4.5/mid', '../Output/4. Plotting/Figures 4.5/low', '../Output/4. Plotting/Figures 4.6',
             '../Output/4. Plotting/Figures 4.6/high', '../Output/4. Plotting/Figures 4.6/mid', '../Output/4. Plotting/Figures 4.6/low', 
             '../Output/4. Plotting/Figures 4.7','../Output/4. Plotting/Figures 4.7/high', '../Output/4. Plotting/Figures 4.7/mid', 
             '../Output/4. Plotting/Figures 4.7/low']

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
#   Plots showing the continuum selection (Figures 4.4)

def envir_plot(line):
    axes = [] 
    fig, axes = plt.subplots(3,1, figsize=(19,9))
    axes[0].plot(wave, high)
    axes[1].plot(wave, mid)
    axes[2].plot(wave, low)
    for ax in axes:
        ax.axhline(y = 0, color = 'r', linestyle = '-')
        ax.set_xlim((line.w - 200, line.w + 200))
        ax.axvspan(line.left[0], line.left[1], color = 'red', alpha=0.1, linestyle = '-', label = 'continuum')
        ax.axvspan(line.right[0], line.right[1], color = 'red', alpha=0.1, linestyle = '-')
        if (line.type == 'weak'):
            ax.set_ylim([-0.01, 0.05])
        elif (line.name == 'OII 3727, 3729'):
            ax.set_ylim([-0.05, 0.4])
        else:
            ax.set_ylim([-0.03, 0.2])
        for l in lines:
            if ((l.w > ax.get_xlim()[0]) and (l.w < ax.get_xlim()[1])):
                if (line.type == 'weak'):
                    ax.annotate(l.name, (l.w-5, -0.0075))
                elif (line.name == 'OII 3727, 3729'):
                    ax.annotate(l.name, (l.w-5, -0.04))
                else:
                    ax.annotate(l.name, (l.w-5, -0.02))
    fig.tight_layout()                    
    fig.savefig('../output/4. Plotting/Figures 4.4/{}.png'.format(line.name), format='png', dpi = 1200)
    plt.close(fig)

if (plot_4_4 == True):
    for l in lines:
        envir_plot(l)


#%%
#   Overplots different profile fitting every emission line (Figures 4.5)
    
#   Function for fitting and overplotting the graph
def overplot(line, second = None):
    
    def overplot_bin(flux, flux_name, line, second = None):
        
        #   1. Get the continuum and reduce the spectrum by the fitline
        
        continuum = np.where(((wave > line.left[0]) & (wave < line.left[1])) | ((wave > line.right[0]) & (wave < line.right[1])))[0]
        coeff = np.polyfit(wave[continuum], flux[continuum], 1)
        fitline = wave*coeff[0] + coeff[1]
        reduced_spectrum = flux - fitline
        
        #   2. Fit the reduced spectrum with the three distributions 
        if (second == None):
            if (type(line) == lier.singlet):
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
                coeff, var_matrix = curve_fit(gauss, wave, reduced_spectrum, p0=pg, method = 'lm')
                gau = coeff[0]*np.exp(-(wave-coeff[1])**2/(2.*coeff[2]**2))  
                coeff, var_matrix = curve_fit(lorentz, wave, reduced_spectrum, p0=pl, method = 'lm')
                lor = coeff[0]*((0.5*coeff[2])/((wave-coeff[1])**2+(0.5*coeff[2])**2))       
                coeff, var_matrix = curve_fit(voigt, wave, reduced_spectrum, p0=pv, method = 'lm')
                voi = coeff[0] * np.real(wofz((wave - coeff[1] + 1j*coeff[3])/coeff[2]/np.sqrt(2))) / coeff[2] /np.sqrt(2*np.pi)
           
            elif (type(line) == lier.doublet):
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
                coeff, var_matrix = curve_fit(gauss, wave, reduced_spectrum, p0=pg, method = 'lm')
                gau = coeff[0]*np.exp(-(wave-coeff[2])**2/(2.*coeff[3]**2)) + coeff[1]*np.exp(-(wave-(coeff[2]*line.w2/line.w1))**2/(2.*coeff[3]**2))    
                coeff, var_matrix = curve_fit(lorentz, wave, reduced_spectrum, p0=pl, method = 'lm')
                lor = coeff[0]*((0.5*coeff[3])/((wave-coeff[2])**2+(0.5*coeff[3])**2)) + coeff[1]*((0.5*coeff[3])/((wave-(coeff[2]*line.w2/line.w1))**2+(0.5*coeff[3])**2))        
                coeff, var_matrix = curve_fit(voigt, wave, reduced_spectrum, p0=pv, method = 'lm')
                voi = coeff[0] * np.real(wofz((wave - coeff[2] + 1j*coeff[4])/coeff[3]/np.sqrt(2))) / coeff[3] /np.sqrt(2*np.pi) + (coeff[1] * np.real(wofz((wave - (coeff[2]*line.w2/line.w1) + 1j*coeff[4])/coeff[3]/np.sqrt(2))) / coeff[3] /np.sqrt(2*np.pi))    
            
            else:
                print('Error: Input not lier.line')
                return
        
        elif (type(second) == lier.singlet):
            if (line.name == 'NII 6583'):
                NII_other = 6549.86
                def gauss(x, *p):
                    NII_A, NII_mean, NII_sigma, Ha_A, Ha_mean, Ha_sigma = p
                    return NII_A*np.exp(-(x-NII_mean)**2/(2.*NII_sigma**2)) + 0.3256*NII_A*np.exp(-(x-(NII_mean*NII_other/NII.w))**2/(2.*NII_sigma**2)) + Ha_A*np.exp(-(x-Ha_mean)**2/(2.*Ha_sigma**2))
                def lorentz(x, *p):
                    NII_A, NII_mean, NII_gamma, Ha_A,  Ha_mean, Ha_gamma = p
                    return NII_A*((0.5*NII_gamma)/((x-NII_mean)**2+(0.5*NII_gamma)**2)) + 0.3256*NII_A*((0.5*NII_gamma)/((x-(NII_mean*NII_other/NII.w))**2+(0.5*NII_gamma)**2)) + Ha_A*((0.5*Ha_gamma)/((x-Ha_mean)**2+(0.5*Ha_gamma)**2))
                def voigt(x, *p):
                    NII_A, NII_mean, NII_sigma, NII_gamma, Ha_A, Ha_mean, Ha_sigma, Ha_gamma = p
                    return NII_A*np.real(wofz((x-NII_mean+1j*NII_gamma)/NII_sigma/np.sqrt(2)))/NII_sigma/np.sqrt(2*np.pi) + 0.3256*NII_A*np.real(wofz((x-(NII_mean*NII_other/NII.w)+1j*NII_gamma)/NII_sigma/np.sqrt(2)))/NII_sigma/np.sqrt(2*np.pi) + Ha_A*np.real(wofz((x-(Ha_mean)+1j*Ha_gamma)/Ha_sigma/np.sqrt(2)))/Ha_sigma/np.sqrt(2*np.pi)
                pg = [1, NII.w, 1, 1, second.w, 1]
                pl = [1, NII.w, 2.5, 1, second.w, 2.5]
                pv = [1, NII.w, 1, 1, 1, second.w, 1, 1]
                coeff, var_matrix = curve_fit(gauss, wave, reduced_spectrum, p0=pg, method = 'lm')
                gau = coeff[0]*np.exp(-(wave-coeff[1])**2/(2.*coeff[2]**2)) + 0.3256*coeff[0]*np.exp(-(wave-(coeff[1]*NII_other/NII.w))**2/(2.*coeff[2]**2)) + coeff[3]*np.exp(-(wave-coeff[4])**2/(2.*coeff[5]**2))    
                coeff, var_matrix = curve_fit(lorentz, wave, reduced_spectrum, p0=pl, method = 'lm')
                lor = coeff[0]*((0.5*coeff[2])/((wave-coeff[1])**2+(0.5*coeff[2])**2)) + 0.3256*coeff[0]*((0.5*coeff[2])/((wave-(coeff[1]*NII_other/NII.w))**2+(0.5*coeff[2])**2)) + coeff[3]*((0.5*coeff[5])/((wave-coeff[4])**2+(0.5*coeff[5])**2))   
                coeff, var_matrix = curve_fit(voigt, wave, reduced_spectrum, p0=pv, method = 'lm')
                voi = coeff[0]*np.real(wofz((wave-coeff[1]+1j*coeff[3])/coeff[2]/np.sqrt(2)))/coeff[2]/np.sqrt(2*np.pi) + 0.3256*coeff[0]*np.real(wofz((wave-(coeff[1]*NII_other/NII.w)+1j*coeff[3])/coeff[2]/np.sqrt(2)))/coeff[2]/np.sqrt(2*np.pi) + coeff[4]*np.real(wofz((wave-(coeff[5])+1j*coeff[7])/coeff[6]/np.sqrt(2)))/coeff[6]/np.sqrt(2*np.pi)
                
            elif (type(line) == lier.singlet):
                def gauss(x, *p):
                    A, mean, sigma, B, mean_b, sigma_b = p
                    return A*np.exp(-(x-mean)**2/(2.*sigma**2)) + B*np.exp(-(x-mean_b)**2/(2.*sigma_b**2))
                def lorentz(x, *p):
                    A, mean, gamma, B, mean_b, gamma_b = p
                    return A*((0.5*gamma)/((x-mean)**2+(0.5*gamma)**2)) + B*((0.5*gamma_b)/((x-mean_b)**2+(0.5*gamma_b)**2))
                def voigt(x, *p):
                    A, mean, sigma, gamma, B, mean_b, sigma_b, gamma_b = p
                    return A * np.real(wofz((x - mean + 1j*gamma)/sigma/np.sqrt(2))) / sigma /np.sqrt(2*np.pi) + B * np.real(wofz((x - mean_b + 1j*gamma_b)/sigma_b/np.sqrt(2))) / sigma_b /np.sqrt(2*np.pi)
                pg = [1, line.w, 1, 1, second.w, 1]
                pl = [1, line.w, 2.5, 1, second.w, 2.5]
                pv = [1, line.w, 1, 1, 1, second.w, 1, 1]
                coeff, var_matrix = curve_fit(gauss, wave, reduced_spectrum, p0=pg, method = 'lm')
                gau = coeff[0]*np.exp(-(wave-coeff[1])**2/(2.*coeff[2]**2)) + coeff[3]*np.exp(-(wave-coeff[4])**2/(2.*coeff[5]**2))  
                coeff, var_matrix = curve_fit(lorentz, wave, reduced_spectrum, p0=pl, method = 'lm')
                lor = coeff[0]*((0.5*coeff[2])/((wave-coeff[1])**2+(0.5*coeff[2])**2)) + coeff[3]*((0.5*coeff[5])/((wave-coeff[4])**2+(0.5*coeff[5])**2))              
                coeff, var_matrix = curve_fit(voigt, wave, reduced_spectrum, p0=pv, method = 'lm')
                voi = coeff[0] * np.real(wofz((wave - coeff[1] + 1j*coeff[3])/coeff[2]/np.sqrt(2))) / coeff[2] /np.sqrt(2*np.pi) + coeff[4] * np.real(wofz((wave - coeff[5] + 1j*coeff[7])/coeff[6]/np.sqrt(2))) / coeff[6] /np.sqrt(2*np.pi)
            else:
                print('Error: Double line, single line input not yet supported')
                return
            
        else:
            print('Error: Input type of second line incorrect (must be singlet)')
            return
        
        #   3. Overplot the line fitted and the spectrum
        
        fig, ax = plt.subplots(1, 1, figsize = (7,5))
        ax.axvspan(line.left[0], line.left[1], color = 'red', alpha=0.1, linestyle = '-', label = 'continuum')
        ax.axvspan(line.right[0], line.right[1], color = 'red', alpha=0.1, linestyle = '-')
        if (type(line) == lier.singlet):
            ax.axvline(line.w, color = 'k', linestyle = '--')   
        if (type(line) == lier.doublet):
            ax.axvline(line.w1, color = 'k', linestyle = '--')   
            ax.axvline(line.w2, color = 'k', linestyle = '--')   
        if (second != None):
            ax.axvline(second.w, color = 'k', linestyle = '--')  
        ax.plot(wave, gau, label='Gaussian fit', linestyle = '--', color = 'orange')
        ax.plot(wave, lor, label='Lorentzian fit', linestyle = '--', color = 'green')
        ax.plot(wave, voi, label='Voigt fit', linestyle = '-', color = 'red')
        ax.plot(wave, reduced_spectrum)
        ax.axhline(y = 0, color = 'k', linestyle = '-')
        ax.set_xlim(line.left[0], line.right[1])
        y_temp = np.max(flux[np.where((wave>line.left[0]) & (wave<line.right[1]))[0]])
        ax.set_ylim([-0.2*y_temp, 1.5*y_temp])
        ax.legend()
        ax.set_title(line.name +'_'+ flux_name)
        fig.savefig('../Output/4. Plotting/Figures 4.5/{}/{}_overplot.png'.format(flux_name, line.name), format='png', dpi = 1200)
        plt.close(fig)
    overplot_bin(high, 'High', line, second = second)
    overplot_bin(mid, 'Mid', line, second = second)
    overplot_bin(low, 'Low', line, second = second)

if (plot_4_5 == True):
    overplot(H_epsilon)
    overplot(OIId)
    overplot(NeIII)
    overplot(OIII)
    overplot(OI)
    overplot(NII, H_alpha)
    overplot(SIId)


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
    elif (type(line) == lier.singlet):
        coeff = get_p(flux, line, fit = 'gaussian')
        return np.sqrt(2*np.pi)*coeff[0]*coeff[2]
    else:
        coeff = get_p(flux, line, fit = 'gaussian')
        return np.sqrt(2*np.pi)*coeff[0]*coeff[3] + np.sqrt(2*np.pi)*coeff[1]*coeff[3]


#%%
#   Plot component of doublets or overlapping singlet (Figures 4.6)

def component(line):
    
    def component_bin(line, flux, flux_name):
        
        #   1. Get the continuum and reduce the spectrum by the fitline
        
        continuum = np.where(((wave > line.left[0]) & (wave < line.left[1])) | ((wave > line.right[0]) & (wave < line.right[1])))[0]
        coeff = np.polyfit(wave[continuum], flux[continuum], 1)
        fitline = wave*coeff[0] + coeff[1]
        reduced_spectrum = flux - fitline
    
        #   2. Plot the spectrum with the fitted voigt profile
        
        fig, ax = plt.subplots(1, 1, figsize = (7, 5))
        if (type(line) == lier.singlet):
            if (line.name == 'NII 6583'):
                NII_other = 6549.86
                coeff = get_p(flux, NII, H_alpha)
                voi1 = coeff[0]*np.real(wofz((wave-coeff[1]+1j*coeff[3])/coeff[2]/np.sqrt(2)))/coeff[2]/np.sqrt(2*np.pi) 
                voi2 = 0.3256*coeff[0]*np.real(wofz((wave-(coeff[1]*NII_other/NII.w)+1j*coeff[3])/coeff[2]/np.sqrt(2)))/coeff[2]/np.sqrt(2*np.pi)
            else:
                print('Input line must be doublet')
        else:
            coeff = get_p(flux, line)
            voi1 = coeff[0] * np.real(wofz((wave - coeff[2] + 1j*coeff[4])/coeff[3]/np.sqrt(2))) / coeff[3] /np.sqrt(2*np.pi)
            voi2 = coeff[1] * np.real(wofz((wave - (coeff[2]*line.w2/line.w1) + 1j*coeff[4])/coeff[3]/np.sqrt(2))) / coeff[3] /np.sqrt(2*np.pi)
        
        ax.plot(wave, voi1, color = 'r')
        ax.plot(wave, voi2, color = 'r')
        ax.plot(wave, reduced_spectrum)
        ax.axhline(y = 0, color = 'k', linestyle = '-')
        ax.set_xlim(line.left[0], line.right[1])
        y_temp = np.max(flux[np.where((wave>line.left[0]) & (wave<line.right[1]))[0]])
        ax.set_ylim([-0.2*y_temp, 1.5*y_temp])
        ax.set_title(line.name +'_'+ flux_name)
        fig.savefig('../Output/4. Plotting/Figures 4.6/{}/{}_component.png'.format(flux_name, line.name), format='png', dpi = 1200)
        plt.close(fig)
    
    component_bin(line, high, 'high')
    component_bin(line, mid, 'mid')
    component_bin(line, low, 'low')

if (plot_4_6 == True):
    component(OIId)
    component(SIId)
    component(NII)

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

if (plot_4_7 == True):
    for l in lines:
            summed_flux(high, l, FWHM(high, l), graph = True, path = '../Output/4. Plotting/Figures 4.7/high')
            summed_flux(mid, l, FWHM(mid, l), graph = True, path = '../Output/4. Plotting/Figures 4.7/mid')
            summed_flux(low, l, FWHM(low, l), graph = True, path = '../Output/4. Plotting/Figures 4.7/low')    

def table_flux(line_array, flux, name):    
    first_row = [name]
    second_row = ['Summed']
    third_row = ['Voigt']
    fourth_row = ['Flux difference (%)']
    
    for l in line_array:
        first_row.append(l.name)
        s = round(summed_flux(flux, l, FWHM(flux, l)),3)
        v = round(voigt_flux(flux, l),3)
        second_row.append(s)
        third_row.append(v)
        fourth_row.append(round(((v - s)/v*100), 3))
    
    return [first_row, second_row, third_row, fourth_row]

print('\n')
print(tabulate(table_flux(strong_lines, high, 'high'), headers='firstrow'), '\n')
print(tabulate(table_flux(strong_lines, mid, 'mid'), headers='firstrow'), '\n')
print(tabulate(table_flux(strong_lines, low, 'low'), headers='firstrow'), '\n')
        
#%%
#   The variance propagate from the original spaxels

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

def table_sigma(line_array, flux, var, name):    
    first_row = [name]
    second_row = ['Summed flux']
    third_row = ['SD']
    for l in line_array:
        first_row.append(l.name)
        s = round(summed_flux(flux, l, FWHM(flux, l)),3)
        second_row.append(s)
        sd = round(np.sqrt(variance(flux, var, l)),3)
        third_row.append(sd)
    return [first_row, second_row, third_row]

print('\n')
print(tabulate(table_sigma(strong_lines, high, high_var, 'high'), headers='firstrow'), '\n')
print(tabulate(table_sigma(strong_lines, mid, mid_var, 'mid'), headers='firstrow'), '\n')
print(tabulate(table_sigma(strong_lines, low, low_var, 'low'), headers='firstrow'), '\n')
print(tabulate(table_sigma(weak_lines, high, high_var, 'high'), headers='firstrow'), '\n')
print(tabulate(table_sigma(weak_lines, mid, mid_var, 'mid'), headers='firstrow'), '\n')
print(tabulate(table_sigma(weak_lines, low, low_var, 'low'), headers='firstrow'), '\n')

#%%
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
        
#   Plot a table showing the SD calculated from the sliding box
def table_box(line_array, flux, name):    
    first_row = [name]
    second_row = ['Summed']
    third_row = ['SD']
    fourth_row = ['# of boxes']
    for l in line_array:
        first_row.append(l.name)
        s = round(summed_flux(flux, l, FWHM(flux, l)),3)
        second_row.append(s)
        temp, sdt = sliding_box(flux, l)
        sd = round(sdt, 3)
        third_row.append(sd)
        fourth_row.append(temp)
    return [first_row, second_row, third_row, fourth_row]

print(tabulate(table_box(strong_lines, high, 'high'), headers='firstrow'), '\n')
print(tabulate(table_box(strong_lines, mid, 'mid'), headers='firstrow'), '\n')
print(tabulate(table_box(strong_lines, low, 'low'), headers='firstrow'), '\n')
print(tabulate(table_box(weak_lines, high, 'high'), headers='firstrow'), '\n')
print(tabulate(table_box(weak_lines, mid, 'mid'), headers='firstrow'), '\n')
print(tabulate(table_box(weak_lines, low, 'low'), headers='firstrow'), '\n')        
        
        
        
        
        