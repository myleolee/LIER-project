# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 18:25:38 2022

@author: Leo Lee (modified from code created by Gerome Algodon)

LIER project data analysis pipeline 
4. Graph Plotting - 4.1. Spectrum plotting

Input: 3 bin_stack.fits files, 3 control_bin_stack.fits files, 3 bin.fits files

Output: 
    stacked_residual.fits
    3 Figures(.png):
        Figures 4.1: Strong line spectum plotted with control line spectrum (left), raw residual spectrum (right)
        Figures 4.2: Residual spectra after Bspline correction
        Figures 4.3: Stacked residual spectra
"""
#%%
#   To control whether the figures will be generated
plot_4_1 = True
plot_4_2 = True
plot_4_3 = True

#%% 
#   Importing required packages

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splrep, BSpline
from matplotlib import rc
import os
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

dir_array = ['../Output/4. Plotting/Figures 4.1',  '../Output/4. Plotting/Figures 4.1/high',  '../Output/4. Plotting/Figures 4.1/mid',
             '../Output/4. Plotting/Figures 4.1/low', '../Output/4. Plotting/Figures 4.2', '../Output/4. Plotting/Figures 4.2/high',
             '../Output/4. Plotting/Figures 4.2/mid', '../Output/4. Plotting/Figures 4.2/low', '../Output/4. Plotting/Figures 4.3']

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

fig, ax = plt.subplots(1, 1, figsize=(19,9))
overall = stack_3[1].data[10] - stack_3_control[1].data[10]
for i in range(11,16):
    ax.plot(wave, stack_3[1].data[i], label = i, color = (1-i*0.04, 0, i*0.04))
    ax.plot(wave, stack_3_control[1].data[i], label = i, color = (1-i*0.04, 0, i*0.04))
    spec = stack_3[1].data[i] - stack_3_control[1].data[i]
    ax.plot(wave, spec, label = i, color = (1-i*0.04, 0, i*0.04))
    overall += spec
ax.plot(wave, overall/6, label = i, color = 'yellow')
fig.legend()
ax.set_xlim(5850, 5900)
ax.set_ylim(-0.02, 0.03)
plt.axhline(y = 0, color = 'k', linestyle = '-')
fig.tight_layout()
fig.show()    

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
        if (plot_4_1 == True):
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
        
        if (plot_4_2 == True):
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
if (plot_4_3 == True):
    plt.savefig('../Output/4. Plotting/Figures 4.3/high.png', format='png')
    plt.close()

plt.figure(figsize=(20,7))
plt.plot(wave, bin2)
plt.axhline(y = 0, color = 'r', linestyle = '-')
plt.xlim((wave[0],wave[-1]))
plt.ylim([-0.05,0.4])
plt.title('Stacked spectra, Mid metalicity bin')
if (plot_4_3 == True):
    plt.savefig('../Output/4. Plotting/Figures 4.3/mid.png', format='png')
    plt.close()

plt.figure(figsize=(20,7))
plt.plot(wave, bin3)
plt.axhline(y = 0, color = 'r', linestyle = '-')
plt.xlim((wave[0],wave[-1]))
plt.ylim([-0.05,0.4])
plt.title('Stacked spectra, Low metalicity bin')
if (plot_4_3 == True):    
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
