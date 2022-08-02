# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 22:35:54 2022


@author: Leo Lee (modified from code created by Gerome Algodon)

LIER project data analysis pipeline 
2. Spaxels reduction

Input: quiscent_red_galaxies.fits, all maps files and logcube files 
Output: 
    spaxel_data_table.fits
    Bin_1.fits
    Bin_2.fits
    Bin_3.fits
    Bin_1_control.fits
    Bin_2_control.fits
    Bin_3_control.fits
    8 plots(.jpg):
        Figure 2.1: EW of different emission lines against Ha, in contours
        Figure 2.2: EW of different emission lines against Ha after zero point correction
        Figure 2.3: EW of OII against Ha, with the bounding line ruling out low OII/Ha
        Figure 2.4: Gaussian flux of OIII against OII and the bounding lines seperating low OIII/OII galaxies
        Figure 2.5: Line ratio in log of NII/OII against NII/Ha 
        Figure 2.6: Rotated version of Figure 2.5 making the trend of data parallel to the x axis
        Figure 2.7: Figure 2.5 with the three seperated regions of high, mid and low metalicity bins
        Figure 2.8: Velocity dispersion of the three metalicity bins
    
    
Cut notation:
    In this section we use serval cut to eliminate unwanted galaxies, this is a quick check table to find out which portion of galaxies are included (or excluded)
    cut1: contains the spaxels after excluding low OII/Ha EW
    cut2: contains the spaxels in cut1 exclude galaxies with high fractional error of OIII/OII flux
    cut3: contains the spaxels in cut2 and high value of OIII/OII (Seyfert)
    cut4: contains the spaxels in cut1 and exclude cut3
    cut5: contains the spaxels in cut4 and those with valid total EW index (mask != True)
    cut6: contains strong line spaxels with valid and low fractional error of NII/OII, NII/Ha
    cut7: contains the spaxels in cut6 and a valid velocity offset (stellar velocity - gas velocity)
"""

#%%
#   Importing required packages

import numpy as np
from astropy.io import fits
from astropy.table import Table,vstack
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import sys
import time
import astropy.constants
from scipy import spatial  # import this to use KDTree and to query the KDTree

import warnings
warnings.filterwarnings("ignore")   # To ignore warnings raised from division of square root

#%%
#   Set value of speed of light and its inverse

c_vel = astropy.constants.c.to('km/s').value
inv_c = 1/c_vel

#%% 
#   Read the selected galaxies from galaxy selection and check the data of the galaxies are found

manga_path = '/home/rbyan'    # The location of the manga repository

log = fits.open('../Data/quiescent_red_sequence_galaxies.fits')     # Open the fits file containing selected galaxies information

plate = log[1].data['plate']
ifu = log[1].data['ifudsgn']


for i in range(len(plate)):
    maps_file = '{0}/manga/spectro/analysis/v3_1_1/3.1.0/HYB10-MILESHC-MASTARHC2/{1}/{2}/manga-{1}-{2}-MAPS-HYB10-MILESHC-MASTARHC2.fits.gz'.format(manga_path,plate[i],ifu[i])
    logc_file = '{0}/manga/spectro/redux/v3_1_1/{1}/stack/manga-{1}-{2}-LOGCUBE.fits.gz'.format(manga_path,plate[i],ifu[i])
    if not os.path.isfile(maps_file):
        sys.exit('Maps file of galaxy plate-ifu: {}-{} not found'.format(plate[i],ifu[i]))
    if not os.path.isfile(logc_file):
        sys.exit('Logcube file of galaxy plate-ifu: {}-{} not found'.format(plate[i],ifu[i]))

#%% 
#   Initialize columns for table

#   1. General information from the spaxel
names = ['plate','ifu','z_vel','gal_red_B-V','stell_vel_ID','emline_ID',
         'rad_norm_spx','azimuth_spx','snr_spx','snr_bin','stell_vel',
         'stell_vel_mask','ha_vel','ha_vel_mask','stell_sigma_cor',
         'stell_sigma_mask','spec_index_Dn4000','spec_index_ivar_Dn4000',
         'spec_index_mask_Dn4000','spec_index_HDeltaA',
         'spec_index_ivar_HDeltaA','spec_index_mask_HDeltaA','flux_r_band',
         'wave_median_LSF', 'vel_median_LSF','OIId-3727_LSF', 'SIId-4068_LSF', 'OIII-4363_LSF', 
         'H_beta_LSF', 'OIII-5007_LSF', 'NII-5755_LSF', 'OI-6300_LSF', 'SIII-6312_LSF', 'H_alpha_LSF',
         'NII-6583_LSF', 'SIId-6716_LSF', 'OIId-7320_LSF', 'SIIId-9071_LSF', 'SIIId-9533_LSF']

dtype = ['i4','i4','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8',
         'f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8',
         'f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8']

assert len(names)==len(dtype)

#   2. Spectral lines information

emline_names = np.array(['H_alpha','H_beta','OII-3727','OII-3729','NII-6585','SII-6718',
                         'SII-6732','OIII-4960','OIII-5008','OI-6302','OI-6365', 
                         'SIII-9071', 'SIII-9533'])     # The names of the spectral lines

emline_indx = np.array([23,14,0,1,24,26,21,15,16,20,21,31,33])    # Corresponding index numbers found on https://trac.sdss.org/wiki/MANGA/TRM/TRM_MPL-7/DAPDataModel#DAPMAPSfile

assert len(emline_names)==len(emline_indx)

for k in range(len(emline_names)):
    names.append('summed_EW_{}'.format(emline_names[k]))
    names.append('summed_EW_IVAR_{}'.format(emline_names[k]))
    names.append('summed_EW_mask_{}'.format(emline_names[k]))
    
    names.append('gauss_EW_{}'.format(emline_names[k]))
    names.append('gauss_EW_IVAR_{}'.format(emline_names[k]))
    names.append('gauss_EW_mask_{}'.format(emline_names[k]))
    
    names.append('gauss_flux_{}'.format(emline_names[k]))
    names.append('gauss_flux_IVAR_{}'.format(emline_names[k]))
    names.append('gauss_flux_mask_{}'.format(emline_names[k]))
    
    dtype.append('f8')
    dtype.append('f8')
    dtype.append('f8')
    
    dtype.append('f8')
    dtype.append('f8')
    dtype.append('f8')
    
    dtype.append('f8')
    dtype.append('f8')
    dtype.append('f8')
    
table = Table(names=names,dtype=dtype)

#%%
#   Define functions for retriving LSF data and r flux data

#   This function gets the r flux value from the logcube

def r_data(logc, cut):
    r = np.ma.MaskedArray(logc['RIMG'].data,mask=np.invert(logc['RIMG'].data>0))
    return r.data.flatten()[cut] 

#   This function returns the wavelength median LSF value and the velocity median LSF value respectively
#   base on the red shifted value of the wavelength of our interested auroral lines ([SII 4068] to [OII 7330])
def LSF(logc, cut, z):
    lower = (4068.6)*(1+z)    #[SII] 4068 from NIST
    upper = (9533.2)*(1+z)   #[SIII] 9533 from NIST
    low_in = np.max(np.where(wave<lower)[0])
    up_in = np.min(np.where(wave>upper)[0]) if (upper < wave.max()) else int(len(wave-1))
    ind = int(round((low_in+up_in)/2))
    wave_med = (logc['LSFPost'].data[ind,:,:]/wave[ind])*c_vel
    LSF_copy = logc['LSFPost'].data[low_in:up_in,:,:].copy()
    wave_temp = wave[low_in:up_in]
    wave_temp = np.expand_dims(wave_temp, axis = 1)     #old version of numpy on cluster do not support tuple or list input
    wave_temp = np.expand_dims(wave_temp, axis = 2) 
    wave_temp = np.repeat(wave_temp, LSF_copy.shape[1], axis=1)
    wave_temp = np.repeat(wave_temp, LSF_copy.shape[2], axis=2)
    LSF_copy=LSF_copy/wave_temp
    vel_med = c_vel*np.median(LSF_copy,axis=0)
    return wave_med.flatten()[cut], vel_med.flatten()[cut]
    
#   This function reutrns the LSF in velocity space of a specific wavelength
def LSF_lines(logc, cut, z, wavelength):
    wave_z = wavelength*(1+z)
    ind = int(round(np.max(np.where(wave<wave_z)[0])))
    LSF = logc['LSFPost'].data[ind,:,:]
    return (c_vel*(LSF/wave[ind])).flatten()[cut]

#%% 
#   Select good and high S/N spaxels 
#   Fill and save the table using the corresponding columns in the maps file

wave = np.array([])
FirstTime = True

for i in range(len(plate)):
    sys.stdout.write('\r'+('Galaxy {}/{}'.format(i+1,len(plate))))  # print statement to visually check progress that deletes and replaces itself
    
    maps_file = '{0}/manga/spectro/analysis/v3_1_1/3.1.0/HYB10-MILESHC-MASTARHC2/{1}/{2}/manga-{1}-{2}-MAPS-HYB10-MILESHC-MASTARHC2.fits.gz'.format(manga_path,plate[i],ifu[i])
    logc_file = '{0}/manga/spectro/redux/v3_1_1/{1}/stack/manga-{1}-{2}-LOGCUBE.fits.gz'.format(manga_path,plate[i],ifu[i])
    
    maps = fits.open(maps_file)
  
    good_spaxel = (maps['BINID'].data[0,:,:].flatten() != -1)  # ignore useless spaxels (like the corners)
    print('\nIFU shape:', maps['BINID'].data[0].shape, '\nNumber of spaxels: ', good_spaxel.shape[0], '\nNumber of good spaxels: ', good_spaxel.sum())
    snrcut = 15
    cut = (maps['SPX_SNR'].data.flatten()>snrcut)
    print('Number of spaxels after cutting S/N < {}: {}'.format(snrcut,cut.sum()))
    totalcut = ((maps['BINID'].data[0,:,:].flatten() != -1) & (maps['SPX_SNR'].data.flatten()>snrcut))  #%% Apply the S/N cut before the output
    print('Total number of spaxels: {}'.format(totalcut.sum()), '\n')
    row = np.zeros((totalcut.sum(),len(dtype)))  # initialize array where we will be storing row data
    
    # Get Plate and IFU
    row[:,0] = (np.full(totalcut.sum(),plate[i]))  # plate and ifu is the same for every spaxel which is why I use np.full     
    row[:,1] = (np.full(totalcut.sum(),ifu[i]))
    
    # Get z_vel and gal_red_B-V
    row[:,2] = (np.full(totalcut.sum(),maps[0].header['SCINPVEL']))
    z = inv_c*(maps[0].header['SCINPVEL'])
    row[:,3] = (np.full(totalcut.sum(),maps[0].header['EBVGAL']))
    
    # Get Bin IDs we want
    row[:,4] = (maps['BINID'].data[1,:,:].flatten()[totalcut])  # stellar
    row[:,5] = (maps['BINID'].data[3,:,:].flatten()[totalcut])  # emline 1 bin per spaxel
    
    # Get coordinates
    row[:,6] = (maps['SPX_ELLCOO'].data[1,:,:].flatten()[totalcut])  # radius normalized by the elliptical Petrosian effective radius from the NSA
    row[:,7] = (maps['SPX_ELLCOO'].data[2,:,:].flatten()[totalcut])  # azimuth angle
    
    # Get continuum signal to noise ratio
    row[:,8] = (maps['SPX_SNR'].data.flatten()[totalcut])  # S/N in each spaxel (emission-line measurements done per spaxel)
    row[:,9] = (maps['BIN_SNR'].data.flatten()[totalcut])  # S/N in each bin (stellar measurements done per bin)
    
    # Get stellar velocity
    row[:,10] = (maps['STELLAR_VEL'].data.flatten()[totalcut])
    row[:,11] = (maps['STELLAR_VEL_MASK'].data.flatten()[totalcut])
    
    # Get H_alpha velocity (indx 18, channel 19)
    row[:,12] = (maps['EMLINE_GVEL'].data[18,:,:].flatten()[totalcut]) 
    row[:,13] = (maps['EMLINE_GVEL_MASK'].data[18,:,:].flatten()[totalcut])
    
    # Get corrected velocity dispersion, stellar sigma correction channel 1, indx 0
    row[:,14] = ((np.sqrt( (maps['STELLAR_SIGMA'].data)**2 - (maps['STELLAR_SIGMACORR'].data[0])**2 )).flatten()[totalcut])
    row[:,15] = (maps['STELLAR_SIGMA_MASK'].data.flatten()[totalcut])
    
    # Get spectral index for Dn4000 (indx 44, channel 45)
    row[:,16] = (maps['SPECINDEX'].data[44,:,:].flatten()[totalcut])
    row[:,17] = (maps['SPECINDEX_IVAR'].data[44,:,:].flatten()[totalcut])
    row[:,18] = (maps['SPECINDEX_MASK'].data[44,:,:].flatten()[totalcut])
    
    # Get spectral index for HDeltaA (indx 21, channel 22) note: this has a correction but D4000 didn't
    row[:,19] = (maps['SPECINDEX'].data[21,:,:].flatten()*maps['SPECINDEX_CORR'].data[21,:,:].flatten())[totalcut]
    row[:,20] = (maps['SPECINDEX_IVAR'].data[21,:,:].flatten()[totalcut])
    row[:,21] = (maps['SPECINDEX_MASK'].data[21,:,:].flatten()[totalcut])
    
    # Get absolute r band flux, wavelength and velocity median LSF and LSF of the auroral lines from the logcube file
    logc = fits.open(logc_file)
    if (FirstTime==True):
        wave = logc['WAVE'].data
        FirstTime = False
    row[:,22] = r_data(logc, totalcut)
    row[:,23] ,row[:,24] = LSF(logc, totalcut, z)
    #   [SII] 4068,4076, [OIII] 4363, [NII] 5755, [SIII] 6312, [OII] 7320,7330, [SIIId] 9071,9533
    
    row[:,25] = LSF_lines(logc, totalcut, z, (3727.092+3729.875)/2)  # [OIId] 3727, 3729
    row[:,26] = LSF_lines(logc, totalcut, z, (4068.60 + 4076.35)/2)    # [SII] 4068,4076
    row[:,27] = LSF_lines(logc, totalcut, z, 4363.209)  # [OIII] 4363
    row[:,28] = LSF_lines(logc, totalcut, z, 4862.721)  # H beta  
    row[:,29] = LSF_lines(logc, totalcut, z, 5008.24)  # [OIII] 5007
    row[:,30] = LSF_lines(logc, totalcut, z, 5754.59)   # [NII] 5755
    row[:,31] = LSF_lines(logc, totalcut, z, 6302.046)  # [OI] 6300
    row[:,32] = LSF_lines(logc, totalcut, z, 6312.06)   # [SIII] 6312
    row[:,33] = LSF_lines(logc, totalcut, z, 6564.6)  # H alpha
    row[:,34] = LSF_lines(logc, totalcut, z, 6585.27)  # [NII] 6583
    row[:,35] = LSF_lines(logc, totalcut, z, (6718.295 + 6732.674)/2)     # [SII] 6716,6731
    row[:,36] = LSF_lines(logc, totalcut, z, (7330.19 + 7319.92)/2)     # [OII] 7320,7330
    row[:,37] = LSF_lines(logc, totalcut, z, 9071.1)    # [SIIId] 9071
    row[:,38] = LSF_lines(logc, totalcut, z, 9533.2)    # [SIIId] 9533
     

    # Get emmision line properties of Ha, hb, o2, n2, s2, o3, o1, s3
    for j in range(len(emline_names)):
        row[:,39+9*j] = (maps['EMLINE_SEW'].data[emline_indx[j],:,:].flatten()[totalcut])          # summed equivalent width
        row[:,40+9*j] = (maps['EMLINE_SEW_IVAR'].data[emline_indx[j],:,:].flatten()[totalcut])  
        row[:,41+9*j] = (maps['EMLINE_SEW_MASK'].data[emline_indx[j],:,:].flatten()[totalcut])  
        
        row[:,42+9*j] = (maps['EMLINE_GEW'].data[emline_indx[j],:,:].flatten()[totalcut])      #Higher S/N if detect      # gauss equivalent width
        row[:,43+9*j] = (maps['EMLINE_GEW_IVAR'].data[emline_indx[j],:,:].flatten()[totalcut])  
        row[:,44+9*j] = (maps['EMLINE_GEW_MASK'].data[emline_indx[j],:,:].flatten()[totalcut])      
        
        row[:,45+9*j] = (maps['EMLINE_GFLUX'].data[emline_indx[j],:,:].flatten()[totalcut])        # gaussian flux
        row[:,46+9*j] = (maps['EMLINE_GFLUX_IVAR'].data[emline_indx[j],:,:].flatten()[totalcut])   # gaussian flux error
        row[:,47+9*j] = (maps['EMLINE_GFLUX_MASK'].data[emline_indx[j],:,:].flatten()[totalcut])   # same mask for flux and error

    table = vstack([table,Table(row,names=names,dtype=dtype)])
    maps.close()
    del maps
    logc.close()
    del logc

print('We have {} spaxels after cutting S/N < {}'.format(len(table),snrcut))
table.write('../Data/spaxel_data_table.fits', format='fits',overwrite=True)
log.close()
del log

#%%
#   Open the table

spaxel_data_table = fits.open('../Data/spaxel_data_table.fits')
                              
#%%
#   Define functions for getting flux, equivalent width and inverse variance 

def get_data(name,mask_name):
    return np.ma.MaskedArray(spaxel_data_table[1].data[name], mask = spaxel_data_table[1].data[mask_name]>0)

def get_data_dublet(name1, name2, mask_name1, mask_name2):
    return np.ma.MaskedArray((spaxel_data_table[1].data[name1]+spaxel_data_table[1].data[name2]), mask = ((spaxel_data_table[1].data[mask_name1] > 0) | (spaxel_data_table[1].data[mask_name2] > 0)))
    
def get_data_dublet_ivar(name1, name2, mask_name1, mask_name2):
    ivar1 = np.ma.MaskedArray(spaxel_data_table[1].data[name1], mask = spaxel_data_table[1].data[mask_name1])
    ivar2 = np.ma.MaskedArray(spaxel_data_table[1].data[name2], mask = spaxel_data_table[1].data[mask_name2])
    ivar = 1/((1/ivar1)+(1/ivar2))
    return np.ma.MaskedArray(ivar, mask = ((spaxel_data_table[1].data[mask_name1] > 0) | (spaxel_data_table[1].data[mask_name2] > 0)))

#%%
#   Get equivalent width and inverse variance from the table

#   H_alpha 6564
ha_ew = get_data('summed_EW_H_alpha','summed_EW_mask_H_alpha')
ha_ew_ivar = get_data('summed_EW_IVAR_H_alpha','summed_EW_mask_H_alpha')

#   H_beta 4861
hb_ew = get_data('summed_EW_H_beta','summed_EW_mask_H_beta')
hb_ew_ivar = get_data('summed_EW_IVAR_H_beta','summed_EW_mask_H_beta')

#   [OIId] 3727
oII_ew = get_data('summed_EW_OII-3727','summed_EW_mask_OII-3727')
oII_ew_ivar = get_data('summed_EW_IVAR_OII-3727','summed_EW_mask_OII-3727')

#   [OIII] 5008
oIII_ew = get_data('summed_EW_OIII-5008','summed_EW_mask_OIII-5008')
oIII_ew_ivar = get_data('summed_EW_IVAR_OIII-5008','summed_EW_mask_OIII-5008')

#   [OI] 6302
oI_ew = get_data('summed_EW_OI-6302','summed_EW_mask_OI-6302')
oI_ew_ivar = get_data('summed_EW_IVAR_OI-6302 ','summed_EW_mask_OI-6302')

#   [NII] 6585
nII_ew = get_data('summed_EW_NII-6585','summed_EW_mask_NII-6585')
nII_ew_ivar = get_data('summed_EW_IVAR_NII-6585','summed_EW_mask_NII-6585')

#   [SIId] 6718, 6732
sII_ew = get_data_dublet('summed_EW_SII-6718', 'summed_EW_SII-6732', 'summed_EW_mask_SII-6718', 'summed_EW_mask_SII-6732')
sII_ew_ivar = get_data_dublet_ivar('summed_EW_ivar_SII-6718', 'summed_EW_ivar_SII-6732', 'summed_EW_mask_SII-6718', 'summed_EW_mask_SII-6732')

#   [SIII] 9071, 9533
sIII_ew = get_data_dublet('summed_EW_SIII-9071', 'summed_EW_SIII-9533', 'summed_EW_mask_SIII-9071', 'summed_EW_mask_SIII-9533')
sIII_ew_ivar = get_data_dublet_ivar('summed_EW_ivar_SIII-9071', 'summed_EW_ivar_SIII-9533', 'summed_EW_mask_SIII-9071', 'summed_EW_mask_SIII-9533')

#%%
#   Get gaussian flux and inverse variance from the table

#   H_alpha 6564
ha_flux = get_data('gauss_flux_H_alpha','gauss_flux_mask_H_alpha')
ha_flux_ivar = get_data('gauss_flux_IVAR_H_alpha','gauss_flux_mask_H_alpha')

#   H_beta 4861
hb_flux = get_data('gauss_flux_H_beta','gauss_flux_mask_H_beta')
hb_flux_ivar = get_data('gauss_flux_IVAR_H_beta','gauss_flux_mask_H_beta')

#  [OII] 3727
oII_flux = get_data_dublet('gauss_flux_OII-3727', 'gauss_flux_OII-3729', 'gauss_flux_mask_OII-3727', 'gauss_flux_mask_OII-3729')
oII_flux_ivar = get_data_dublet_ivar('gauss_flux_ivar_OII-3727', 'gauss_flux_ivar_OII-3729', 'gauss_flux_mask_OII-3727', 'gauss_flux_mask_OII-3729')

#   [OIII] 5008
oIII_flux = get_data('gauss_flux_OIII-5008','gauss_flux_mask_OIII-5008')
oIII_flux_ivar = get_data('gauss_flux_IVAR_OIII-5008','gauss_flux_mask_OIII-5008')

#   [OI] 6302
oI_flux = get_data('gauss_flux_OI-6302','gauss_flux_mask_OI-6302')
oI_flux_ivar = get_data('gauss_flux_IVAR_OI-6302 ','gauss_flux_mask_OI-6302')

#   [NII] 6585
nII_flux = get_data('gauss_flux_NII-6585','gauss_flux_mask_NII-6585')
nII_flux_ivar = get_data('gauss_flux_IVAR_NII-6585','gauss_flux_mask_NII-6585')

#   [SIId] 6718, 6732
sII_flux = get_data_dublet('gauss_flux_SII-6718', 'gauss_flux_SII-6732', 'gauss_flux_mask_SII-6718', 'gauss_flux_mask_SII-6732')
sII_flux_ivar = get_data_dublet_ivar('gauss_flux_ivar_SII-6718', 'gauss_flux_ivar_SII-6732', 'gauss_flux_mask_SII-6718', 'gauss_flux_mask_SII-6732')

#   [SIII] 9071, 9533
sIII_flux = get_data_dublet('gauss_flux_SIII-9071', 'gauss_flux_SIII-9533', 'gauss_flux_mask_SIII-9071', 'gauss_flux_mask_SIII-9533')
sIII_flux_ivar = get_data_dublet_ivar('gauss_flux_ivar_SIII-9071', 'gauss_flux_ivar_SIII-9533', 'gauss_flux_mask_SIII-9071', 'gauss_flux_mask_SIII-9533')

#%%   
#   2.1 Zero point correction

#   1. Define the function for plotting EW of different emission lines against H alpha, level of contour = 12

def plot_EW(EW, xlim, ylim, title, number=0):
    plt.subplot(2,4,number) if (number!=0) else plt.figure(figsize=(5,5))
    hist_info = plt.hist2d(np.ma.filled(ha_ew),np.ma.filled(EW), bins=300, cmap=plt.cm.gray_r, norm=mpl.colors.LogNorm(), range=[xlim,ylim])  # PLOTS HALPHA AS X AXIS FOR ALL THE GRAPHS
    plt.plot(0,0,'x',c='b',label='(0,0)')   # Creates red x at origin
    plt.legend() 
    plt.xlabel(r'H$\alpha$ EW $[\AA]$')
    plt.ylabel(r'{} EW $[\AA]$'.format(title))
    plt.title(r'{} vs H$\alpha$'.format(title))
    lv = 12     # Total levels of contour
    dlogz=(np.log(hist_info[0].max())-np.log(10))/lv  
    loglvarr=np.arange(lv)*dlogz+np.log(10)
    lvarr=np.exp(loglvarr)
    plt.contour(hist_info[0].transpose(),levels=lvarr,colors='red',extent=[xlim[0],xlim[1],ylim[0],ylim[1]])
    return hist_info
  
#   2. Plot the emission line against Ha (Figure 2.1)

plt.figure(figsize=(18,9))
p1 = plot_EW(oII_ew, [-2,10], [-5,40], r'OII $\lambda3727$', 1)   # 0.4
p2 = plot_EW(nII_ew, [-2,10], [-2,5], r'NII $\lambda6549$', 2)    # 0.04
p3 = plot_EW(sII_ew, [-2,10], [-3,10], r'SII $\lambda\lambda6718,6732$', 3)   # 0.0
p4 = plot_EW(oIII_ew, [-2,10], [-1.5,6], r'OIII $\lambda5008$', 4)    # 0.245
p5 = plot_EW(hb_ew, [-2,10], [-1.5,4], r'H$\beta$', 5)    # -0.035
p6 = plot_EW(oI_ew, [-2,10], [-2,4], r'OI $\lambda6302$', 6)      # -0.052
p7 = plot_EW(sIII_ew, [-2,10], [-4,8], r'SIII $\lambda9071,9533$', 7)     # 0.05\
plt.savefig('../Output/2. Spaxels reduction/Figure 2.1.jpg', format='jpg')
plt.close()

#   3. Apply zero point correction according to the graph above

oII_ew_corr = oII_ew -0.4
nII_ew_corr = nII_ew -0.04
sII_ew_corr = sII_ew
oIII_ew_corr = oIII_ew -0.245
hb_ew_corr = hb_ew +0.035
oI_ew_corr = oI_ew +0.052
sIII_ew_corr = sIII_ew - 0.05

#   4. Plot the corrected graph against Ha (Figure 2.2)

plt.figure(figsize=(18,9))
p8 = plot_EW(oII_ew_corr,[-2,10],[-5,40],r'OII $\lambda3727$', 1) 
p9 = plot_EW(nII_ew_corr,[-2,10],[-2,5],r'NII $\lambda6585$', 2)
p10 = plot_EW(sII_ew_corr,[-2,10],[-3,10],r'SII $\lambda\lambda6718,6732$', 3)
p11 = plot_EW(oIII_ew_corr,[-2,10],[-1.5,6],r'OIII $\lambda5008$', 4) #0.245
p12 = plot_EW(hb_ew_corr,[-2,10],[-1.5,4],r'H$\beta$', 5) #-0.035
p13 = plot_EW(oI_ew_corr,[-2,10],[-2,4],r'OI $\lambda6302$', 6) #-0.052
p14 = plot_EW(sIII_ew_corr,[-2,10],[-4,8],r'SIII $\lambda9071,9533$', 7) #0.05
plt.savefig('../Output/2. Spaxels reduction/Figure 2.2.jpg', format='jpg')
plt.close()

#   5. Zero point correction for all the flux, according to the correction of EW

def flux_corr(EW, flux, corr_value):
    cont = flux/EW
    return flux+(corr_value*cont)

oII_flux_corr = flux_corr(oII_ew, oII_flux, -0.4)
nII_flux_corr = flux_corr(nII_ew, nII_flux, -0.04)
sII_flux_corr = flux_corr(sII_ew, sII_flux, 0)
oIII_flux_corr = flux_corr(oIII_ew, oIII_flux, -0.245)
sIII_flux_corr = flux_corr(sIII_ew, sIII_flux, -0.05)
hb_flux_corr = flux_corr(hb_ew, hb_flux, 0.035)
oI_flux_corr = flux_corr(oI_ew, oI_flux, 0.052)
sIII_flux_corr = flux_corr(sIII_ew, sIII_flux, -0.05) 

#%%
#   2.2 Low OII/Ha cut

#   1. Plot EW of OII Vs Ha and draw a line eliminating galaxies from other ionization mechanism (Figure 2.3)
x = np.linspace(-2,10,len(spaxel_data_table[1].data['ifu']))
y = 5*x-5
y_cut = 5*ha_ew-5
cut1 = ((oII_ew_corr>y_cut) & (ha_ew.mask==False) & (oII_ew_corr.mask==False)).data


plot_EW(oII_ew_corr,[-2,10],[-5,40],r'OII $\lambda3727$')
plt.plot(x, y, ls='--', c='b', label='Bounding line')
plt.legend()
plt.savefig('../Output/2. Spaxels reduction/Figure 2.3.jpg', format='jpg')
plt.close()

print('We have {} spaxels after eliminating low OII/Ha \n'.format(cut1.sum()))

#%% 
#   2.3 Seyfert galaxies cut

#   1. Select spaxels with low fractional error 
up_lim_frac_err_oIII_oII = 0.3

frac_err_oIII_oII = np.ma.sqrt((1/(oIII_flux_corr**2*oIII_flux_ivar)) + (1/(oII_flux_corr**2*oII_flux_ivar)))
cut2 = (cut1 & (frac_err_oIII_oII<up_lim_frac_err_oIII_oII) & (frac_err_oIII_oII.mask==False)).data

print('We have {} spaxels with low fractional error of OIII/OII'.format(cut2.sum()))

#   2. Plot gaussian flux of OIII Vs OII and draw a line eliminating galaxies with high OIII/OII values (>1) (Figure 2.4)
x = np.linspace(-2.5,15,cut2.sum()) 
y = x
y_cut = oII_flux_corr

plt.figure(figsize=(5,5))
plt.hist2d(np.ma.filled(oII_flux_corr[cut2]),np.ma.filled(oIII_flux_corr[cut2]), bins=300, cmap=plt.cm.gray_r, norm=mpl.colors.LogNorm(), range=[[-2.5,15],[-2.5,15]])
plt.plot(0,0,'x',c='b',label='(0,0)')  # Creates blue x at origin
plt.plot(x , y, ls='--', c='r')
plt.legend() 
plt.xlabel(r'OII flux')
plt.ylabel(r'OIII flux')
plt.title(r'OIII vs OII of galaxies with low fractional error')
plt.savefig('../Output/2. Spaxels reduction/Figure 2.4.jpg', format='jpg')
plt.close()

cut3 = (cut2 & (oIII_flux_corr<=y_cut)).data
print('We found {} spaxels containing Seyfert galaxy'.format(cut2.sum() - cut3.sum()))
cut4 = ((cut1==True) & ((cut2 != cut3) == False))
print('We have {} spaxels after eliminating low OII/Ha and Seyfert galaxies \n'.format(cut4.sum()))

#%%
#   2.4 Strong lines spaxels selection

#   1. Calculate the total EW index and seperate the spaxels into strong and zero lines spaxels
tot_ew_indx = ha_ew + 1.03*nII_ew_corr + 5*oII_ew_corr + 0.5*(oIII_ew_corr+sII_ew_corr)
strong_line_percentile = 80   # Spaxels higher than this value will be classified as strong lines sample

cut5 = cut4 & (tot_ew_indx.mask == False)
strong_line = (cut5 & (tot_ew_indx>np.percentile(tot_ew_indx[cut5],strong_line_percentile))).data
print('We have {} strong line spaxels\n'.format(strong_line.sum()))

#%%
#   2.5 Metalicity bins seperation

#   1. Calculate the flux ratios and fractional error of nII/ha and nII/oII in log10
lr_nII_ha = np.ma.log10(nII_flux_corr)-np.ma.log10(ha_flux)    # lr = line ratio
lr_nII_oII = np.ma.log10(nII_flux_corr)-np.ma.log10(oII_flux_corr)  
frac_err_nII_ha = np.ma.sqrt(1/((nII_flux_corr)**2*(nII_flux_ivar)) + 1/((ha_flux)**2*(ha_flux_ivar))) / np.log(10)  
frac_err_nII_oII = np.ma.sqrt(1/((nII_flux_corr)**2*(nII_flux_ivar)) + 1/((oII_flux_corr)**2*(oII_flux_ivar))) / np.log(10)

#   2. Select the spaxels with strong emission and low fractional error of nII/ha and nII/oII in log10
up_lim_frac_err_nII_ha = 0.3    # Upper limit fmrom fractional error cut
up_lim_frac_err_nII_oII = 0.3

cut6 = ((lr_nII_ha.mask == False) & (lr_nII_oII.mask == False) & (strong_line) & (frac_err_nII_ha<up_lim_frac_err_nII_ha) & (frac_err_nII_oII<up_lim_frac_err_nII_oII))
print('We have {} spaxels with strong emission and good frac errors'.format(cut6.sum()))

#   3. Select the spaxels with valid velocity offset
vel_offset = get_data('stell_vel','stell_vel_mask') - get_data('ha_vel','ha_vel_mask')
cut7 = ((cut6)&(vel_offset.mask==False))
print('We have {} spaxels with strong emission, good frac errors and usable stellar and gas velocity'.format(cut7.sum()))

#   4. Plot nII/oII against nII/Ha (Figure 2.5)
xlim_lr = np.array([-1.5,1.5])
ylim_lr = np.array([-1.5,1.5])
x_lr = np.linspace(xlim_lr[0],xlim_lr[1],100)
y_lr = 1.3*x_lr - 0.3

plt.figure(figsize=(5,5))
p = plt.hist2d(lr_nII_ha[cut7], lr_nII_oII[cut7], bins=100, cmap=plt.cm.gray_r, norm=mpl.colors.LogNorm(), range=[xlim_lr,ylim_lr])
lv = 12     # Total levels of contour
dlogz=(np.log(p[0].max())-np.log(10))/lv 
loglvarr=np.arange(lv)*dlogz+np.log(10)
lvarr=np.exp(loglvarr)
plt.contour(p[0].transpose(), levels=lvarr, colors='red', extent=[xlim_lr[0],xlim_lr[1],ylim_lr[0],ylim_lr[1]])
plt.plot(x_lr,y_lr,ls='dashed',c='b')
plt.xlabel('log [NII]/H alpha')
plt.ylabel('log [NII]/[OII]')
plt.title('log [NII]/[OII] Vs log [NII]/H alpha')
plt.savefig('../Output/2. Spaxels reduction/Figure 2.5.jpg', format='jpg')
plt.close()

#   5. Rotate the plot for dividing the sample into three equal size along the line showing the slope of the data (Figure 2.6)
def rotate(theta,x,y):
    c,s = np.cos(theta),np.sin(theta)
    x_rot = c*x - s*y
    y_rot = s*x + c*y
    return x_rot,y_rot

theta = -np.arctan(1.3)  # slope of y_lr  since m = arctan(theta), we rotate by -arctan(theta)
x_rot,y_rot = rotate(theta,lr_nII_ha,lr_nII_oII)
x_lr_rot,y_lr_rot = rotate(theta,x_lr,y_lr)
xlim_lr_rot = np.array([-1,1])
ylim_lr_rot = np.array([-1,1])

plt.figure(figsize=(5,5))
plt.hist2d(x_rot[cut7],y_rot[cut7], bins=100, cmap=plt.cm.gray_r,
           norm=mpl.colors.LogNorm(), range=[xlim_lr_rot,ylim_lr_rot])
plt.plot(x_lr_rot, y_lr_rot, ls='dashed', c='r')

x_lr_33 = np.full(100,np.percentile(x_rot[cut7],33.33))
x_lr_66 = np.full(100,np.percentile(x_rot[cut7],66.66))
y_lr_33 = np.linspace(ylim_lr_rot[0],ylim_lr_rot[1],100)

plt.plot(x_lr_33,y_lr_33,ls='dashed',c='b')
plt.plot(x_lr_66,y_lr_33,ls='dashed',c='b')
plt.xlabel(r'$x_{rot}$')
plt.ylabel(r'$y_{rot}$')
plt.xlim(xlim_lr_rot)
plt.ylim(ylim_lr_rot)
plt.savefig('../Output/2. Spaxels reduction/Figure 2.6.jpg', format='jpg')
plt.close()

#   6. Remake graph with the lines diving data into different metalicity bins (Figure 2.7)
x_bin_cut1,y_bin_cut1 = rotate(-theta,x_lr_33,y_lr_33)
x_bin_cut2,y_bin_cut2 = rotate(-theta,x_lr_66,y_lr_33)
plt.figure(figsize=(5,5))
plt.hist2d(lr_nII_ha[cut7],lr_nII_oII[cut7], bins=100, cmap=plt.cm.gray_r,  # a copy-paste from above
           norm=mpl.colors.LogNorm(),range=[xlim_lr,ylim_lr])
plt.plot(x_lr,y_lr,ls='dashed',c='r')
plt.plot(x_bin_cut1,y_bin_cut1,ls='dashed',c='b')  # New part
plt.plot(x_bin_cut2,y_bin_cut2,ls='dashed',c='b')
plt.xlabel(r'log [NII]/H$\alpha$')
plt.ylabel('log [NII]/[OII]')
plt.xlim(xlim_lr)
plt.ylim(ylim_lr)
plt.text(-0.45,0.75,r'High [NII]/H$\alpha$')
plt.text(-1.40,0.45,r'Mid [NII]/H$\alpha$')
plt.text(-1.4,-0.25,r'Low [NII]/H$\alpha$')
plt.title('log [NII]/[OII] Vs log [NII]/H alpha')
plt.savefig('../Output/2. Spaxels reduction/Figure 2.7.jpg', format='jpg')
plt.close()

#   7. Divide the data into three metalicity bins
bin1_high_nII_ha = ((cut7) & (x_rot>np.percentile(x_rot[cut7],(200/3)))).data
bin2_mid_nII_ha = ((cut7) & (x_rot>np.percentile(x_rot[cut7],(100/3))) & (x_rot<np.percentile(x_rot[cut7],(200/3)))).data
bin3_low_nII_ha = ((cut7) & (x_rot<np.percentile(x_rot[cut7],(100/3)))).data

print('We have {} spaxels in high metalicity bin'.format(bin1_high_nII_ha.sum()))
print('We have {} spaxels in mid metalicity bin'.format(bin2_mid_nII_ha.sum()))
print('We have {} spaxels in low metalicity bin\n'.format(bin3_low_nII_ha.sum()))

#%%
#   2.6 Zero line spaxels selection

#   1. Use the negative equilvalent width part to estimate the variance of the emission line
def ivar_neg_ew(ew):
    ncut1 = (cut4 & ((np.ma.filled(ew)<0)))
    sd1 = np.sqrt(np.sum(ew[ncut1]**2)/ncut1.sum())
    ncut2 = (ncut1 & (ew>(-3*sd1)))
    return ncut2.sum()/np.sum(ew[ncut2]**2)

ivar_ha = ivar_neg_ew(ha_ew)   
ivar_oII = ivar_neg_ew(oII_ew_corr) 
ivar_nII = ivar_neg_ew(nII_ew_corr)
ivar_sII = ivar_neg_ew(sII_ew_corr)
ivar_oIII = ivar_neg_ew(oIII_ew_corr)
ivar_hb = ivar_neg_ew(hb_ew_corr)

#   2. Select the zero lines spaxels and make sure the they have valid velocity dispersion and Dn4000
mult_eli = ((ha_ew)**2*ivar_ha + (oII_ew_corr)**2*ivar_oII + (nII_ew_corr)**2*ivar_nII + (sII_ew_corr)**2*ivar_sII + (oIII_ew_corr)**2*ivar_oIII < 6.25 ) #Signal to noise squared of all the strong lines
zero_cut = ((mult_eli)&(mult_eli.mask==False)&(cut4)&(np.invert(strong_line))).data
Dn4000 = get_data('spec_index_Dn4000','spec_index_mask_Dn4000')
vdisp = get_data('stell_sigma_cor','stell_sigma_mask')                 # get dispersion velocityz
zero_cut = ((zero_cut) & (Dn4000.mask==False) & (vdisp.mask==False) & np.isfinite(vdisp)).data
print('Zero line sample has {} spaxels\n'.format(zero_cut.sum()))

#%%
#   2.7 Matching strong line and zero line samples

#   1. Calculate the absolute r band (corrected for galactic extinction and relativistic effect)
flux_r_band = np.log10((spaxel_data_table[1].data['flux_r_band']*10**(0.4*spaxel_data_table[1].data['gal_red_B-V']*2.285))*(1+spaxel_data_table[1].data['z_vel']/c_vel)**4)

#   2. Normalized the parameters for samples matching
def norm_0_1(array,indx):
    # This shifts the array so that the 5th and 95th percentile of array[indx] are 0 and 1 respectively
    # note: this will shift the entire array but the shift is only based on the indexed array
    return ((array-np.percentile(array[indx],5)) / (np.percentile(array[indx],95)-np.percentile(array[indx],5)))

LSF = spaxel_data_table[1].data['vel_median_LSF']
norm_vdisp = norm_0_1(vdisp,zero_cut).data  
norm_Dn4000 = norm_0_1(Dn4000,zero_cut).data  
norm_flux_r_band = norm_0_1(flux_r_band,zero_cut)
vdisp_LSF = np.sqrt(vdisp**2 + LSF**2)
norm_LSF = norm_0_1(vdisp_LSF, zero_cut)

control = spatial.KDTree(list(zip(norm_vdisp[zero_cut],norm_Dn4000[zero_cut], norm_flux_r_band[zero_cut], norm_LSF[zero_cut])))   # create a KDTree based on the normalized array 

#   3. Define functions needed for samples matching
def redo_gt_1(used_indx, d):
    # Function that checks which spaxels to redo in the search for a control sample
    # Here the condition is to redo a spaxel if it appears more than TWICE in the
    # control group. 
    # array = array of the number indices of the spaxels to be used in the control sample
    # EX: array = ([[3,7],[8,2],[2,5],[2,3]]) the spaxel with index '2' is used 3 times
    # so the output will be: ([False, False, False, False, False, False, True, False])
    # notice this code flattens the array. Since things can be repeated twice, only the 
    # third instance of 2 is marked to be re-done
    unique, return_inverse, counts = np.unique(used_indx,return_inverse=True,return_counts=True)
    redo = np.full(len(return_inverse),False)    # initialize array with False: Meaning don't redo anything
    for i in range(len(np.where(counts>1)[0])):
        count_gt_1 = np.where(return_inverse==np.where(counts>1)[0][i])[0]
        count_gt_1 = np.delete(count_gt_1, np.argmin(d[count_gt_1]))
        redo[count_gt_1] = True
    return redo

def check_unused_indx(array,used_indx):
    # array = only the length of this matters, it must be the len(array)==# of spaxels in total control sample
    # used_indx = array of the number indices of the spaxels to be used in this specific control sample
    # EX: if there are 10 spaxels in the total control sample (that is len(array)=10)and the specific control sample only uses 
    #     the 3rd and 5th spaxel (that is used_indx=[2,4]), this will return 
    #     array([ True,  True, False,  True,  True, False,  True,  True,  True, True])
    unused=np.full(len(array),False)  # True if indx does not appear in used_indx
    for i in range(len(array)):
        if (i==used_indx).sum()==0:
            unused[i] = True
    return unused

def get_control(bin_cut):    
    # Build the KDTree and get the control samples of each strong line spaxels
    dist, bin_control = control.query(list(zip(norm_vdisp[bin_cut], norm_Dn4000[bin_cut],   # 
                                                norm_flux_r_band[bin_cut],
                                                norm_LSF[bin_cut])),k=2)
    dist, bin_control = dist.ravel(),bin_control.ravel()
    redo = redo_gt_1(bin_control , dist)  # see if we have to redo anything
#   if redo.sum()!=0:
    print('We have to redo {}/{} queries'.format(redo.sum(),len(redo)))
    max_itt = 25
    itt = 0
    while (redo.sum() != 0):   # loops until redo.sum() is 0
        # check what indices were not used
        unused = check_unused_indx(norm_vdisp[zero_cut],bin_control) 
        # make new control sample out of unused indices
        control_new = spatial.KDTree(list(zip(norm_vdisp[zero_cut][unused],
                                                    norm_Dn4000[zero_cut][unused],
                                                    norm_flux_r_band[zero_cut][unused],
                                                    norm_LSF[zero_cut][unused])))
        # Search again for each spaxel with the new control
        dist_new, bin_control_new = control_new.query(list(zip(norm_vdisp[bin_cut],  
                                                norm_Dn4000[bin_cut],
                                                norm_flux_r_band[bin_cut],
                                                norm_LSF[bin_cut])),k=2)
        # Turn the new index into old index so we can replace the
        bin_control_new = np.where(unused==True)[0][bin_control_new]  
        # Replace only the ones we need to redo
        bin_control[redo]=bin_control_new.ravel()[redo]
        dist[redo] = dist_new.ravel()[redo]
        redo = redo_gt_1(bin_control, dist)  # see if we have to redo anything still
        print('We have to redo {} queries'.format(redo.sum()))
        itt += 1 
        if itt == max_itt:   # stops infinite loop as long as max_itt is an int > 1
            print('Reached max itteration of {}'.format(max_itt))
            break
    bin_control = np.where(zero_cut==True)[0][bin_control]   # changes indices to global indx only works if control is based on zero_cut
    return bin_control,dist

#   4. Create the control sample
bin1_control,bin1_dist = get_control(bin1_high_nII_ha)   
print()
bin2_control,bin2_dist = get_control(bin2_mid_nII_ha)
print()
bin3_control,bin3_dist = get_control(bin3_low_nII_ha)
print()

#   5. Limit the max_distance between neighboring points and remake bin_split and bin_control
max_dist = 0.05 # 3 dimensions 95th-5th = 1 when norm , 1/10=0.1, 0.1/2=0.05

bin1_indx = np.where(bin1_high_nII_ha==True)[0]
bin2_indx = np.where(bin2_mid_nII_ha==True)[0]
bin3_indx = np.where(bin3_low_nII_ha==True)[0]
def dist_selection(dist, control_indx, bin_indx):                                # this also makes the two new arrays a global indx (see return)
    control_new = control_indx.copy()
    indx_new = bin_indx.copy()
    dist_new = dist.copy()
    gt_max = np.where(dist>=max_dist)[0]
    if len(gt_max)!=0:
        delete_control = np.array([],dtype='i4')
        delete_bin = np.array([],dtype='i4')
        for j in range(len(gt_max)):
            if gt_max[j]%2==0:
                delete_control = np.append(delete_control,[gt_max[j],gt_max[j]+1])
                delete_bin  = np.append(delete_bin, [int(gt_max[j]/2)])
            else:
                delete_control = np.append(delete_control,[gt_max[j],gt_max[j]-1])
                delete_bin = np.append(delete_bin, [int((gt_max[j]-1)/2)])
        indx_new = np.delete(indx_new, delete_bin)
        control_new = np.delete(control_new, delete_control)
        dist_new = np.delete(dist_new, delete_control)
    return  dist_new, indx_new, control_new

bin1_dist_new, bin1_indx_new, bin1_control_new = dist_selection(bin1_dist,bin1_control,bin1_indx)
bin2_dist_new, bin2_indx_new, bin2_control_new = dist_selection(bin2_dist,bin2_control,bin2_indx)
bin3_dist_new, bin3_indx_new, bin3_control_new = dist_selection(bin3_dist,bin3_control,bin3_indx)

#%%
#   2.8 Velocity offset bins separation

#   1. Plot the histogram of velocity offset of the strong line samples (Figure 2.8)
number_of_velocity_bins = 25

plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
bin1_hist = plt.hist(np.ma.filled(vel_offset[bin1_indx_new]), range=(-500,500), bins=number_of_velocity_bins)
plt.title(r'High [NII]/H$\alpha$')
plt.xlabel('Velocity offset (Stellar - Gas)')
plt.subplot(1,3,2)
bin2_hist = plt.hist(np.ma.filled(vel_offset[bin2_indx_new]), range=(-500,500), bins=number_of_velocity_bins)
plt.title(r'Mid [NII]/H$\alpha$')
plt.xlabel('Velocity offset (Stellar - Gas)')
plt.subplot(1,3,3)
bin3_hist = plt.hist(np.ma.filled(vel_offset[bin3_indx_new]), range=(-500,500), bins=number_of_velocity_bins)
plt.title(r'Low [NII]/H$\alpha$')
plt.xlabel('Velocity offset (Stellar - Gas)')
plt.savefig('../Output/2. Spaxels reduction/Figure 2.8.jpg', format='jpg')
plt.close()

#   2. Seperate the samples to velocity bins using the above histogram

def split_bins(bin_num,bin_hist,bin_control):
    # Creates bin_splits based on the histograms above. Change the histogram, change the bin_split
    split = []
    avg = np.zeros(len(bin_hist[0]))
    control = []
    for i in range(len(bin_hist[1])-1):
        temp = np.where((vel_offset[bin_num]>=bin_hist[1][i]) & (vel_offset[bin_num]<bin_hist[1][i+1]))
        split.append(bin_num[temp])
        avg[i] = np.ma.average(vel_offset[bin_num[temp]])
        temp = np.asarray(temp)
        temp = 2*temp
        temp = np.append(temp, temp+1)
        temp = np.sort(temp)
        control.append(bin_control[temp])
    return split, avg, control
    
bin1_split, bin1_avg, bin1_control_split = split_bins(bin1_indx_new, bin1_hist, bin1_control_new)
bin2_split, bin2_avg, bin2_control_split = split_bins(bin2_indx_new, bin2_hist, bin2_control_new)
bin3_split, bin3_avg, bin3_control_split = split_bins(bin3_indx_new, bin3_hist, bin3_control_new)

#%%
#   Save the bins data to a fits file

def save_bin(bin_split,bin_avg,bin_control,num):
    bin_s = fits.HDUList()
    for i in range(len(bin_split)):
        bin_s.append(fits.ImageHDU(bin_split[i],name='{}_{}'.format(num,i+1)))
    bin_s.append(fits.ImageHDU(bin_avg,name='AVG_OFFSET_SUBBIN'))
    bin_s.writeto('../Data/Bin/Bin_{}.fits'.format(num),overwrite=True)
    bin_c = fits.HDUList()
    for i in range(len(bin_control)):
        bin_c.append(fits.ImageHDU(bin_control[i],name='{}_{}'.format(num,i+1)))
    bin_c.append(fits.ImageHDU(bin_avg,name='AVG_OFFSET_SUBBIN'))
    bin_c.writeto('../Data/Bin/Bin_{}_Control.fits'.format(num),overwrite=True)
    
save_bin(bin1_split,bin1_avg,bin1_control_split,1)
save_bin(bin2_split,bin2_avg,bin2_control_split,2)
save_bin(bin3_split,bin3_avg,bin3_control_split,3)