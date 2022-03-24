# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 12:22:03 2022

@author: Leo Lee (modified from code created by Gerome Algodon)

Overall data analysing pipeline of the LIER project
"""

#%%
#   Importing required packages

import time
import numpy as np 
from astropy.io import fits
from astropy.table import Table, vstack
import astropy.constants
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import sys
import warnings   

warnings.filterwarnings("ignore")   # To ignore warnings raised from division of square root

#%%
"""
1. Galaxy selection

Input: drpall and dapall from MANGA (2 fits file)
Output: 3 plots(.jpg) and 1 fits file (quiscent_red_galaxies.fits)
"""

#%%
#   Read drpall and dapall, which should be stored in Data repository

drp_all = fits.open('../Data/drpall-v3_1_1.fits')
dap_all = fits.open('../Data/dapall-v3_1_1-3.1.0.fits')

#%%
#   From dapall, extract Dn4000, from drpall, extract r band flux
#   Exclude faulted spaxels in the dap

dapindx = dap_all["HYB10-MILESHC-MASTARHC2"].data['DAPDONE']
Dn4000 = dap_all["HYB10-MILESHC-MASTARHC2"].data['SPECINDEX_1RE'][:,44][dapindx]  #Index 44, channel 45 according to https://sdss-mangadap.readthedocs.io/en/latest/datamodel.html#dap-maps-file

drpindx = dap_all["HYB10-MILESHC-MASTARHC2"].data['DRPALLINDX'][dapindx] #select the corresponding DRP index
#    Absolute magnitude in rest-frame SDSS r-band, from elliptical Petrosian fluxes (NSA)
r = drp_all["MANGA"].data['nsa_elpetro_absmag'][drpindx,4]  #According to https://github.com/sciserver/sqlloader/blob/master/schema/sql/MangaTables.sql

#%%
#   Select red quiscent galaxy using the graph of Dn4000 Vs r band flux

xlim = np.array([-24,-15])   #  xy axis of the three plots
ylim = np.array([1,2.25])

#   1. Plot the galaxy with Dn4000 to r (Figure 1.1)
plt.figure(num='Figure 1.1',figsize=[6,5])
plt.xlim(xlim)
plt.ylim(ylim)
plt.plot(r,Dn4000,'.', color='black', alpha=0.1)
plt.xlabel('$M_r$')
plt.ylabel('$D_n$4000')
plt.title('$D_n$4000 Vs $M_r$')
plt.savefig('../Output/1. Galaxy Selection/Figure 1.1.jpg', format='jpg')
plt.close()

#   2. Draw lines to seperate red galaxies region on the plot (Figure 1.2)

x = np.linspace(xlim[0],xlim[1],len(r))    # For input of the straight line
y_mid = -0.025*x + 1.45     # Equations of the middle line
y_up = -0.025*x + 1.55      # Equations of the upper bounding line
y_low = -0.025*x + 1.35     # Equations of the lower bounding line
lv = 12     # Total levels of contour

plt.figure(num='Figure 1.2',figsize=[6,5])
counts, xbins, ybins, images= plt.hist2d(r,Dn4000, bins=30, cmap=plt.cm.gray_r,norm=mpl.colors.LogNorm(), range=[xlim,ylim], alpha=0)
dlogz=(np.log(counts.max())-np.log(10))/lv   #Step size in log space
loglvarr=np.arange(lv)*dlogz+np.log(10)      #The level marks in log space
lvarr=np.exp(loglvarr)      #The level marks in linear space
plt.contour(counts.transpose(),colors='black', levels=lvarr, extent=[xlim[0],xlim[1],ylim[0],ylim[1]])
plt.plot(x, y_mid, ls='--', c='b', label='Centre line')
plt.plot(x, y_up, ls='--', c='r', label='Bounding lines')
plt.plot(x, y_low, ls='--', c='r')
plt.xlabel('$M_r$')
plt.ylabel('$D_n$4000')
plt.title('$D_n$4000 Vs $M_r$ in contour (12 log levels)')
plt.legend()
plt.savefig('../Output/1. Galaxy Selection/Figure 1.2.jpg', format='jpg')
plt.close()

#   3. Select the galaxies inside the two bounding lines (Figure 1.3)
y_up_cut = -0.025*r + 1.55      # Implementation of the upper cut
y_low_cut = -0.025*r + 1.35     # Implementation of the lower cut

red_seq = (Dn4000>y_low_cut)&(Dn4000<y_up_cut) # Apply cut on data
plt.figure(figsize=[6,5])
plt.xlim(xlim)
plt.ylim(ylim)
plt.plot(r[red_seq],Dn4000[red_seq], '.', c='k', alpha=0.1, label='Red quiscent galaxies') # Skimmed out the selected red galaxies
plt.plot(x, y_low, ls='--', c='r')
plt.plot(x, y_up, ls='--', c='r')
plt.xlabel('$M_r$')
plt.ylabel('$D_n$4000')
plt.legend()
plt.savefig('../Output/1. Galaxy Selection/Figure 1.3.jpg', format='jpg')
plt.close()

#%% 
#   Save the selected data to a new fits table
ifudsgn = drp_all['MANGA'].data['ifudsgn'][drpindx][red_seq]
plate = drp_all['MANGA'].data['plate'][drpindx][red_seq]
table = Table([plate,ifudsgn],names=['plate','ifudsgn'])
table.write('../Data/quiescent_red_sequence_galaxies.fits', format='fits', overwrite=True)


# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 22:35:54 2022


@author: Leo Lee (modified from code created by Gerome Algodon)

LIER project data analysis pipeline 
2. Spaxels reduction

Input: quiscent_red_galaxies.fits, all maps files and logcube files 
Output: spaxel_data_table.fits, 
"""

#%%
#   Set value of speed of light and its inverse

c = astropy.constants.c.to('km/s').value
inv_c = 1/c

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
         'wave_median_LSF', 'vel_median_LSF', 'SIId-4068_LSF', 'OIII-4363_LSF',
         'NII-5755_LSF', 'SIII-6312_LSF', 'OIId-7320_LSF', 'SIIId-9071_LSF', 'SIIId-9533_LSF']

dtype = ['i4','i4','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8',
         'f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8',
         'f8','f8','f8','f8','f8','f8']

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
    wave_med = (logc['LSFPost'].data[ind,:,:]/wave[ind])*c
    LSF_copy = logc['LSFPost'].data[low_in:up_in,:,:].copy()
    wave_temp = wave[low_in:up_in]
    wave_temp = np.expand_dims(wave_temp, axis = 1)     #old version of numpy on cluster do not support tuple or list input
    wave_temp = np.expand_dims(wave_temp, axis = 2) 
    wave_temp = np.repeat(wave_temp, LSF_copy.shape[1], axis=1)
    wave_temp = np.repeat(wave_temp, LSF_copy.shape[2], axis=2)
    LSF_copy=LSF_copy/wave_temp
    vel_med = c*np.median(LSF_copy,axis=0)
    return wave_med.flatten()[cut], vel_med.flatten()[cut]
    
#   This function reutrns the LSF in velocity space of a specific wavelength
def LSF_lines(logc, cut, z, wavelength):
    wave_z = wavelength*(1+z)
    ind = int(round(np.max(np.where(wave<wave_z)[0])))
    LSF = logc['LSFPost'].data[ind,:,:]
    return (c*(LSF/wave[ind])).flatten()[cut]

#%% 
#   Select good and high S/N spaxels and fill the table using the corresponding columns in the maps file

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
    #   [SII] 4068,4076, [OIII] 4363, [NII] 5755, [SIII] 6312, [OII] 7320,7330, [SIIId] 9071, 9533
    row[:,25] = LSF_lines(logc, totalcut, z, (4068.60 + 4076.35)/2)    # [SII] 4068,4076
    row[:,26] = LSF_lines(logc, totalcut, z, 4363.209)  # [OIII] 4363
    row[:,27] = LSF_lines(logc, totalcut, z, 5754.59)   # [NII] 5755
    row[:,28] = LSF_lines(logc, totalcut, z, 6312.06)   # [SIII] 6312
    row[:,29] = LSF_lines(logc, totalcut, z, (7330.19 + 7319.92)/2)     # [OII] 7320,7330
    row[:,30] = LSF_lines(logc, totalcut, z, 9071.1)     # [SIIId] 9071
    row[:,31] = LSF_lines(logc, totalcut, z, 9533.2)     # [SIIId] 9533
     

    # Get emmision line properties of Ha, hb, o2, n2, s2, o3, o1
    for j in range(len(emline_names)):
        row[:,32+9*j] = (maps['EMLINE_SEW'].data[emline_indx[j],:,:].flatten()[totalcut])          # summed equivalent width
        row[:,33+9*j] = (maps['EMLINE_SEW_IVAR'].data[emline_indx[j],:,:].flatten()[totalcut])  
        row[:,34+9*j] = (maps['EMLINE_SEW_MASK'].data[emline_indx[j],:,:].flatten()[totalcut])  
        
        row[:,35+9*j] = (maps['EMLINE_GEW'].data[emline_indx[j],:,:].flatten()[totalcut])      #Higher S/N if detect      # gauss equivalent width
        row[:,36+9*j] = (maps['EMLINE_GEW_IVAR'].data[emline_indx[j],:,:].flatten()[totalcut])  
        row[:,37+9*j] = (maps['EMLINE_GEW_MASK'].data[emline_indx[j],:,:].flatten()[totalcut])      
        
        row[:,38+9*j] = (maps['EMLINE_GFLUX'].data[emline_indx[j],:,:].flatten()[totalcut])        # gaussian flux
        row[:,39+9*j] = (maps['EMLINE_GFLUX_IVAR'].data[emline_indx[j],:,:].flatten()[totalcut])   # gaussian flux error
        row[:,40+9*j] = (maps['EMLINE_GFLUX_MASK'].data[emline_indx[j],:,:].flatten()[totalcut])   # same mask for flux and error

    table = vstack([table,Table(row,names=names,dtype=dtype)])
    maps.close()
    del maps
    logc.close()
    del logc

print('We have {} spaxels after cutting S/N < {}'.format(len(table),snrcut))
table.write('../Data/spaxel_data_table.fits', format='fits',overwrite=True)
log.close()
del log