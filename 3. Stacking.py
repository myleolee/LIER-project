# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 23:55:01 2022


@author: Leo Lee (modified from code created by Gerome Algodon)

LIER project data analysis pipeline 
3. Stacking

Input: spaxel_data_table.fits, 3 Bins.fits files, 3 Bins_Control.fits files, all maps file, all logcube file
Output: 
    3 flux_bin.fits files
    3 flux_bin_control.fits files
    3 drizzled_bin.fits files
    3 drizzled_bin_control.fits files
    3 bin_stack.fits file
    3 control_bin_stack.fits file
    
Note: Please fill in the maximum lambda at Section 3.2 before running the code (default value: 9600)
"""

import time
t = time.time()
# =============================================================================
# Import data
# =============================================================================
from astropy.io import fits
import numpy as np
import sys
import extinction as ext
import lier
import astropy.constants
c_vel = astropy.constants.c.to('km/s').value

manga_path = '/home/rbyan'    # The location of the manga repository


spaxel_data_table = fits.open('../Data/spaxel_data_table.fits')
bin1,bin1_control = fits.open('../Data/Bin/Bin_1.fits'),fits.open('../Data/Bin/Bin_1_Control.fits')
bin2,bin2_control = fits.open('../Data/Bin/Bin_2.fits'),fits.open('../Data/Bin/Bin_2_Control.fits')
bin3,bin3_control = fits.open('../Data/Bin/Bin_3.fits'),fits.open('../Data/Bin/Bin_3_Control.fits')


#%%
#   3.1 Galactic extinction correction

#   1. Define a function for returning an array after applying the extinction curve

def corr_extinct(wave_array, bmv, flux_array):
     trans = 10**(ext.fitzpatrick99(wave_array, bmv, 3.1)/-2.512)
     return flux_array*trans

#%%
#   Define the length of the primary wavelength array and the number of velocity offset bins 
length_of_wave_array = 4563
number_of_velocity_offset_bins = 25
    
#   Define the function for getting the flux from the logcube and maps file

def get_flux(bin_num,name):
    
    #   1. Initialize the fits file and the lists of the fits file
    
    flux_bin = fits.HDUList()     
    flux_array, flux_corr_array, ivar_array, mask_array, LSFpre_array = [],[],[],[],[]
        
    #   2. Fill the initiallize array with empty arrays that to be filled later then combine all subbins into totbin
    
    for t in range(number_of_velocity_offset_bins):
        flux_array.append(np.zeros((len(bin_num[t].data), length_of_wave_array)))
        flux_corr_array.append(np.zeros((len(bin_num[t].data), length_of_wave_array)))
        ivar_array.append(np.zeros((len(bin_num[t].data), length_of_wave_array)))
        mask_array.append(np.zeros((len(bin_num[t].data), length_of_wave_array)))
        LSFpre_array.append(np.zeros((len(bin_num[t].data), length_of_wave_array)))
        if t==0:
            totbin = bin_num[t].data
            subbin = np.full(len(bin_num[t].data),t,dtype='int64')
            spxnum = np.arange(len(bin_num[t].data),dtype='int64')
        else:
            totbin = np.append(totbin,bin_num[t].data)
            subbin = np.append(subbin,np.full(len(bin_num[t].data),t,dtype='int64'))
            spxnum = np.append(spxnum,np.arange(len(bin_num[t].data),dtype='int64'))
    totbin,subbin,spxnum = totbin.astype(int),subbin.astype(int),spxnum.astype(int)
                
    #   3. Pull ID for every spaxel: plate-ifu = galaxy, spaxel_id = specific spaxel
    
    plate = spaxel_data_table[1].data['plate'][totbin]
    ifu = spaxel_data_table[1].data['ifu'][totbin]
    bmv = spaxel_data_table[1].data['gal_red_B-V'][totbin]
    spaxel_id = spaxel_data_table[1].data['emline_ID'][totbin]
        
    #   4. Combine plate and ifu to find unique galaxies 
    
    plate_ifu = np.zeros((len(plate)),dtype='U25')
    for j in range(len(plate)):
        plate_ifu[j] = '{}-{}'.format(plate[j],ifu[j])
    plate_ifu_unique, u_indx = np.unique(plate_ifu,return_index=True)
    plate_unique, ifu_unique = plate[u_indx], ifu[u_indx]
    bmv_unique = bmv[u_indx]
    print(len(plate_unique))
    print(len(ifu_unique))
    
    #   5. For each galaxies, pull out the corresponding spaxels    
    
    for i in range(len(plate_unique)):
        #   Show progress
        sys.stdout.write('\r'+('     Galaxy {}/{}'.format(i+1,len(plate_unique))))  
                                  
        #   Load files
        maps_file = '{0}/manga/spectro/analysis/v3_1_1/3.1.0/HYB10-MILESHC-MASTARHC2/{1}/{2}/manga-{1}-{2}-MAPS-HYB10-MILESHC-MASTARHC2.fits.gz'.format(manga_path, plate_unique[i], ifu_unique[i])
        logcube_file = '{0}/manga/spectro/redux/v3_1_1/{1}/stack/manga-{1}-{2}-LOGCUBE.fits.gz'.format(manga_path, plate_unique[i], ifu_unique[i])
        maps = fits.open(maps_file)
        log = fits.open(logcube_file)
            
        #   Make primary the wavelength array
        if len(flux_bin)==0:  
            flux_bin.append(fits.PrimaryHDU(log['WAVE'].data))
            
        #   Find all the spaxels that are in this galaxy 
        spx_in_gal = np.where(plate_ifu_unique[i]==plate_ifu)[0]
        
        
        #   For every spaxel in this unique galaxy pull out the required data
        for s in range(len(spx_in_gal)): 
            #   Find the coordinate for this spaxel using the maps file
            spaxel_coord = (spaxel_id[spx_in_gal][s]==maps['BINID'].data[3,:,:])
            assert spaxel_coord.sum()==1    # make sure there is only one match
                
            #   Pull out the needed arrays from the logcube file and place it in the f indx
            subbin_num = subbin[spx_in_gal[s]]  # find subbin number of this spaxel
            spxnum_num = spxnum[spx_in_gal[s]]  # find slot to put in this specific spaxel
            flux_array[subbin_num][spxnum_num,:] = log['FLUX'].data[:,spaxel_coord].flatten()
            
            #   Correct for galactic extinction for the flux correction array
            flux_corr_array[subbin_num][spxnum_num,:] = corr_extinct(log['WAVE'].data, bmv_unique[i], flux_array[subbin_num][spxnum_num,:])
            ivar_array[subbin_num][spxnum_num,:] = log['IVAR'].data[:,spaxel_coord].flatten()
            mask_array[subbin_num][spxnum_num,:] = log['MASK'].data[:,spaxel_coord].flatten()
            LSFpre_array[subbin_num][spxnum_num,:] = log['LSFPRE'].data[:,spaxel_coord].flatten()
                
    print()
    
    #   6. After getting all the data, write it into a single fits file
    
    for b in range(len(flux_array)):
        sys.stdout.write('\r'+('     Sub-bin {}/{}'.format(b+1,len(flux_array)))) 
        flux_bin.append(fits.ImageHDU(flux_array[b],name='FLUX_SUBBIN_{}'.format(b+1)))
        flux_bin.append(fits.ImageHDU(flux_corr_array[b],name='FLUX_CORR_SUBBIN_{}'.format(b+1)))
        flux_bin.append(fits.ImageHDU(ivar_array[b],name='IVAR_SUBBIN_{}'.format(b+1)))
        flux_bin.append(fits.ImageHDU(mask_array[b],name='MASK_SUBBIN_{}'.format(b+1)))
        flux_bin.append(fits.ImageHDU(LSFpre_array[b],name='LSFPRE_SUBBIN_{}'.format(b+1)))
    
    #   7. Save the result as a fits file once we went through all the subbin
    
    flux_bin.writeto('../Data/Flux/{}.fits'.format(name),overwrite=True)
    

#%%
#   Get the flux of all the spaxels used in the analysis

print('\nWorking on Bin 1')
get_flux(bin1,'flux_bin_1')
print('\nWorking on Bin 2')
get_flux(bin2,'flux_bin_2')
print('\nWorking on Bin 3')
get_flux(bin3,'flux_bin_3')
print('\nWorking on Bin 1 Control')
get_flux(bin1_control,'flux_bin_1_control')
print('\nWorking on Bin 2 Control')
get_flux(bin2_control,'flux_bin_2_control')
print('\nWorking on Bin 3 Control')
get_flux(bin3_control,'flux_bin_3_control')       

#%%
#   Read the flux files
flux_bin1, flux_bin1_control = fits.open('../Data/Flux/flux_bin_1.fits'), fits.open('../Data/Flux/flux_bin_1_control.fits')
flux_bin2, flux_bin2_control = fits.open('../Data/Flux/flux_bin_2.fits'), fits.open('../Data/Flux/flux_bin_2_control.fits')
flux_bin3, flux_bin3_control = fits.open('../Data/Flux/flux_bin_3.fits'), fits.open('../Data/Flux/flux_bin_3_control.fits')
    
#%%
#   3.2 Spaxels Drizzling

#   1. Find out length of wavelength array after drizzle 

maximum_lambda = 9600
wave = flux_bin1[0].data
wave_array_length = len(np.where(wave<maximum_lambda)[0])
max_z = lier.round_down((wave[-1] / maximum_lambda)-1, 3)

#   Define the function for producing the drizzled file
def drizzle(bin_num,flux_num, name): 
    dri_flux_bin = fits.HDUList()
    dri_flux, dri_flux_mask, dri_var, dri_var_mask = [], [], [], []
    wave = flux_num[0].data
    wave_n = wave[wave < maximum_lambda]   #   wave_n is the version of wave after drizzle
    dri_flux_bin.append(fits.ImageHDU(wave_n))
    
    for s in range(len(bin_num)-1):    #   s stands for subbin number
        #   Get required data
        z = (spaxel_data_table[1].data['z_vel'][bin_num[s].data])/c_vel
        stell_vel = np.ma.MaskedArray(spaxel_data_table[1].data['stell_vel'],
                                      mask=spaxel_data_table[1].data['stell_vel_mask'] > 0)[bin_num[s].data]
        vel_off = bin_num['AVG_OFFSET_SUBBIN'].data[s]
        flux =  np.ma.MaskedArray(flux_num['FLUX_CORR_SUBBIN_{}'.format(s+1)].data,
                                  mask=flux_num['MASK_SUBBIN_{}'.format(s+1)].data>0)
        var =  np.ma.power(np.ma.MaskedArray(flux_num['IVAR_SUBBIN_{}'.format(s+1)].data,
                                  mask=flux_num['MASK_SUBBIN_{}'.format(s+1)].data>0),-1)
    
        if (np.any(flux.mask!=var.mask)):
            print('The mask of flux and variance are different, subbin {}'.format(s+1))
        
        #   Initialize arrays for holding results
        flux_n = np.zeros(( len(flux) , len(wave_n) ))   # len(flux) is the nummber of spaxels in the subbin
        f_weight = np.zeros(( len(flux) , len(wave_n) ))    # and len(wave_n) is the length of the new wave array
        var_n = np.zeros(( len(flux) , len(wave_n) )) 
        v_weight = np.zeros(( len(flux) , len(wave_n) ))
        
        #   Apply the drizzle code on every spaxels
        for p in range(len(flux)):
            zp = (1+(stell_vel[p] - vel_off)/c_vel)*(1+z[p])-1    # redshift at spaxel p, the spectra is shifted according to the gas velocity (emission lines by the gas are aligned)
            flux_n[p,:], f_weight[p,:] = specdrizzle_fast(wave, flux.data[p,:], zp, wave_n,mask=flux.mask[p,:].astype(int))
            var_n[p,:], v_weight[p,:] = specdrizzle_fast(wave, var.data[p,:], zp, wave_n,mask=var.mask[p,:].astype(int))
            sys.stdout.write('\r'+( '     Drizzing spaxel {}/{} of sub-bin {}/{}'.format(p+1,len(flux),s+1,len(bin_num)-1))) 
        
        #   Add the drizzled array of the subbin to the overall array
        dri_flux.append(flux_n)
        dri_flux_mask.append(f_weight==0)
        dri_var.append(var_n)
        dri_var_mask.append(v_weight==0)
    
    #    After getting all the data, write it into a single fits file
    for b in range(len(bin_num)-1):
        sys.stdout.write('\r'+('     Sub-bin {}/{}'.format(b+1,len(bin_num)-1))) 
        dri_flux_bin.append(fits.ImageHDU(dri_flux[b],name='DRIZZLED_FLUX_SUBBIN_{}'.format(b+1)))
        dri_flux_bin.append(fits.ImageHDU((dri_flux_mask[b].astype(int)),name='DRIZZLED_FLUX_MASK_SUBBIN_{}'.format(b+1)))
        dri_flux_bin.append(fits.ImageHDU(dri_var[b],name='DRIZZLED_VAR_SUBBIN_{}'.format(b+1)))
        dri_flux_bin.append(fits.ImageHDU((dri_var_mask[b].astype(int)),name='DRIZZLED_VAR_MASK_SUBBIN_{}'.format(b+1)))
    
    #   Save the result as a fits file once we went through all the subbin
    dri_flux_bin.writeto('../Data/Flux/{}.fits'.format(name),overwrite=True)    

#   2. This function is written by Francesco, to drizzle the wave spectra of a single spaxel

def specdrizzle_fast(wave, spec, z, wave_n, mask=None, flux=True):
# wave   - input wavelength array 
# spec  - input spectrum of the specific spaxel
# z     - redshift (Gas velocity relative to center of galaxy and galaxy redshift)
# wave_n - output rest frame wavelength (Shortened)
# spec_n - output drizzled spectrum
# weight - output weight of the drizzled spectrum
# mask  - mask of input spectrum (good spaxels = 0)
# flux  - 1 if total flux of the spectrum is conserved in the drizzling procedure

    #    Initialize output spectrum
    spec_n = np.zeros(len(wave_n))
    weight = np.zeros(len(wave_n))
    
    #    All pixels are good if mask not defined
    if mask is None:
        mask = spec*0.
        print('not masking')
    
    #   Conserve flux after drizzle
    specz = spec*(1.0+z)    #   Consider a 2D histogram, the wavelength array (x axis) is shortened by 1+z, therefore the flux (y axis) are mutiplied by 1+z

    #   Invert and convert the input mask array into binary (1 = good spaxels)
    mask_inv = np.where(mask==0, 1, 0)
    
    #   Turn the original wavelength array into wavelength array before redshifting 
    wave = wave/(1.0+z)
    wl_lo, wl_hi = _specbin(wave)
    dwlz = wl_hi - wl_lo    #   This is the spacing of the wavelength array before redshifting
  
    wldrz_lo, wldrz_hi= _specbin(wave_n)
    dwldrz = wldrz_hi - wldrz_lo   #    This is the spacing of the wavelength array after drizzling
  
    
    #   Calculate the new spectral flux and uncertainty values, loop over the new bins
    for j in range(wave_n.shape[0]):    #   j is the bin number of new bin array
        
        #   Find the first old bin which is partially covered by the new bin
        start_v = np.where(wl_hi > wldrz_lo[j])[0]
        if len(start_v) == 0:
            spec_n[j] = 0.
            weight[j] = 0
            continue
        else:
            start=start_v[0]
            
        #   Find the last old bin which is partially covered by the new bin
        stop = np.where(wl_lo <= wldrz_hi[j])[0][-1]
        
        #   If the new bin falls entirely within one old bin, the two bins are the same, therefore the new flux and new error are the same as for that bin
        if stop == start:
            spec_n[j]= specz[start]*mask_inv[start]
            weight[j] = mask_inv[start]
      
        #   Else, multiply the first and last old bin widths by P_ij(the proportion of the bin which is in the new bin), all the ones in between have P_ij = 1 
        else:
            start_factor = (wl_hi[start] - wldrz_lo[j])/(dwlz[start])   #   The proportion of the starting old bin which is in the new bin
            end_factor = (wldrz_hi[j] - wl_lo[stop])/(dwlz[stop])   #   The proportion of the stopping old bin which is in the new bin

            dwlz[start] *= start_factor
            dwlz[stop] *= end_factor

            #   The weight is given by sum(the proportion of the spacing of each bin* mask of each bin)/the spacing of the new bin, each mask of each bin = 1, weight = 1, (after drizzle, the old bin length should be same as new bin length)
            weight[j] = np.sum(dwlz[start:stop+1]*mask_inv[start:stop+1])/dwldrz[j]
            if weight[j]>0:
                #   3. Get the flux of the spaxel after drizzle, if one of the spaxels are invalid, the proportion is not counted
                
                spec_n[j] = np.sum(dwlz[start:stop+1]*specz[start:stop+1]*mask_inv[start:stop+1])/np.sum(dwlz[start:stop+1]*mask_inv[start:stop+1])
         
            # Put back the old bin widths to their initial values for later use
            dwlz[start] /= start_factor
            dwlz[stop] /= end_factor 
            
    return spec_n, weight

#   Given a wavelength array, return the upper and lower limit of each bins
def _specbin(wave):
  #     The array is filled by ([wave[1]-wave[0], wave[1]-wave[0], wave[1+n]-wave[n]...]) 
  dwl_lo = wave - np.roll(wave, 1)
  dwl_lo[0] = dwl_lo[1]

  #     The array is filled by ([wave[1]-wave[0], wave[2]-wave[1], wave[1+n]-wave[n]...]) 
  dwl_hi = np.roll(wave, -1) - wave
  dwl_hi[-1] = dwl_hi[-2]

  #     Find the upper and lower limits of the each spectral bins
  wl_lo = wave - dwl_lo/2.0
  wl_hi = wave + dwl_hi/2.0
  return wl_lo, wl_hi

#   Drizzle the strong line and control lines spaxels
drizzle(bin1, flux_bin1, 'drizzled_bin_1_{}.fits'.format(maximum_lambda))
drizzle(bin2, flux_bin2, 'drizzled_bin_2_{}.fits'.format(maximum_lambda))
drizzle(bin3, flux_bin3, 'drizzled_bin_3_{}.fits'.format(maximum_lambda))
drizzle(bin1_control, flux_bin1_control, 'drizzled_bin_1_control_{}.fits'.format(maximum_lambda))
drizzle(bin2_control, flux_bin2_control, 'drizzled_bin_2_control_{}.fits'.format(maximum_lambda))
drizzle(bin3_control, flux_bin3_control, 'drizzled_bin_3_control_{}.fits'.format(maximum_lambda))

#%%
#   Read the drizzled flux files

drizzled_flux_bin1 , drizzled_flux_bin1_control = fits.open('../Data/Flux/drizzled_bin_1_{}.fits'.format(maximum_lambda)), fits.open('../Data/Flux/drizzled_bin_1_control_{}.fits'.format(maximum_lambda))
drizzled_flux_bin2 , drizzled_flux_bin2_control = fits.open('../Data/Flux/drizzled_bin_2_{}.fits'.format(maximum_lambda)), fits.open('../Data/Flux/drizzled_bin_2_control_{}.fits'.format(maximum_lambda))
drizzled_flux_bin3 , drizzled_flux_bin3_control = fits.open('../Data/Flux/drizzled_bin_3_{}.fits'.format(maximum_lambda)), fits.open('../Data/Flux/drizzled_bin_3_control_{}.fits'.format(maximum_lambda))

#%%
#   3.3 Stacking

#   Define the function for stacking the spaxels of a velocity offset bin
def stack(d_bin, name):
    wave = d_bin[0].data
    
    #   1. Define the flat region around 6000A
    
    flat = (wave>6000)&(wave<6100)
    
    #   2. Initialize the array for holding the output stacked spectra
    
    stack_flux = np.zeros(( number_of_velocity_offset_bins, len(wave)))
    stack_flux_mask = np.zeros(( number_of_velocity_offset_bins, len(wave) ),dtype='i4')
    stack_var = np.zeros(( number_of_velocity_offset_bins, len(wave) ))  
    stack_var_mask = np.zeros(( number_of_velocity_offset_bins, len(wave) ),dtype='i4')
    
    for s in range(number_of_velocity_offset_bins):
        #   3. Extract drizzled arrays from the fits file
        
        flux_driz = np.ma.MaskedArray(d_bin['DRIZZLED_FLUX_SUBBIN_{}'.format(s+1)].data,
                                      mask=d_bin['DRIZZLED_FLUX_MASK_SUBBIN_{}'.format(s+1)])
        var_driz = np.ma.MaskedArray(d_bin['DRIZZLED_VAR_SUBBIN_{}'.format(s+1)].data,
                                      mask=d_bin['DRIZZLED_VAR_MASK_SUBBIN_{}'.format(s+1)])
        sys.stdout.write('\r'+'Stacking subbin {}/{}'.format(s+1, 25))
        
        #   4.  Normalize the spectrum of each spaxels according to the flat region around 6000A
        med = np.median(flux_driz[:,flat].data,axis=1)
        flux_norm = flux_driz/med[:,None]
        var_norm = var_driz/(med[:,None]**2)
        
        #   5.  Average the flux of the velocity offset subbin
        totalflux_norm = np.ma.average(flux_norm,axis=0)
        totalvar_norm = np.ma.sum(var_norm,axis=0)/((np.sum(np.invert(var_norm.mask),axis=0)).astype('int64'))**2
        
        stack_flux[s,:],stack_var[s,:] = totalflux_norm.data, totalvar_norm.data
        stack_flux_mask[s,:],stack_var_mask[s,:] = totalflux_norm.mask.astype(int), totalvar_norm.mask.astype(int)
    
    fit = fits.HDUList()
    fit.append(fits.PrimaryHDU(wave))
    fit.append(fits.ImageHDU(stack_flux,name='stacked_flux'))
    fit.append(fits.ImageHDU(stack_flux_mask,name='stacked_flux_mask'))
    fit.append(fits.ImageHDU(stack_var,name='stacked_var'))
    fit.append(fits.ImageHDU(stack_var_mask,name='stacked_var_mask'))
    fit.writeto('../Data/Stack/{}.fits'.format(name),overwrite=True)

#%%
#   Apply the stack functions on the metallicty bins

stack(drizzled_flux_bin1, 'bin_1_stack')
stack(drizzled_flux_bin2, 'bin_2_stack')
stack(drizzled_flux_bin3, 'bin_3_stack')
stack(drizzled_flux_bin1_control, 'control_bin_1_stack')
stack(drizzled_flux_bin2_control, 'control_bin_2_stack')
stack(drizzled_flux_bin3_control, 'control_bin_3_stack')
