# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 16:52:27 2022

@author: Leo Lee (modified from code created by Gerome Algodon)

LIER project data analysis pipeline 
1. Galaxy selection

Input: drpall and dapall from MANGA (2 fits file)
Output: 
    3 plots(.jpg):
        Figure 1.1: Dn4000 Vs r band magnitude(from NSA) 
        Figure 1.2: Dn4000 Vs r band magnitude in 12 contour levels, with the Bounding lines selecting red galaxies
        Figure 1.3: Dn4000 Vs r band magnitude of the selected portion of galaxies
    1 fits file (quiscent_red_galaxies.fits)
"""

#%%
#   Importing required packages
import numpy as np 
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib as mpl


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
#   1.1 Selection of red quiscent galaxies

#   Select red quiscent galaxies using the graph of Dn4000 Vs r band flux 

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
