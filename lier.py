# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 18:10:57 2022

@author: Leo Lee

This is the module contains handy fuctions used in the LIER project
"""

#%%
#   Inport modules needed

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import astropy.constants

c_vel = astropy.constants.c.to('km/s').value

#%%
#   Define a class to hold the information of the emission lines

#   w: vacuum wavelength of the emission line
#   centre: centre window of the emission line
#   left: left window of the emission line
#   right: right window of the emission line
#   name: name of the line
#   type: whether the line is classified as weak or strong
#   ion: the line is caused by which atom/ion
class singlet():
    def __init__(self, w, left, right, name, t, ion):
        self.w = w
        self.left = left
        self.right = right
        self.name = name
        self.type = t
        self.ion = ion
        self.old = [0,0,0]
                        
    def info(self):
        print('Name: {}'.format(self.name))
        print('Wavelength: {} A'.format(self.w))
        print('Left continuum windows: {} to {} A'.format(self.left[0], self.left[1]))
        print('Right continuum windows: {} to {} A'.format(self.right[0], self.right[1]))    
        print('Line type: {}'.format(self.type))
        
        
#   w1: vacuum wavelength of the emission line with shorter wavelength
#   w2: vacuum wavelength of the emission line with longer wavelength
#   centre: centre window of the emission line
#   left: left window of the emission line
#   right: right window of the emission line
#   name: name of the line
#   type: whether the line is classified as weak or strong
#   ion: the line is caused by which atom/ion
class doublet():
    def __init__(self, w1, w2, left, right, name, t, ion):
        self.w1 = w1
        self.w2 = w2
        self.w = (w1+w2)/2
        self.left = left
        self.right = right
        self.name = name
        self.type = t
        self.ion = ion
        self.old = [0,0,0]
    
    def info(self):
        print('\nName: {}'.format(self.name))
        print('Wavelength: {} and {} A'.format(self.w1, self.w2))
        print('Left continuum windows: {} to {} A'.format(self.left[0], self.left[1]))
        print('Right continuum windows: {} to {} A'.format(self.right[0], self.right[1]))
        print('Line type: {}'.format(self.type))


