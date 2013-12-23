# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 01:16:09 2013

@author: asharma6
"""

import lammpsio
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
from multiprocessing import Pool

class properties:
    def __init__(self,logfilename, *properties):
        """
        Args:
            filename:
                Filename of the LAMMPS logfile. Trajectory file will be added 
                later with the addition of properties based on trajectory analysis.
            properties:
                Properties you want to be evaluated. Right now, pymatgen only supports viscosity (argument: viscosity) evaluation.
                Example: properties ('log.lammps', 'viscosity') will return the viscosity of the system.
        """
        self.l=lammpsio.LammpsLog(logfilename,*properties)
        self.l.parselog()
        
        #Viscosity Calculation
        if 'viscosity' in properties:

            def autocorrelate (a):
                b=np.concatenate((a,np.zeros(len(a))),axis=1)
                c= np.fft.ifft(np.fft.fft(b)*np.conjugate(np.fft.fft(b))).real
                d=c[:len(c)/2]
                d=d/(np.array(range(len(a)))+1)[::-1]
                return d            
                
            NCORES=8
            p=Pool(NCORES)
            a1=self.l.LOG['pxy']
            a2=self.l.LOG['pxz']
            a3=self.l.LOG['pyz']
            a4=self.l.LOG['pxx']-self.l.LOG['pyy']
            a5=self.l.LOG['pyy']-self.l.LOG['pzz']
            a6=self.l.LOG['pxx']-self.l.LOG['pzz']
            array_array=[a1,a2,a3,a4,a5,a6]
            pv=p.map(autocorrelate,array_array)
            pcorr = (pv[0]+pv[1]+pv[2])/6+(pv[3]+pv[4]+pv[5])/24
            
            visco = (scipy.integrate.cumtrapz(pcorr,self.l.LOG['step'][:len(pcorr)]))*self.l.timestep*10**-15*1000*101325.**2*self.l.LOG['vol'][-1]*10**-30/(1.38*10**-23*self.l.temp)  
            plt.plot(np.array(self.l.LOG['step'][:len(pcorr)-1])*self.l.timestep,visco)
            plt.xlabel('Time (femtoseconds)')
            plt.ylabel('Viscosity (cp)')
            plt.savefig('viscosity_parallel.png')
        
            output=open('viscosity_parallel.txt','w')
            output.write('#Time (fs), Average Pressure Correlation (atm^2), Viscosity (cp)\n')
            for line in zip(np.array(self.l.LOG['step'][:len(pcorr)-1])*self.l.timestep-self.l.cutoff,pcorr,visco):
              output.write(' '.join(str(x) for x in line)+'\n')
            output.close()

