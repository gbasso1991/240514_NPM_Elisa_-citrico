#%%!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OWON con python

Created on Tue Oct 17 10:39:52 2023

@author: giuliano
https://open.spotify.com/intl-es/album/6BkeUWI72Lssc077AxqQek?si=fMfHdBnTQR-NP4Gvs03ECg

La idea es:
    -> run routine
    -> save 1 file per second with average of 10/20 measurements
    -> stop routine
"""
from vds1022 import *
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from astropy.io import ascii
from datetime import datetime
from astropy.table import Table, Column, MaskedColumn
import os
import fnmatch

#%%              
def lector_archivos(path):
    '''Toma archivos .txt con datos en columna:
        |t| |CH1| |CH2|
        
        Devuelve arrays de numpy con dicha info 
    '''
    
    data = pd.read_table(path,sep='\t',header=0,
                        names=('t','CH1','CH2'),usecols=(0,1,2),
                        decimal=',',engine='python') 

    t = pd.Series(data['t']).to_numpy(dtype=float)
    CH1= pd.Series(data['CH1']).to_numpy(dtype=float)
    CH2= pd.Series(data['CH2']).to_numpy(dtype=float)
    
    return t,CH1,CH2

#Multiple measurements: uso fetch_iter() en loop


def iter_plotter(x,y,g,h,indx,plot_range=0.04):
    
    fig,(ax0,ax1)=plt.subplots(nrows=2,figsize=(9,6),constrained_layout=True)
    ax0.plot(x,y,'-',c='tab:blue')
    ax0.grid()
    ax0.set_title('CH1',loc='left')
    ax0.set_ylim(-plot_range,plot_range)
    ax0.set_xlim(0,max(x)/5)
    
    ax1.plot(g,h,'.-',c='tab:orange')
    ax1.grid()
    ax1.set_title('CH2',loc='left')
    ax1.set_ylim(-20,20)
    ax1.set_xlim(0,max(g)/5)
    
    plt.suptitle(indx)  
    plt.show()                 


def promediador_señales(freq=20, num_prom=20, name='xxxkHz_yyydA_100Mss_ciclo', subdir=True, num_ciclos=False,plot_range=0.04):
    '''
    Durante medida iterativa, promedio un numero de ciclos x segundos especificado
    Luego guardo archivo ascii con:
        |t| |CH1| |CH2|
    Parameters
    ----------
    num_prom : numero de frames que promedio por segundo
               Default es 10.
    
    Salva 1 ciclo 
    
    '''    
    encabezado = ['t', 'CH1', 'CH2']
    formato = {'t': '%e', 'CH1': '%e', 'CH2': '%e'}
       
    t = np.arange(5000) * 1e-8
    y1 = np.zeros(5000)
    y2 = np.zeros(5000)
    if subdir:
        datename = datetime.today().strftime('%y%m%d_%H%M%S')
        if not os.path.exists(datename):
            os.makedirs(datename)
    indx_corte = 0
    for indx, frame in enumerate(owon.fetch_iter(freq=freq)):
        y1 += frame.ch1.y()
        y2 += frame.ch2.y()
        
        if (indx > 0 and indx % (num_prom - 1) == 0):
            print('-'*50)
            print(int(indx/(num_prom - 1)), frame.ch1.describe())
            y1 = y1 / num_prom
            y2 = y2 / num_prom
            
            iter_plotter(t, y1, t, y2,int(indx/(num_prom - 1)),plot_range=plot_range)
            ciclo_out = Table([t, y1, y2])
            
            ascii.write(ciclo_out, os.path.join(os.getcwd(), datename, name + str(indx // 20).zfill(3) + '.txt'),
                        names=encabezado, overwrite=True, delimiter='\t',
                        formats=formato)
            indx_corte += 1
        
        if num_ciclos and indx_corte == num_ciclos:  # Verifica si se ha alcanzado el número de ciclos deseado
            print('-'*50,'\nAdquisición finalizada')
            print(f'Se han salvado {num_ciclos} archivos')
            break  


def ploteo_en_iteracion(freq=10):
    for indx,frame in enumerate(owon.fetch_iter(freq=50)):
        print(indx) 
        x,y=frame.ch1.x()-frame.ch1.x()[0],frame.ch1.y()
        g,h=frame.ch2.x()-frame.ch2.x()[0],frame.ch2.y()
        iter_plotter(x,y,g,h,indx)

           
#%% Osciloscopio

owon = VDS1022(debug=False)
owon.set_sampling('100M')
owon.set_channel(channel=CH1,range='0.20v',offset=0.5,probe='1x',coupling=AC)
owon.set_channel(channel=CH2,range='50v',offset=0.5,probe='1x',coupling=AC)
owon.set_trigger(source=CH2,mode=EDGE,condition=RISE,position=0.5)

#%
#Single Measure
frames = owon.fetch()
frames.plot(backend='matplotlib')
 
#%% Ejecuto funcion de promediado / salva ciclos
promediador_señales(freq=20,num_prom=20, 
                    name='265kHz_150dA_100Mss_FF1',
                    plot_range=0.10)

#%% Para desconectar Osciloscopio
owon.dispose()

#%% Ploteo en iteracion s/ salvar ciclos
for indx,frame in enumerate(owon.fetch_iter(freq=5)):
    print(indx) 
    x,y=frame.ch1.x()-frame.ch1.x()[0],frame.ch1.y()
    g,h=frame.ch2.x()-frame.ch2.x()[0],frame.ch2.y()
    iter_plotter(x,y,g,h,indx,plot_range=0.1)


#%%
