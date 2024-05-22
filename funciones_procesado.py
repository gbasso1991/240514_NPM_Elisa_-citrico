#Packages y funciones del archivo possessor.py para utilizar en otros scripts

import time
from datetime import datetime
from numpy.core.numeric import indices 
import fnmatch
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import scipy as sc
import os
from scipy.signal import find_peaks 
from scipy.integrate import cumulative_trapezoid, trapezoid

from scipy.fft import fft, ifft, rfftfreq,irfft 
from astropy.io import ascii
from astropy.table import Table, Column, MaskedColumn
from sklearn.metrics import r2_score
from pprint import pprint
from uncertainties import ufloat, unumpy
'''Funcion: fft_smooth()'''

def fft_smooth(data_v,freq_n):
    """
    fft low pass filter para suavizar la señal. 
    data_v: datos a filtrar (array)
    frec_n: numero N de armonicos que conservo: primeros N 
    y ultimos N. 
    """
    fft_data_v = fft(np.array(data_v))
    s_fft_data_v = np.zeros(len(data_v),dtype=complex)
    s_fft_data_v[0:int(freq_n)] = fft_data_v[0:int(freq_n)]
    s_fft_data_v[-1-int(freq_n): ] = fft_data_v[-1-int(freq_n):] 
    s_data_v = np.real(ifft(s_fft_data_v))
    
    return s_data_v
     
'''Funcion: medida_cruda(path,delta_t).'''
def medida_cruda(path,delta_t):
    '''
    Funcion para levantar los archivos .txt en df de pandas.\n
    Identifica las dos señales de cada canal como señal y referencia.\n
    Recibe datos en mV y pasa a V. 
    '''
    df = pd.read_table(path,skiprows=4,names=['idx','v','v_r'],
                                    engine='python',sep='\s+')
    #paso de mV a V
    df['v'] = df['v']*0.001   
    df['v_r'] = df['v_r']*0.001 
    #agrego col de tiempos, la hago arrancar de 0 
    df.insert(loc=1,column='t',value=delta_t*(df['idx']-df['idx'][0]))
    # elimino columna de indice
    del df['idx'] 
    return df

def medida_cruda_autom(path,*kwargs):
    '''
    Funcion para levantar los archivos .txt en df de pandas.\n
    Identifica las dos señales de cada canal como señal y referencia.\n
    Recibe datos en mV y pasa a V. 
    '''
    df = pd.read_table(path,skiprows=1,names=['t','v','v_r'],
                                    engine='python',sep='\s+')
    #paso de mV a V
    # df['v'] = df['v']#*0.001   
    # df['v_r'] = df['v_r']#*0.001 
    #agrego col de tiempos, la hago arrancar de 0 
    # df.insert(loc=1,column='t',value=delta_t*(df['idx']-df['idx'][0]))
    #elimino columna de indice
    # del df['idx'] 
    return df    


'''Funcion: ajusta_seno(). Utiliza subrutina sinusoide()'''    
def sinusoide(t,A,B,C,D):
    '''
    Crea sinusoide con params: 
        A=offset, B=amp, C=frec, D=fase
    '''
    return(A + B*np.sin(2*np.pi*C*t - D))

from scipy.optimize import curve_fit
def ajusta_seno(t,v_r):
    '''
    Calcula params de iniciacion y ajusta sinusoide via curve_fit
    Para calacular la frecuencia mide tiempo entre picos
    '''
    offset0 = v_r.mean() 
    amplitud0=(v_r.max()-v_r.min())/2

    v_r_suave = fft_smooth(v_r,np.around(int(len(v_r)*6/1000)))
    indices, _ = find_peaks(v_r_suave,height=0)
    t_entre_max = np.mean(np.diff(t[indices]))
    frecuencia0 = 1 /t_entre_max
    
    fase0 = 2*np.pi*frecuencia0*t[indices[0]] - np.pi/2

    p0 = [offset0,amplitud0,frecuencia0,fase0]
    
    coeficientes, _ = curve_fit(sinusoide,t,v_r,p0=p0)
    
    offset=coeficientes[0]
    amplitud = coeficientes[1]
    frecuencia = coeficientes[2]
    fase = coeficientes[3]

    return offset, amplitud , frecuencia, fase

'''Funcion: resta_inter() '''
def resta_inter(t,v,v_r,fase,frec,offset,t_f,v_f,v_r_f,fase_f,frec_f,graf):    
    '''
    Funcion para la interpolacion y resta. 
    Grafica si parametro graf = graficos[1] = 1
    Desplazamiento temporal para poner en fase las referencias, y
    resta de valores medios de referencias.
    '''    
    '''
    Calculo el modulo 2 pi de las fases y calcula el tiempo de fase 0 
    '''
    t_fase = np.mod(fase,2*np.pi)/(2*np.pi*frec)
    t_fase_f = np.mod(fase_f,2*np.pi)/(2*np.pi*frec_f)
    '''
    Desplaza en tiempo para que haya coincidencia de fase e/ referencias.
    La amplitud debería ser positiva siempre por el valor inicial 
    del parámetro de ajuste.
    '''
    t_1 = t - t_fase 
    t_f_aux = t_f - t_fase_f     
    '''
    Correccion por posible diferencia de frecuencias dilatando el 
    tiempo del fondo 
    '''
    t_f_mod = t_f_aux*frec_f/frec 
    '''
    Resta el offset de la referencia
    '''
    v_r_1 = v_r - offset
    '''
    Resta medida y fondo interpolando para que corresponda mejor el
    tiempo. No trabaja más con la medidas individuales, sólo con la
    resta. Se toma la precaución para los casos de trigger distinto
    entre fondo y medida. Comienza en fase 0 o sea t=0.
    '''
    t_min=0
    t_max=t_f_mod.iloc[-1]
    '''
    Recorta el tiempo a mirar
    '''
    t_aux = t_1[np.nonzero((t_1>=t_min) & (t_1<=t_max))]
    ''' 
    Interpola el fondo a ese tiempo
    '''
    interpolacion_aux = np.interp(t_aux,t_f_mod,v_f)
    interpolacion = np.empty_like(v)   
    '''
    Cambia índice viejo por índice nuevo en la base original
    '''
    for w in range(0,len(t_1),1):  
        #obtengo el indice donde esta el minimo de la resta: 
        index_min_m = np.argmin(abs(t_aux - t_1[w]))
        #defino c/ elemento de interpolacion: 
        interpolacion[w] = interpolacion_aux[index_min_m]
    '''
    Defino la resta entre la señal (m o c) y la interpolacion de la señal
    de fondo'''
    Resta = v - interpolacion
    '''
    Comparacion de las referencias de muestra y fondo desplazadas en
    tiempo y offset
    '''
    
    if graf!=0:
        '''Rutina de ploteo para resta_inter'''
        def ploteo(t_f_mod,v_r_f,t_1,v_r_1):
            fig = plt.figure(figsize=(10,8),constrained_layout=True)
            ax = fig.add_subplot(2,1,1)
            plt.plot(t_1,v_r_1,lw=1,label=str(graf).capitalize())
            plt.plot(t_f_mod,v_r_f,lw=1,label='Fondo')
            plt.legend(loc='best')
            plt.grid()
            plt.xlabel('t (s)')
            plt.ylabel('Referencia (V)')
            plt.title('Referencias desplazadas y restados sus offsets',loc='left')
            
            ax = fig.add_subplot(2,1,2)
            plt.plot(t_f_mod,v_f,lw=1,label='Fondo')
            plt.plot(t_1,v,lw=1,label=str(graf).capitalize())
            plt.plot(t_1,interpolacion,lw=1,label='Interpolacion del fondo al tiempo de la '+str(graf))
            plt.legend(loc='best')
            plt.grid()
            plt.xlabel('t (s)')
            plt.ylabel('Señal (V)')
            plt.title(str(graf).capitalize()+' y fondo',loc='left')
            #plt.savefig('Comparacion_de_medidas_'+str(graf)+'.png',dpi=300)
            fig.suptitle('Comparacion de señales',fontsize=20)
            return fig
        figura = ploteo(t_f_mod,v_r_f,t_1,v_r_1)
    else:
        figura = 'Figura off'

    return Resta , t_1 , v_r_1 , figura 

'''Funcion: encuentra_ruido().Es subrutina del filtro Actis, en filtrando_ruido()'''
def encuentra_ruido(t,v,ancho,entorno):
    from scipy.signal import lfilter
    '''
    Toma una señal (t,v) y calcula la derivada de v respecto de t.
    Valor absoluto medio: ruido_tranqui. 
    Marca los puntos con derivada en valor absoluto mayor 
    que "ancho" veces "ruido_tranqui" y un entorno de estos puntos 
    igual a "entorno" puntos para cada lado.
    '''
    '''Suaviza con un promedio leve'''
    WindowSize = 5
    be = (1/WindowSize)*np.ones(WindowSize)
    t_1 = t[WindowSize+1:]-WindowSize*(t[1]-t[0])/2
    v_fe = lfilter(be,1,v)
    v_1 = v_fe[WindowSize+1:] 

    '''Calcula la derivada de v respecto a t'''
    derivada = np.diff(v_1)/np.diff(t_1)
    t_2 = t_1[:-1]+(t_1[1]-t_1[0])/2

    '''Suaviza la derivada'''
    t_3 = t_2[WindowSize+1:] - WindowSize*(t_2[1]-t_2[0])/2
    derivada0 = lfilter(be,1,derivada)
    derivada2 = derivada0[WindowSize+1:]

    '''
    El ruido caracteristico de la señal es el valor medio 
    del valor absoluto de la derivada
    '''
    ruido_tranqui = np.mean(abs(derivada2))
    aux_1 = np.zeros(len(derivada2)+1)
    '''
    Marca puntos que superan en ancho veces el ruido normal
    '''
    for jq in range(len(derivada2)):
        if abs(derivada2[jq])>ancho*ruido_tranqui:
            aux_1[jq] = 1
        else:
            aux_1[jq] = 0    
    '''Prepara el marcador '''
    marcador = np.zeros_like(derivada2)
    '''
    Si hay un solo cambio de signo en la derivada, 
    no lo marca, ya que debe tratarse de un pico 
    natural de la señal
    ''' 
    for jq in range(entorno,len(derivada2)-entorno):
        if max(aux_1[jq-entorno:jq+entorno]) == 1:
            marcador[jq + int(np.round(entorno/2))] = 1
        else:
            marcador[jq + int(np.round(entorno/2))] = 0
    
    '''Acomodo los extremos '''
    for jq in range(entorno):
        if marcador[entorno+1] == 1:
            marcador[jq] = 1
        if marcador[len(derivada2)- entorno] == 1:
            marcador[len(derivada2)-jq-1]=1
        
    return t_3, marcador

'''Funcion: filtrando_ruido()'''
def filtrando_ruido(t,v_r,v,filtrar,graf):
    '''
    Sin filtrar: filtrarmuestra/cal = 0
    Actis:       filtrarmuestra/cal = 1
    Fourier:     filtrarmuestra/cal = 2   
    Fourier+Actis: filtrarmuestra/cal = 3   
    ''' 
    if filtrar == 0:
        t_2 = t
        v_r_2 = v_r
        v_2 = v
        figura_2 = 'No se aplico filtrado'

    elif filtrar==2 or filtrar==3: 
        '''Filtro por Fourier'''
        freq = np.around(len(v_r)/5)
        v_2 = fft_smooth(v,freq)
        v_r_2 = fft_smooth(v_r,freq)
        t_2 = t   

        if graf !=0:
            '''Control de que el suavizado final sea satisfactorio'''
            figura_2 = plt.figure(figsize=(10,8),constrained_layout=True)
            ax1 = figura_2.add_subplot(2,1,1)
            plt.plot(t,v_r,'.-',label='Sin filtrar')
            plt.plot(t_2,v_r_2,lw=1,label='Filtrada')
            plt.legend(ncol=2,loc='lower center')
            plt.grid()
            plt.title('Señal de referencia',loc='left',fontsize=15)
            plt.ylim(1.25*min(v_r_2),1.25*max(v_r_2))            
            ax2 = figura_2.add_subplot(2,1,2,sharex=ax1)
            plt.plot(t,v,'.-',label='Sin filtrar')
            plt.plot(t_2,v_2,lw=1,label='Filtrada')
            #plt.plot(t,v,lw=1,label='Zona de ruido')
            plt.legend(ncol=2,loc='lower center')
            plt.grid()
            plt.xlabel('t (s)')
            plt.title('Señal de '+ str(graf),loc='left',fontsize=15)  
            plt.xlim(t[0],t[-1]/4)#provisorio
            figura_2.suptitle('Filtro de Fourier - '+str(graf).capitalize(),fontsize=20)
            
        else:
            figura_2 = 'Figura off'   

    elif filtrar ==1: #filtro Actis
        '''
        Identifica el ruido: factor del ruido natural a partir del 
        cual se considera que hay que filtrar
        '''
        ancho=2.5
        '''
        Puntos a ambos lados del sector de señal ruidosa que serán
        incluidos en el ruido.
        '''
        entorno=5

        '''Aca ejecuto funcion encuentra_ruido(t,v,ancho,enterno)
        obtengo: t_2 y marcador'''
        t_2 , marcador = encuentra_ruido(t,v,ancho,entorno)

        '''
        Ajuste ruido
        Params: 
            ancho= puntos a cada lado de la region a filtrar que serán considerados para el ajuste
            grado_pol= grado del polinomio a ajustar.
        '''
        puntos_ajuste=80
        grado_pol=3
        '''Tiempos y señales en las mismas dimensiones que los marcadores'''
        interpolador = sc.interpolate.interp1d(t,v,kind='slinear') 
        v_2 = interpolador(t_2)

        interpolador_r = sc.interpolate.interp1d(t,v_r,kind='slinear') 
        v_r_2 = interpolador_r(t_2)

        '''
        Comienza a filtrar una vez que tiene suficientes puntos detras
        '''
        w=puntos_ajuste + 1
        '''
        Barre la señal. NO FILTRA ni el principio ni el final, 
        por eso mas adelante se eliminan 1er y ultimo semiperiodos.
        ''' 
        while w<len(v_2):
            
            if marcador[w-1]==0:
                '''Si no hay ruido deja la señal como estaba'''
                w += 1    
            elif marcador[w-1]==1:
                '''Si hay ruido'''
                q=w
                '''Busca hasta donde llega el ruido'''
                while marcador[q-1]==1 and q<len(v_2):
                    q+=1
                '''si se tienen suficientes puntos del otro lado
                realiza el ajuste'''
                if q<len(v_2)-puntos_ajuste:
                    y = np.concatenate((v_2[w-puntos_ajuste:w],v_2[1+q:1+q+puntos_ajuste]),dtype=float)
                    x = np.concatenate((t_2[w-puntos_ajuste:w],t_2[1+q:1+q+puntos_ajuste]),dtype=float)
                    p = np.polyfit(x,y,grado_pol)
                    v_2[w:q-1]= np.polyval(p,t_2[w:q-1])

                    y_r = np.concatenate((v_r_2[w-puntos_ajuste:w],v_r_2[1+q:1+q+puntos_ajuste]),dtype=float)
                    x_r = np.concatenate((t_2[w-puntos_ajuste:w],t_2[1+q:1+q+puntos_ajuste]),dtype=float)
                    p_r= np.polyfit(x_r,y_r,grado_pol)
                    v_r_2[w:q-1]= np.polyval(p_r,t_2[w:q-1])
                w=q 

        if graf !=0:
            '''Control de que el suavizado final sea satisfactorio'''
            figura_2 = plt.figure(figsize=(10,8),constrained_layout=True)
            ax1 = figura_2.add_subplot(2,1,1)
            plt.plot(t,v_r,'.-',label='Referencia de '+ str(graf))
            plt.plot(t_2,v_r_2,lw=1,label='Referencia de '+str(graf)+' filtrada')
            plt.legend(ncol=2,loc='lower center')
            plt.grid()
            plt.ylim(1.25*min(v_r_2),1.25*max(v_r_2))
            plt.title('Señal de referencia',loc='left',fontsize=15)
            
            ax2 = figura_2.add_subplot(2,1,2,sharex=ax1)
            plt.plot(t,v,'-',lw=1.5,label='Resta de señales')
            plt.plot(t_2,v_2,lw=0.9,label='Sin ruido')
            plt.plot(t_2,marcador,lw=1, alpha=0.8 ,label='Zona de ruido')
            #plt.plot(t,v,lw=1,label='Zona de ruido')
            plt.legend(ncol=3,loc='lower center')
            plt.grid()
            plt.xlabel('t (s)')
            plt.title('Señal de ' + str(graf),loc='left',fontsize=15)
            plt.xlim(t[0],t[-1]/4)#provisorio
            figura_2.suptitle('Filtro Actis - '+str(graf).capitalize(),fontsize=20)
            
        else:
            figura_2 = "Figura Off"
        
    return t_2 , v_r_2 , v_2 ,figura_2 

'''Funcion: recorte()'''
def recorte(t,v_r,v,frecuencia,graf):
    '''
    Recorta un numero entero de periodos o ciclos,arrancando en fase 0 (campo max o campo min segun polaridad)
    Grafico: señal de muestra/calibracion, s/ fondo, s/ valor medio y recortadas a un numero entero de ciclos.
    '''
    #Numero de ciclos
    N_ciclos =  int(np.floor(t[-1]*frecuencia)) 
    
    #Indices ciclo
    indices_recorte = np.nonzero(np.logical_and(t>=0,t<N_ciclos/frecuencia))  
    
    #Si quisiera recortar ciclos a ambos lados
    # largo = indices_ciclo[-1][0]
    #if np.mod(largo,N_ciclos) == 0:
    # largo = largo - np.mod(largo,N_ciclos)
    #elif np.mod(largo,N_ciclos) <= 0.5:
        #largo = largo - np.mod(largo,N_ciclos)
    #else:
    # largo = largo + N_ciclos - np.mod(largo,N_ciclos)
    '''
    Recorto los vectores
    '''
    t_2 = t[indices_recorte]
    v_2 = v[indices_recorte]
    v_r_2 = v_r[indices_recorte]
    if graf !=0:
        '''
        Señal de muestra/calibracion, s/ fondo, s/ valor medio y 
        recortadas a un numero entero de ciclos
        '''
        figura = plt.figure(figsize=(10,8),constrained_layout=True)
        ax1 = figura.add_subplot(2,1,1)
        plt.plot(t_2,v_r_2,'.-',lw=1)
        plt.grid()
        plt.title('Señal de referencia',loc='left',fontsize=15)
        plt.ylabel('Señal (V)')
        plt.axvspan(0,1/frecuencia, facecolor='#2ca02c',label='Período 1/{}'.format(N_ciclos),alpha=0.4)
        plt.ylim(1.3*min(v_r_2),1.3*max(v_r_2))            
        plt.legend(loc='lower left')
        ax2 = figura.add_subplot(2,1,2,sharex=ax1)
        plt.plot(t_2,v_2,'.-',lw=1)
        plt.axvspan(0,1/frecuencia, facecolor='#2ca02c',label='Período 1/{}'.format(N_ciclos),alpha=0.4)
        plt.legend(loc='lower left')
        plt.grid()
        plt.xlabel('t (s)')
        plt.ylabel('Señal (V)')
        plt.ylim(1.3*min(v_2),1.3*max(v_2))
        plt.xlim(0,(N_ciclos//2)/frecuencia)
        plt.title('Señal de '+ str(graf),loc='left',fontsize=15)  
        figura.suptitle('Número entero de períodos - '+str(graf).capitalize(),fontsize=20)
    else:
        figura ='Figura off'

    return t_2 , v_r_2 , v_2 , N_ciclos, figura

'''Funcion: promediado_ciclos()'''
def promediado_ciclos(t,v_r,v,frecuencia,N_ciclos):
    '''
    '''
    t_f = t[t<t[0]+1/frecuencia]
    v_r_f = np.zeros_like(t_f)
    v_f = np.zeros_like(t_f)
    for indx in range(N_ciclos):
        if t_f[-1] + indx/frecuencia < t[-1]: 
            interpolador_r = sc.interpolate.interp1d(t,v_r,kind='linear')
            interpolador = sc.interpolate.interp1d(t,v,kind='linear')
            v_r_f = v_r_f + interpolador_r(t_f + indx/frecuencia)/N_ciclos
            v_f = v_f + interpolador(t_f + indx/frecuencia)/N_ciclos

        else: #True en la ultima iteracion
            interpolador_r_2 = sc.interpolate.interp1d(t,v_r,kind='slinear')
            interpolador_2 = sc.interpolate.interp1d(t,v,kind='slinear')
            v_r_f = v_r_f + interpolador_r_2(t_f + (indx-1)/frecuencia)/N_ciclos
            v_f = v_f + interpolador_2(t_f + (indx-1)/frecuencia)/N_ciclos
    
    '''Quita valor medio'''
    v_f = v_f - v_f.mean()
    v_r_f = v_r_f - v_r_f.mean()
    '''Paso temporal'''
    delta_t = (t_f[-1]-t_f[0])/len(t_f)
    return t_f , v_r_f, v_f , delta_t

def fourier_señales(t,t_c,v,v_c,v_r_m,v_r_c,delta_t,polaridad,filtro,frec_limite,name):
    '''
    Toma señales de muestra, calibracion y referencia obtieniendo via fft frecuencias y fases.
    frec_muestreo = sample rate 1/delta_t (tipicamente 1e8 o 5e7).    
    Las señales indefectiblemente deben estar recortadas a N ciclos.
    Establecer frec limite permite filtrar la interferencia de alta señal del generador RF\n
    Se conoce la polaridad de la señal(del ajuste lineal sobre el ciclo paramagnetico). 
    '''
    t = t - t[0] #Muestra 
    t_r = t.copy() #y su referencia
    t_c = t_c - t_c[0] #Calibracion 
    t_r_c = t_c.copy() #y su referencia
    
    y = polaridad*v     #muestra (magnetizacion)
    y_c = polaridad*v_c #calibracion (magnetizacion del paramagneto)
    y_r = v_r_m        #referencia muestra (campo)
    y_r_c = v_r_c      #referencia calibracion (campo)
    
    N = len(v)
    N_c = len(v_c)
    N_r_m = len(v_r_m)
    N_r_c = len(v_r_c)
    
    #Para que el largo de los vectores coincida
    if len(t)<len(y): #alargo t
        t = np.pad(t,(0,delta_t*(len(y)-len(t))),mode='linear_ramp',end_values=(0,max(t)+delta_t*(len(y)-len(t))))
    elif len(t)>len(y):#recorto t    
        t=np.resize(t,len(y))

    if len(t_c)<len(y_c): #alargo t
        t_c = np.pad(t_c,(0,delta_t*(len(y_c)-len(t_c))),mode='linear_ramp',end_values=(0,max(t_c)+delta_t*(len(y_c)-len(t_c))))
    elif len(t_c)>len(y_c):#recorto t    
        t_c=np.resize(t_c,len(y_c))
    
    #Idem referencias
    if len(t_r)<len(y_r): #alargo t
        t_r = np.pad(t_r,(0,delta_t*(len(y_r)-len(t_r))),mode='linear_ramp',end_values=(0,max(t_r)+delta_t*(len(y_r)-len(t_r))))
    elif len(t_r)>len(y_r):#recorto t    
        t_r=np.resize(t_r,len(y_r))

    if len(t_r_c)<len(y_r_c): #alargo t
        t_r_c = np.pad(t_r_c,(0,delta_t*(len(y_r_c)-len(t_r_c))),mode='linear_ramp',end_values=(0,max(t_r_c)+delta_t*(len(y_r_c)-len(t_r_c))))
    elif len(t_r_c)>len(y_r_c):#recorto t    
        t_r_c=np.resize(t_r_c,len(y_r_c))

#Aplico transformada de Fourier
    f = rfftfreq(N,d=delta_t) #obtengo la mitad de los puntos, porque uso rfft
    f_HF = f.copy() 
    #f_HF = f_HF[np.nonzero(f>=frec_limite)] #aca estan el resto 
    f = f[np.nonzero(f<=frec_limite)] #limito frecuencias 
    g_aux = fft(y,norm='forward') 
    #“forward” applies the 1/n factor on the forward tranform
    g = abs(g_aux)  #magnitud    
    fase = np.angle(g_aux)
    
    #Idem p/ calibracion
    f_c = rfftfreq(N_c, d=delta_t)
    f_c_HF = f_c.copy()
    f_c = f_c[np.nonzero(f_c<=frec_limite)]
    g_c_aux = fft(y_c,norm='forward') 
    g_c = abs(g_c_aux)
    fase_c= np.angle(g_c_aux)
    
    #Idem p/ Referencia
    f_r = rfftfreq(N_r_m, d=delta_t)
    f_r = f_r[np.nonzero(f_r<=frec_limite)]
    g_r_aux = fft(y_r,norm='forward')
    g_r = abs(g_r_aux)
    fase_r = np.angle(g_r_aux)
    #y para ref de calibracion
    f_r_c = rfftfreq(N_r_c, d=delta_t)
    f_r_c = f_r_c[np.nonzero(f_r_c<=frec_limite)]
    g_r_c_aux = fft(y_r_c,norm='forward')
    g_r_c = abs(g_r_c_aux)
    fase_r_c = np.angle(g_r_c_aux)
     
    #Recorto vectores hasta frec_limite
    g_HF = g.copy() 
    g_c_HF = g_c.copy()
    #g_HF = g_HF[np.nonzero(f>=frec_limite)]#magnitud de HF
    g = np.resize(g,len(f))
    g_c = np.resize(g_c,len(f_c))
    g_r = np.resize(g_r,len(f_r))
    g_r_c = np.resize(g_r_c,len(f_r_c))
    g_HF = np.resize(g_HF,len(f_HF))
    g_HF[np.argwhere(f_HF<=frec_limite)]=0 #Anulo LF
    g_c_HF = np.resize(g_c_HF,len(f_c_HF))
    g_c_HF[np.argwhere(f_c_HF<=frec_limite)]=0 #Anulo LF

#Obtengo frecuencias cuya intensidad relativa supera umbral dado por el filtro
    indices,_=find_peaks(abs(g),threshold=max(g)*filtro)
    # anulo armonico fundamental descomentando siguiente linea
    #indices = np.delete(indices,0)
    
    indices_c,_=find_peaks(abs(g_c),threshold=max(g_c)*filtro)

    indices_r,_=find_peaks(abs(g_r),threshold=max(g_r)*filtro)

    indices_r_c,_=find_peaks(abs(g_r_c),threshold=max(g_r_c)*filtro)

#En caso de frecuencia anomala menor que la fundamental en Muestra
    #if f[indices[0]]<f_r[indices_r[0]]:
    #    print('ATENCION: detectada subfrecuencia anómala en el espectro de la señal de muestra {:.2f} Hz\n'.format(f[indices[0]]))
    #    indices = np.delete(indices,0) 
    #else:
        #pass reeemplazado por lineas siguiente 14 Mar 2022
    
    for elem in indices:
        if f[elem]<0.9*f_r[indices_r[0]]:
            print('ATENCION: detectada subfrecuencia anómala en el espectro de la señal de muestra {:.2f} Hz\n'.format(f[elem]))
            indices = np.delete(indices,0)
            
    for elem in indices_c:
        if f_c[elem]<0.9*f_r[indices_r[0]]:
            print('ATENCION: detectada subfrecuencia anómala en el espectro de la señal de calibracion {:.2f} Hz\n'.format(f_c[elem]))
            indices_c = np.delete(indices_c,0)
            
    armonicos = f[indices]
    amplitudes = g[indices]
    fases = fase[indices]

    armonicos_c = f_c[indices_c]
    amplitudes_c = g_c[indices_c]
    fases_c = fase_c[indices_c]

    armonicos_r = f_r[indices_r]
    amplitudes_r = g_r[indices_r]
    fases_r = fase_r[indices_r]

    armonicos_r_c = f_r_c[indices_r_c]
    amplitudes_r_c = g_r_c[indices_r_c]
    fases_r_c = fase_r_c[indices_r_c]
#Imprimo tabla 
    print('''Espectro de la señal de referencia:\nFrecuencia (Hz) - Intensidad rel - Fase (rad)''')
    for i in range(len(indices_r)):
        print(f'{armonicos_r[i]:<10.2f}    {amplitudes_r[i]/max(amplitudes_r):>12.2f}    {fases_r[i]:>12.4f}')
    
    print('''\nEspectro de la señal de calibracion:\nFrecuencia (Hz) - Intensidad rel - Fase (rad)''')
    for i in range(len(indices_c)):
        print(f'{armonicos_c[i]:<10.2f}    {amplitudes_c[i]/max(amplitudes_c):>12.2f}    {fases_c[i]:>12.4f}')
    
    print('''\nEspectro de la señal de muestra:\nFrecuencia (Hz) - Intensidad rel - Fase (rad)''')
    for i in range(len(indices)):
        print(f'{armonicos[i]:<10.2f}    {amplitudes[i]/max(amplitudes):>12.2f}    {fases[i]:>12.4f}')
  
#Frecuencias/indices multiplo impar/par de la fundamental
    frec_multip = []
    indx_impar = []
    indx_par=[]

    for n in range(int(frec_limite//int(armonicos[0]))):
        frec_multip.append((2*n+1)*armonicos[0]/1000)
        if (2*n+1)*indices[0]<=len(f):
            indx_impar.append((2*n+1)*indices[0])
            indx_par.append((2*n)*indices[0])
    
    frec_multip_c = []
    indx_impar_c = []
    indx_par_c=[]
    for n in range(int(frec_limite//int(armonicos_c[0]))):
        frec_multip_c.append((2*n+1)*armonicos_c[0]/1000)
        if (2*n+1)*indices_c[0]<=len(f_c):
            indx_impar_c.append((2*n+1)*indices_c[0])
            indx_par_c.append((2*n)*indices_c[0])

    f_impar= f[indx_impar] #para grafico 1.0
    amp_impar= g[indx_impar]
    fases_impar= fase[indx_impar]
    del indx_par[0]
    f_par= f[indx_par] 
    amp_par= g[indx_par]
    fases_par= fase[indx_par]

    f_impar_c= f_c[indx_impar_c] #para grafico 1.1
    amp_impar_c= g_c[indx_impar_c]
    fases_impar_c= fase_c[indx_impar_c]
    del indx_par_c[0]
    f_par_c= f_c[indx_par_c] 
    amp_par_c= g_c[indx_par_c]
    fases_par_c= fase_c[indx_par_c]

#Reconstruyo señal impar con ifft p/ muestra
    h_aux_impar = np.zeros(len(f),dtype=np.cdouble)
    for W in indx_impar:
        h_aux_impar[W]=g_aux[W]
    rec_impares = irfft(h_aux_impar,n=len(t),norm='forward')
#Reconstruyo señal par con ifft
    h_aux_par = np.zeros(len(f),dtype=np.cdouble)
    for Z in indx_par:
        h_aux_par[Z]=g_aux[Z] 
    rec_pares = irfft(h_aux_par,n=len(t),norm='forward')

#Idem Calibracion
    h_c_aux_impar = np.zeros(len(f_c),dtype=np.cdouble)
    for W in indx_impar_c:
        h_c_aux_impar[W]=g_c_aux[W]
    rec_impares_c = irfft(h_c_aux_impar,n=len(t_c),norm='forward')
#Reconstruyo señal par con ifft
    h_c_aux_par = np.zeros(len(f_c),dtype=np.cdouble)
    for Z in indx_par_c:
        h_c_aux_par[Z]=g_c_aux[Z] 
    rec_pares_c = irfft(h_c_aux_par,n=len(t_c),norm='forward')

#Reconstruyo señal limitada con ifft
    #g_aux = np.resize(g_aux,len(f))
    #rec_limitada = irfft(g_aux,n=len(t),norm='forward')

#Reconstruyo señal de alta frecuencia
    rec_HF = irfft(g_HF,n=len(t),norm='forward')
    rec_c_HF = irfft(g_c_HF,n=len(t_c),norm='forward')
#Resto HF a la señal original y comparo con reconstruida impar
    resta = y - rec_HF
#Veo que tanto se parecen
    #r_2 = r2_score(rec_impares,rec_limitada)
    #r_2_resta  = r2_score(rec_impares,resta)
    
#Grafico 1.0 (Muestra): 
    fig = plt.figure(figsize=(8,12),constrained_layout=True)
    plt.suptitle('Análisis Espectral Muestra',fontsize=20)
#Señal Orig + Ref
    ax1 = fig.add_subplot(3,1,1)
    ax1.plot(t,y/max(y),'.-',lw=0.9,label='Muestra')
    ax1.plot(t_r,y_r/max(y_r),'.-',c='tab:red',lw=0.9,label='Referencia')
    ax1.set_xlabel('t (s)')
    ax1.set_xlim(0,2/armonicos[0])
    ax1.axvspan(0,1/armonicos[0],color='g',alpha=0.3)
    ax1.set_title('Muestra y referencia - '+str(name), loc='left', fontsize=13)
    ax1.legend(loc='best')
    ax1.grid()  
#Espectro de Frecuencias 
    ax2 = fig.add_subplot(3,1,2)
    ax2.plot(f/1000,g,'.-',lw=0.9)
    #ax2.scatter(armonicos/1000,amplitudes,c='r',marker='+',label='armónicos')
    for item in frec_multip:
        ax2.axvline(item,0,1,color='r',alpha=0.4,lw=0.9)   
    ax2.scatter(f_impar/1000,amp_impar,marker='x',c='tab:orange',label='armónicos impares',zorder=2.5)
    ax2.scatter(f_par/1000,amp_par,marker='+',c='tab:green',label='armónicos pares',zorder=2.5)
    ax2.set_title('Espectro de frecuencias - {}% - frec max: {:.0f} kHz'.format(filtro*100,frec_limite/1e3), loc='left', fontsize=13)
    ax2.set_xlabel('Frecuencia (kHz)')
    ax2.set_ylabel('|F{$\epsilon$}|')   
    ax2.set_xlim(0,max(f)/1000)
    ax2.legend(loc='best')
#  Espectro de Fases 
    ax3 = fig.add_subplot(3,1,3)
    ax3.vlines(armonicos/1000,ymin=0,ymax=fases)
    ax3.stem(armonicos/1000,fases,basefmt=' ')
    ax3.scatter(f_impar/1000,fases_impar,marker='x',color='tab:orange',label='armónicos impares',zorder=2.5)
    ax3.scatter(f_par/1000,fases_par,marker='+',color='tab:green',label='armónicos pares',zorder=2.5)    
    #ax3.vlines(f_impar/1000, ymin=0, ymax=fases_impar,color='tab:orange')
    ax3.set_ylim(-np.pi-0.5,np.pi+0.5)
    ax3.set_yticks([-np.pi,-3*np.pi/4,-np.pi/2,-np.pi/4,0,np.pi/4,np.pi/2,3*np.pi/4,np.pi])
    ax3.set_yticklabels(['-$\pi$','','$-\pi/2$','','0','','$\pi/2$','','$\pi$'])
    ax3.axhline(0,0,max(armonicos)/1000,c='k',lw=0.8)
    for item in frec_multip:
        ax3.axvline(item,.1,0.92,color='r',alpha=0.4,lw=0.9)  
    #ax3.scatter(armonicos/1000,theta,label='fases redef')    
    ax3.set_ylabel('Fase')
    ax3.set_xlabel('Frecuencia (kHz)')
    #ax3.legend(loc='best')
    ax3.set_title('Espectro de fases',loc='left', fontsize=13)
    ax3.set_xlim(0,max(f)/1000)
    ax3.grid(axis='y')
    ax3.legend()

#Grafico 1.1 (Calibracion): 
    fig4 = plt.figure(figsize=(8,12),constrained_layout=True)
    plt.suptitle('Análisis Espectral Calibracion',fontsize=20)
#Señal Orig + Ref
    ax1 = fig4.add_subplot(3,1,1)
    ax1.plot(t_c,y_c/max(y_c),'.-',lw=0.9,label='Calibracion')
    ax1.plot(t_r_c,y_r_c/max(y_r_c),'.-',c='tab:red',lw=0.9,label='Referencia')
    ax1.set_xlabel('t (s)')
    ax1.set_xlim(0,2/armonicos_c[0])
    ax1.axvspan(0,1/armonicos_c[0],color='g',alpha=0.3)
    ax1.set_title('Calibracion y referencia - '+str(name)+'_cal', loc='left', fontsize=13)
    ax1.legend(loc='best')
    ax1.grid()  
#Espectro de Frecuencias 
    ax2 = fig4.add_subplot(3,1,2)
    ax2.plot(f_c/1000,g_c,'.-',lw=0.9)
    #ax2.scatter(armonicos/1000,amplitudes,c='r',marker='+',label='armónicos')
    for item in frec_multip:
        ax2.axvline(item,0,1,color='r',alpha=0.4,lw=0.9)   
    ax2.scatter(f_impar_c/1000,amp_impar_c,marker='x',c='tab:orange',label='armónicos impares',zorder=2.5)
    ax2.scatter(f_par_c/1000,amp_par_c,marker='+',c='tab:green',label='armónicos pares',zorder=2.5)
    ax2.set_title('Espectro de frecuencias - {}% - frec max: {:.0f} kHz'.format(filtro*100,frec_limite/1e3), loc='left', fontsize=13)
    ax2.set_xlabel('Frecuencia (kHz)')
    ax2.set_ylabel('|F{$\epsilon$}|')   
    ax2.set_xlim(0,max(f_c)/1000)
    ax2.legend(loc='best')
#  Espectro de Fases 
    ax3 = fig4.add_subplot(3,1,3)
    ax3.vlines(armonicos_c/1000,ymin=0,ymax=fases_c)
    ax3.stem(armonicos_c/1000,fases_c,basefmt=' ')
    ax3.scatter(f_impar_c/1000,fases_impar_c,marker='x',color='tab:orange',label='armónicos impares',zorder=2.5)
    ax3.scatter(f_par_c/1000,fases_par_c,marker='+',color='tab:green',label='armónicos pares',zorder=2.5)    
    #ax3.vlines(f_impar/1000, ymin=0, ymax=fases_impar,color='tab:orange')
    ax3.set_ylim(-np.pi-0.5,np.pi+0.5)
    ax3.set_yticks([-np.pi,-3*np.pi/4,-np.pi/2,-np.pi/4,0,np.pi/4,np.pi/2,3*np.pi/4,np.pi])
    ax3.set_yticklabels(['-$\pi$','','$-\pi/2$','','0','','$\pi/2$','','$\pi$'])
    ax3.axhline(0,0,max(armonicos)/1000,c='k',lw=0.8)
    for item in frec_multip_c:
        ax3.axvline(item,.1,0.92,color='r',alpha=0.4,lw=0.9)  
    #ax3.scatter(armonicos/1000,theta,label='fases redef')    
    ax3.set_ylabel('Fase')
    ax3.set_xlabel('Frecuencia (kHz)')
    #ax3.legend(loc='best')
    ax3.set_title('Espectro de fases',loc='left', fontsize=13)
    ax3.set_xlim(0,max(f)/1000)
    ax3.grid(axis='y')
    ax3.legend()

#Redefino angulos p/ Fasorial impar
    r0 = 1
    theta_0 = 0
    r = amp_impar/max(amp_impar)
    defasaje_m =  fases_r-fases_impar
    r_c = amp_impar_c/max(amp_impar_c)
    defasaje_c = fases_r_c - fases_impar_c

#Grafico 2.0: Espectro Impar, Fasorial, Original+Rec_impar (Muestra)
    fig2 = plt.figure(figsize=(8,12),constrained_layout=True)
    plt.suptitle('Reconstruccion impar',fontsize=20)
# Señal Original + Reconstruida impar
    ax1=fig2.add_subplot(3,1,1)
    ax1.plot(t,y,'.-',lw=0.9,label='Señal original')
    #ax1.plot(t,reconstruida*max(rec2),'r-',lw=0.9,label='Reconstruida ({} armónicos)'.format(len(armonicos)))
    #ax1.plot(t,rec_limitada,'-',lw=0.9,label='Filtrada ({:.0f} kHz)'.format(frec_limite/1e3))
    ax1.plot(t,rec_impares,'-',lw=1.3,label='Componentes impares')
    ax1.plot(t,rec_pares,'-',lw=1.1,label='Componentes pares')
    ax1.set_xlabel('t (s)')
    ax1.set_xlim(0,2/armonicos[0])
    ax1.axvspan(0,1/armonicos[0],color='g',alpha=0.3)
    ax1.set_title(str(name))
    # + ' (R$^2$: {:.3f})'.format(r_2), loc='left', fontsize=13)     
    ax1.grid() 
    ax1.legend(loc='best')
# Espectro en fases impares
    ax2=fig2.add_subplot(3,1,2)
    #ax2.scatter(armonicos/1000,amplitudes,c='r',marker='+',label='armónicos')
    ax2.scatter(f_impar/1000,amp_impar,marker='o',c='tab:blue',label='Armónicos impares',zorder=2.5)
    ax2.vlines(f_impar/1000, ymin=0, ymax=amp_impar)
    ax2.axvline(armonicos_r/1000, ymin=0, ymax=1,c='tab:red',label='Referencia',lw=1,alpha=0.8)
    #ax2.stem(armonicos_r/1000,(amplitudes_r/max(amplitudes_r))*max(amp_impar),basefmt=' ',markerfmt='or',bottom=0.0,label='R')
    #for item in f_impar:
        #ax2.axvline(item,0,1,c='tab:orange',lw=0.9)   
    #ax2.scatter(f_par/1000,amp_par,marker='+',c='tab:green',label='armónicos pares',zorder=2.5)
    ax2.set_title('Espectro de la señal reconstruida', loc='left', fontsize=13)
    ax2.set_xlabel('Frecuencia (kHz)')
    ax2.set_ylabel('|F{$\epsilon$}|')   
    ax2.set_xlim(0,max(f)/1000)
    ax2.set_ylim(0,max(amp_impar)*1.1)
    ax2.grid()
# inset
    axin = ax2.inset_axes([0.4, 0.35, 0.57, 0.6])
    axin.scatter(f_impar/1000,fases_impar,label='Fases')
    axin.vlines(f_impar/1000, ymin=0, ymax=fases_impar)
    axin.scatter(armonicos_r/1000,fases_r,label='Fases ref',c='tab:red')
    axin.vlines(armonicos_r/1000, ymin=0, ymax=fases_r,color='tab:red')
    axin.set_xlabel('Frecuencia (kHz)', fontsize=8)
    axin.set_ylim(-np.pi-0.5,np.pi+0.5)
    axin.set_yticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
    axin.set_yticklabels(['-$\pi$','$-\pi/2$','0','$\pi/2$','$\pi$'])
    axin.axhline(0,0,max(f_impar)/1000,c='k',lw=0.8)
    #axin.set_ylabel('fase')
    #axin.legend(loc='best')
    axin.grid()
    axin.set_title(' Espectro de fases',loc='left', y=0.87, fontsize=10)
    axin.set_xlim(0,max(f_impar)/1000)
# Fasorial impares
    ax3=fig2.add_subplot(3,1,3,polar=True)
    ax3.scatter(theta_0,r0,label='Referencia',marker='D',c='tab:red')
    ax3.plot([0,theta_0], [0,1],'-',c='tab:red')
    #ax3.plot(defasaje_m,r,'-',c='tab:blue',lw=0.7)
    ax3.scatter(defasaje_m, r, label = 'Muestra',c='tab:blue')
    for i in range(len(defasaje_m)):
        ax3.plot([0, defasaje_m[i]], [0, r[i]],'-o',c='tab:blue')
    ax3.spines['polar'].set_visible(False)  # Show or hide the plot spine
    ax3.set_rmax(1.1)
    ax3.set_rticks([0.25,0.5,0.75, 1])  # Less radial ticks
    ##ax3.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax3.grid(True)
    ax3.set_theta_zero_location('N')
    ax3.set_theta_direction(-1)
    ax3.legend(loc='upper left',bbox_to_anchor=(1.02,0.5,0.4,0.5))
    ax3.set_title('Retraso respecto al campo', loc='center', fontsize=13,va='bottom')

#Grafico 2.1: Espectro Impar, Fasorial, Original+Rec_impar (Calibracion)
    fig5 = plt.figure(figsize=(8,12),constrained_layout=True)
    plt.suptitle('Reconstruccion impar (calibracion)',fontsize=20)
# Señal Original + Reconstruida impar
    ax1=fig5.add_subplot(3,1,1)
    ax1.plot(t_c,y_c,'.-',lw=0.9,label='Señal original')
    ax1.plot(t_c,rec_impares_c,'-',lw=1.3,label='Componentes impares')
    ax1.plot(t_c,rec_pares_c,'-',lw=1.1,label='Componentes pares')
    ax1.set_xlabel('t (s)')
    ax1.set_xlim(0,2/armonicos_c[0])
    ax1.axvspan(0,1/armonicos_c[0],color='g',alpha=0.3)
    ax1.set_title(str(name)+'_cal')
    # + ' (R$^2$: {:.3f})'.format(r_2), loc='left', fontsize=13)     
    ax1.grid() 
    ax1.legend(loc='best')
# Espectro en fases impares
    ax2=fig5.add_subplot(3,1,2)
    #ax2.scatter(armonicos/1000,amplitudes,c='r',marker='+',label='armónicos')
    ax2.scatter(f_impar_c/1000,amp_impar_c,marker='o',c='tab:blue',label='Armónicos impares',zorder=2.5)
    ax2.vlines(f_impar_c/1000, ymin=0, ymax=amp_impar_c)
    ax2.axvline(armonicos_r/1000, ymin=0, ymax=1,c='tab:red',label='Referencia',lw=1,alpha=0.8)
    #ax2.stem(armonicos_r/1000,(amplitudes_r/max(amplitudes_r))*max(amp_impar),basefmt=' ',markerfmt='or',bottom=0.0,label='R')
    
    #for item in f_impar:
        #ax2.axvline(item,0,1,c='tab:orange',lw=0.9)   
    #ax2.scatter(f_par/1000,amp_par,marker='+',c='tab:green',label='armónicos pares',zorder=2.5)
    ax2.set_title('Espectro de la señal de calibracion reconstruida', loc='left', fontsize=13)
    ax2.set_xlabel('Frecuencia (kHz)')
    ax2.set_ylabel('|F{$\epsilon$}|')   
    ax2.set_xlim(0,max(f)/1000)
    ax2.set_ylim(0,max(amp_impar)*1.1)
    ax2.grid()
# inset
    axin = ax2.inset_axes([0.4, 0.35, 0.57, 0.6])
    axin.scatter(f_impar_c/1000,fases_impar_c,label='Fases')
    axin.vlines(f_impar_c/1000, ymin=0, ymax=fases_impar_c)
    axin.scatter(armonicos_r/1000,fases_r,label='Fases ref',c='tab:red')
    axin.vlines(armonicos_r/1000, ymin=0, ymax=fases_r,color='tab:red')
    axin.set_xlabel('Frecuencia (kHz)', fontsize=8)
    axin.set_ylim(-np.pi-0.5,np.pi+0.5)
    axin.set_yticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
    axin.set_yticklabels(['-$\pi$','$-\pi/2$','0','$\pi/2$','$\pi$'])
    axin.axhline(0,0,max(f_impar)/1000,c='k',lw=0.8)
    #axin.set_ylabel('fase')
    #axin.legend(loc='best')
    axin.grid()
    axin.set_title(' Espectro de fases',loc='left', y=0.87, fontsize=10)
    axin.set_xlim(0,max(f_impar)/1000)
# Fasorial impares
    ax3=fig5.add_subplot(3,1,3,polar=True)
    ax3.scatter(theta_0,r0,label='Referencia',marker='D',c='tab:red')
    ax3.plot([0,theta_0], [0,1],'-',c='tab:red')
    #ax3.plot(defasaje_m,r,'-',c='tab:blue',lw=0.7)
    ax3.scatter(defasaje_c, r_c, label = 'Muestra',c='tab:blue')
    for i in range(len(defasaje_c)):
        ax3.plot([0, defasaje_c[i]], [0, r_c[i]],'-o',c='tab:blue')
    ax3.spines['polar'].set_visible(False)  # Show or hide the plot spine
    ax3.set_rmax(1.1)
    ax3.set_rticks([0.25,0.5,0.75, 1])  # Less radial ticks
    ##ax3.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax3.grid(True)
    ax3.set_theta_zero_location('N')
    ax3.set_theta_direction(-1)
    ax3.legend(loc='upper left',bbox_to_anchor=(1.02,0.5,0.4,0.5))
    ax3.set_title('Retraso respecto al campo', loc='center', fontsize=13,va='bottom')

#Grafico 3: Altas frecuencias (muestra)
    fig3 = plt.figure(figsize=(8,12),constrained_layout=True)
    plt.suptitle('Altas frecuencias',fontsize=20)
# Espectro en fases impares
    ax1=fig3.add_subplot(2,1,1)
    ax1.stem(f_impar/1000,amp_impar,linefmt='C0-',basefmt=' ',markerfmt='.',bottom=0.0,label='armónicos impares')
    #ax1.stem(f_par/1000,amp_par,linefmt='C2-',basefmt=' ',markerfmt='.g',bottom=0.0,label='armónicos pares')
    ax1.plot(f_HF[f_HF>frec_limite]/1000,g_HF[f_HF>frec_limite],'.-',lw=0.9,c='tab:orange',label='Alta frecuencia')
    #ax1.scatter(armonicos/1000,amplitudes,c='r',marker='+',label='armónicos')
    #ax1.scatter(f_impar/1000,amp_impar,marker='o',c='tab:blue',label='Armónicos impares',zorder=2.5)
    #ax1.vlines(f_impar/1000, ymin=0, ymax=amp_impar)
    #ax1.axvline(armonicos_r/1000, ymin=0, ymax=1,c='tab:red',label='Referencia',lw=1,alpha=0.8)
    #ax1.stem(armonicos_r/1000,(amplitudes_r/max(amplitudes_r))*max(amp_impar),basefmt=' ',markerfmt='or',bottom=0.0,label='R')
    ax1.legend(loc='best')
    ax1.set_title('Espectro de la señal', loc='left', fontsize=13)
    ax1.set_xlabel('Frecuencia (kHz)')
    ax1.set_ylabel('|F{$\epsilon$}|')   
    #ax1.set_xlim(0,max(f)/1000)
    #ax1.set_ylim(0,max(amp_impar)*1.1)
    ax1.grid()

# Señal HF + Reconstruida LF
    ax2=fig3.add_subplot(2,1,2)
    ax2.plot(t,rec_impares,'-',lw=1,label=f'Componentes impares (f<{frec_limite/1e6:.0f} MHz)',c='tab:blue',zorder=3)
    ax2.plot(t,rec_pares,'-',lw=1,label=f'Componentes pares (f<{frec_limite/1e6:.0f} MHz)',c='tab:green',zorder=3)  
    ax2.plot(t,rec_HF,'-',lw=0.9,label=f'Altas frecuencias ($f>${(frec_limite/1e6):.0f} MHz)',c='tab:orange',zorder=2)
    ax2.plot(t,y,'-',lw=1.2,label='Señal original',c='tab:red',zorder=1,alpha=0.8)    
    #ax2.plot(t,resta,'-',lw=0.9,label='Resta')  
    ax2.set_xlabel('t (s)')
    ax2.set_xlim(0,5/armonicos[0])
    ax2.set_ylim(1.1*min(y),1.1*max(y))
    ax2.axvspan(0,1/armonicos[0],color='g',alpha=0.3)
    ax2.set_title(str(name))
    # + ' (R$^2$: {:.3f})'.format(r_2), loc='left', fontsize=13)     
    ax2.grid() 
    ax2.legend(loc='best')

#Grafico 3.1: Altas frecuencias (calibracion)
    fig6 = plt.figure(figsize=(8,12),constrained_layout=True)
    plt.suptitle('Altas frecuencias',fontsize=20)
# Espectro en fases impares
    ax1=fig6.add_subplot(2,1,1)
    ax1.stem(f_impar_c/1000,amp_impar_c,linefmt='C0-',basefmt=' ',markerfmt='.',bottom=0.0,label='armónicos impares')
    #ax1.stem(f_par_c/1000,amp_par_c,linefmt='C2-',basefmt=' ',markerfmt='.g',bottom=0.0,label='armónicos pares')
    ax1.plot(f_c_HF[f_c_HF>frec_limite]/1000,g_c_HF[f_c_HF>frec_limite],'.-',lw=0.9,c='tab:orange',label='Alta frecuencia')
    #ax1.scatter(armonicos/1000,amplitudes,c='r',marker='+',label='armónicos')
    #ax1.scatter(f_impar/1000,amp_impar,marker='o',c='tab:blue',label='Armónicos impares',zorder=2.5)
    #ax1.vlines(f_impar/1000, ymin=0, ymax=amp_impar)
    #ax1.axvline(armonicos_r/1000, ymin=0, ymax=1,c='tab:red',label='Referencia',lw=1,alpha=0.8)
    #ax1.stem(armonicos_r/1000,(amplitudes_r/max(amplitudes_r))*max(amp_impar),basefmt=' ',markerfmt='or',bottom=0.0,label='R')
    ax1.legend(loc='best')
    ax1.set_title('Espectro de la señal', loc='left', fontsize=13)
    ax1.set_xlabel('Frecuencia (kHz)')
    ax1.set_ylabel('|F{$\epsilon$}|')   
    #ax1.set_xlim(0,max(f)/1000)
    #ax1.set_ylim(0,max(amp_impar)*1.1)
    ax1.grid()

# Señal HF + Reconstruida LF
    ax2=fig6.add_subplot(2,1,2)
    ax2.plot(t_c,rec_impares_c,'-',lw=1,label=f'Componentes impares (f<{frec_limite/1e6:.0f} MHz)',c='tab:blue',zorder=3)
    ax2.plot(t_c,rec_pares_c,'-',lw=1,label=f'Componentes pares (f<{frec_limite/1e6:.0f} MHz)',c='tab:green',zorder=3)  
    ax2.plot(t_c,rec_c_HF,'-',lw=0.9,label=f'Altas frecuencias ($f>${(frec_limite/1e6):.0f} MHz)',c='tab:orange',zorder=2)
    ax2.plot(t_c,y_c,'-',lw=1.2,label='Señal original',c='tab:red',zorder=1,alpha=0.8)    
    #ax2.plot(t,resta,'-',lw=0.9,label='Resta')  
    ax2.set_xlabel('t (s)')
    ax2.set_xlim(0,5/armonicos_c[0])
    ax2.set_ylim(1.1*min(y_c),1.1*max(y_c))
    ax2.axvspan(0,1/armonicos_c[0],color='g',alpha=0.3)
    ax2.set_title(str(name)+'_cal')
    # + ' (R$^2$: {:.3f})'.format(r_2), loc='left', fontsize=13)     
    ax2.grid() 
    ax2.legend(loc='best')

    return armonicos, armonicos_r, amplitudes, amplitudes_r, fases , fases_r , fig, fig2, indices, indx_impar, rec_impares,rec_impares_c,fig3,fig4,fig5,fig6


def fourier_señales_3(t,v,v_r_m,delta_t,polaridad,filtro,frec_limite,name, figuras=0):
    '''
    Toma señales de muestra, calibracion y referencia obtieniendo via fft frecuencias y fases.
    frec_muestreo = sample rate 1/delta_t (tipicamente 1e8 o 5e7).    
    Las señales indefectiblemente deben estar recortadas a N ciclos.
    Establecer frec limite permite filtrar la interferencia de alta señal del generador RF\n
    Se conoce la polaridad de la señal(del ajuste lineal sobre el ciclo paramagnetico). 
    
    Adaptada para usarse en powerslave.py
    n muestras
    1 fondo
    
    '''
    t = t - t[0] #Muestra 
    t_r = t.copy() #y su referencia
    y = polaridad*v     #muestra (magnetizacion)
    y_r = v_r_m        #referencia muestra (campo)
    N = len(v)
    N_r_m = len(v_r_m)

    #Para que el largo de los vectores coincida
    if len(t)<len(y): #alargo t
        t = np.pad(t,(0,delta_t*(len(y)-len(t))),mode='linear_ramp',end_values=(0,max(t)+delta_t*(len(y)-len(t))))
    elif len(t)>len(y):#recorto t    
        t=np.resize(t,len(y))

    #Idem referencias
    if len(t_r)<len(y_r): #alargo t
        t_r = np.pad(t_r,(0,delta_t*(len(y_r)-len(t_r))),mode='linear_ramp',end_values=(0,max(t_r)+delta_t*(len(y_r)-len(t_r))))
    elif len(t_r)>len(y_r):#recorto t    
        t_r=np.resize(t_r,len(y_r))

    #Aplico transformada de Fourier
    f = rfftfreq(N,d=delta_t) #obtengo la mitad de los puntos, porque uso rfft
    f_HF = f.copy() 
    #f_HF = f_HF[np.nonzero(f>=frec_limite)] #aca estan el resto 
    f = f[np.nonzero(f<=frec_limite)] #limito frecuencias 
    g_aux = fft(y,norm='forward') 
    #“forward” applies the 1/n factor on the forward tranform
    g = abs(g_aux)  #magnitud    
    fase = np.angle(g_aux)
    
    #Idem p/ Referencia
    f_r = rfftfreq(N_r_m, d=delta_t)
    f_r = f_r[np.nonzero(f_r<=frec_limite)]
    g_r_aux = fft(y_r,norm='forward')
    g_r = abs(g_r_aux)
    fase_r = np.angle(g_r_aux)
     
    #Recorto vectores hasta frec_limite
    g_HF = g.copy() 
    #g_HF = g_HF[np.nonzero(f>=frec_limite)]#magnitud de HF
    g = np.resize(g,len(f))
    g_r = np.resize(g_r,len(f_r))
    g_HF = np.resize(g_HF,len(f_HF))
    g_HF[np.argwhere(f_HF<=frec_limite)]=0 #Anulo LF

    #Obtengo frecuencias cuya intensidad relativa supera umbral dado por el filtro
    indices,_=find_peaks(abs(g),threshold=max(g)*filtro)
    # anulo armonico fundamental descomentando siguiente linea
    #indices = np.delete(indices,0)
    indices_r,_=find_peaks(abs(g_r),threshold=max(g_r)*filtro)

    for elem in indices:
        if f[elem]<0.9*f_r[indices_r[0]]:
            print('ATENCION: detectada subfrecuencia anómala en el espectro de la señal de muestra {:.2f} Hz\n'.format(f[elem]))
            indices = np.delete(indices,0)
            
    armonicos = f[indices]
    amplitudes = g[indices]
    fases = fase[indices]

    armonicos_r = f_r[indices_r]
    amplitudes_r = g_r[indices_r]
    fases_r = fase_r[indices_r]

    #Imprimo tabla 
    print('''Espectro de la señal de referencia:\nFrecuencia (Hz) - Intensidad rel - Fase (rad)''')
    for i in range(len(indices_r)):
        print(f'{armonicos_r[i]:<10.2f}    {amplitudes_r[i]/max(amplitudes_r):>12.2f}    {fases_r[i]:>12.4f}')
    

    print('''\nEspectro de la señal de muestra:\nFrecuencia (Hz) - Intensidad rel - Fase (rad)''')
    for i in range(len(indices)):
        print(f'{armonicos[i]:<10.2f}    {amplitudes[i]/max(amplitudes):>12.2f}    {fases[i]:>12.4f}')
  
    #Frecuencias/indices multiplo impar/par de la fundamental
    frec_multip = []
    indx_impar = []
    indx_par=[]

    for n in range(int(frec_limite//int(armonicos[0]))):
        frec_multip.append((2*n+1)*armonicos[0]/1000)
        if (2*n+1)*indices[0]<=len(f):
            indx_impar.append((2*n+1)*indices[0])
            indx_par.append((2*n)*indices[0])

    f_impar= f[indx_impar] #para grafico 1.0
    amp_impar= g[indx_impar]
    fases_impar= fase[indx_impar]
    del indx_par[0]
    f_par= f[indx_par] 
    amp_par= g[indx_par]
    fases_par= fase[indx_par]

    #Reconstruyo señal impar con ifft p/ muestra
    h_aux_impar = np.zeros(len(f),dtype=np.cdouble)
    for W in indx_impar:
        h_aux_impar[W]=g_aux[W]
    rec_impares = irfft(h_aux_impar,n=len(t),norm='forward')
    #Reconstruyo señal par con ifft
    h_aux_par = np.zeros(len(f),dtype=np.cdouble)
    for Z in indx_par:
        h_aux_par[Z]=g_aux[Z] 
    rec_pares = irfft(h_aux_par,n=len(t),norm='forward')

    #Reconstruyo señal limitada con ifft
    #g_aux = np.resize(g_aux,len(f))
    #rec_limitada = irfft(g_aux,n=len(t),norm='forward')

    #Reconstruyo señal de alta frecuencia
    rec_HF = irfft(g_HF,n=len(t),norm='forward')
    #Resto HF a la señal original y comparo con reconstruida impar
    resta = y - rec_HF

    #Grafico 1.0 (Muestra): 
    fig = plt.figure(figsize=(8,12),constrained_layout=True)
    plt.suptitle('Análisis Espectral Muestra',fontsize=20)

    #Señal Orig + Ref
    ax1 = fig.add_subplot(3,1,1)
    ax1.plot(t,y/max(y),'.-',lw=0.9,label='Muestra')
    ax1.plot(t_r,y_r/max(y_r),'.-',c='tab:red',lw=0.9,label='Referencia')
    ax1.set_xlabel('t (s)')
    ax1.set_xlim(0,2/armonicos[0])
    ax1.axvspan(0,1/armonicos[0],color='g',alpha=0.3)
    ax1.set_title('Muestra y referencia - '+str(name), loc='left', fontsize=13)
    ax1.legend(loc='best')
    ax1.grid()  

    #Espectro de Frecuencias 
    ax2 = fig.add_subplot(3,1,2)
    ax2.plot(f/1000,g,'.-',lw=0.9)
    #ax2.scatter(armonicos/1000,amplitudes,c='r',marker='+',label='armónicos')
    for item in frec_multip:
        ax2.axvline(item,0,1,color='r',alpha=0.4,lw=0.9)   
    ax2.scatter(f_impar/1000,amp_impar,marker='x',c='tab:orange',label='armónicos impares',zorder=2.5)
    #ax2.scatter(f_par/1000,amp_par,marker='+',c='tab:green',label='armónicos pares',zorder=2.5)
    ax2.set_title('Espectro de frecuencias - {}% - frec max: {:.0f} kHz'.format(filtro*100,frec_limite/1e3), loc='left', fontsize=13)
    ax2.set_xlabel('Frecuencia (kHz)')
    ax2.set_ylabel('|F{$\epsilon$}|')   
    ax2.set_xlim(0,max(f)/1000)
    ax2.legend(loc='best')

    #  Espectro de Fases 
    ax3 = fig.add_subplot(3,1,3)
    ax3.vlines(armonicos/1000,ymin=0,ymax=fases)
    ax3.stem(armonicos/1000,fases,basefmt=' ')
    ax3.scatter(f_impar/1000,fases_impar,marker='x',color='tab:orange',label='armónicos impares',zorder=2.5)
    #ax3.scatter(f_par/1000,fases_par,marker='+',color='tab:green',label='armónicos pares',zorder=2.5)    
    #ax3.vlines(f_impar/1000, ymin=0, ymax=fases_impar,color='tab:orange')
    ax3.set_ylim(-np.pi-0.5,np.pi+0.5)
    ax3.set_yticks([-np.pi,-3*np.pi/4,-np.pi/2,-np.pi/4,0,np.pi/4,np.pi/2,3*np.pi/4,np.pi])
    ax3.set_yticklabels(['-$\pi$','','$-\pi/2$','','0','','$\pi/2$','','$\pi$'])
    ax3.axhline(0,0,max(armonicos)/1000,c='k',lw=0.8)
    for item in frec_multip:
        ax3.axvline(item,.1,0.92,color='r',alpha=0.4,lw=0.9)  
    #ax3.scatter(armonicos/1000,theta,label='fases redef')    
    ax3.set_ylabel('Fase')
    ax3.set_xlabel('Frecuencia (kHz)')
    #ax3.legend(loc='best')
    ax3.set_title('Espectro de fases',loc='left', fontsize=13)
    ax3.set_xlim(0,max(f)/1000)
    ax3.grid(axis='y')
    ax3.legend()

    
    #Redefino angulos p/ Fasorial impar
    r0 = 1
    theta_0 = 0
    r = amp_impar/max(amp_impar)
    defasaje_m =  fases_r-fases_impar
    
    #Grafico 2.0: Espectro Impar, Fasorial, Original+Rec_impar (Muestra)
    fig2 = plt.figure(figsize=(8,12),constrained_layout=True)
    plt.suptitle('Reconstruccion impar',fontsize=20)
    
    # Señal Original + Reconstruida impar
    ax1=fig2.add_subplot(2,1,1)
    ax1.plot(t,y,'.-',lw=0.9,label='Señal original')
    ax1.plot(t,rec_impares,'-',lw=1.3,label='Componentes impares')
    ax1.plot(t,rec_pares,'-',lw=1.1,label='Componentes pares')
    ax1.set_xlabel('t (s)')
    ax1.set_xlim(0,3/armonicos[0])
    ax1.axvspan(0,1/armonicos[0],color='g',alpha=0.3)
    ax1.set_title(str(name))
    ax1.grid() 
    ax1.legend(loc='best')
    ax1.set_ylim(-max(rec_impares)*10,max(rec_impares)*10)
    
    # Espectro en fases impares
    ax2=fig2.add_subplot(2,1,2)
    ax2.scatter(f_impar/1000,amp_impar,marker='o',c='tab:blue',label='Armónicos impares',zorder=2.5)
    ax2.vlines(f_impar/1000, ymin=0, ymax=amp_impar)
    ax2.axvline(armonicos_r/1000, ymin=0, ymax=1,c='tab:red',label='Referencia',lw=1,alpha=0.8)
    ax2.set_title('Espectro de la señal reconstruida', loc='left', fontsize=13)
    ax2.set_xlabel('Frecuencia (kHz)')
    ax2.set_ylabel('|F{$\epsilon$}|')   
    ax2.set_xlim(0,max(f)/1000)
    ax2.set_ylim(0,max(amp_impar)*1.1)
    ax2.grid()
    
    #inset
    axin = ax2.inset_axes([0.4, 0.35, 0.57, 0.6])
    axin.scatter(f_impar/1000,fases_impar,label='Fases')
    axin.vlines(f_impar/1000, ymin=0, ymax=fases_impar)
    axin.scatter(armonicos_r/1000,fases_r,label='Fases ref',c='tab:red')
    axin.vlines(armonicos_r/1000, ymin=0, ymax=fases_r,color='tab:red')
    axin.set_xlabel('Frecuencia (kHz)', fontsize=8)
    axin.set_ylim(-np.pi-0.5,np.pi+0.5)
    axin.set_yticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
    axin.set_yticklabels(['-$\pi$','$-\pi/2$','0','$\pi/2$','$\pi$'])
    axin.axhline(0,0,max(f_impar)/1000,c='k',lw=0.8)
    #axin.set_ylabel('fase')
    #axin.legend(loc='best')
    axin.grid()
    axin.set_title(' Espectro de fases',loc='left', y=0.87, fontsize=10)
    axin.set_xlim(0,max(f_impar)/1000)
    
    # Fasorial impares
    # ax3=fig2.add_subplot(3,1,3,polar=True)
    # ax3.scatter(theta_0,r0,label='Referencia',marker='D',c='tab:red')
    # ax3.plot([0,theta_0], [0,1],'-',c='tab:red')
    # #ax3.plot(defasaje_m,r,'-',c='tab:blue',lw=0.7)
    # ax3.scatter(defasaje_m, r, label = 'Muestra',c='tab:blue')
    # for i in range(len(defasaje_m)):
    #     ax3.plot([0, defasaje_m[i]], [0, r[i]],'-o',c='tab:blue')
    # ax3.spines['polar'].set_visible(False)  # Show or hide the plot spine
    # ax3.set_rmax(1.1)
    # ax3.set_rticks([0.25,0.5,0.75, 1])  # Less radial ticks
    # ##ax3.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    # ax3.grid(True)
    # ax3.set_theta_zero_location('N')
    # ax3.set_theta_direction(-1)
    # ax3.legend(loc='upper left',bbox_to_anchor=(1.02,0.5,0.4,0.5))
    # ax3.set_title('Retraso respecto al campo', loc='center', fontsize=13,va='bottom')

    
    #Grafico 3: Altas frecuencias (muestra)
    # fig3 = plt.figure(figsize=(8,12),constrained_layout=True)
    # plt.suptitle('Altas frecuencias',fontsize=20)

    # # Espectro en fases impares
    # ax1=fig3.add_subplot(2,1,1)
    # ax1.stem(f_impar/1000,amp_impar,linefmt='C0-',basefmt=' ',markerfmt='.',bottom=0.0,label='armónicos impares')
    # #ax1.stem(f_par/1000,amp_par,linefmt='C2-',basefmt=' ',markerfmt='.g',bottom=0.0,label='armónicos pares')
    # ax1.plot(f_HF[f_HF>frec_limite]/1000,g_HF[f_HF>frec_limite],'.-',lw=0.9,c='tab:orange',label='Alta frecuencia')
    # ax1.legend(loc='best')
    # ax1.set_title('Espectro de la señal', loc='left', fontsize=13)
    # ax1.set_xlabel('Frecuencia (kHz)')
    # ax1.set_ylabel('|F{$\epsilon$}|')   
    # #ax1.set_xlim(0,max(f)/1000)
    # #ax1.set_ylim(0,max(amp_impar)*1.1)
    # ax1.grid()

    # # Señal HF + Reconstruida LF
    # ax2=fig3.add_subplot(2,1,2)
    # ax2.plot(t,rec_impares,'-',lw=1,label=f'Componentes impares (f<{frec_limite/1e6:.0f} MHz)',c='tab:blue',zorder=3)
    # ax2.plot(t,rec_pares,'-',lw=1,label=f'Componentes pares (f<{frec_limite/1e6:.0f} MHz)',c='tab:green',zorder=3)  
    # ax2.plot(t,rec_HF,'-',lw=0.9,label=f'Altas frecuencias ($f>${(frec_limite/1e6):.0f} MHz)',c='tab:orange',zorder=2)
    # ax2.plot(t,y,'-',lw=1.2,label='Señal original',c='tab:red',zorder=1,alpha=0.8)    
    # ax2.set_xlabel('t (s)')
    # ax2.set_xlim(0,5/armonicos[0])
    # ax2.set_ylim(1.1*min(y),1.1*max(y))
    # ax2.axvspan(0,1/armonicos[0],color='g',alpha=0.3)
    # ax2.set_title(str(name))
    # ax2.grid() 
    # ax2.legend(loc='best')

    defase_fundamental = defasaje_m[0]
    return fig, fig2,  rec_impares,defase_fundamental




def fourier_señales_4(t,v,v_r_m,delta_t,polaridad,filtro,frec_limite,name,d_phi_m,figuras=0,correccion_fase_espuria=False):
    '''
    Toma señales de muestra, calibracion y referencia obtieniendo via fft frecuencias y fases.
    frec_muestreo = sample rate 1/delta_t (tipicamente 1e8 o 5e7).    
    Las señales indefectiblemente deben estar recortadas a N ciclos.
    Establecer frec limite permite filtrar la interferencia de alta señal del generador RF\n
    Se conoce la polaridad de la señal(del ajuste lineal sobre el ciclo paramagnetico). 
    
    Adaptada para usarse en powerslave_descuento_de_fase.py para medididas con barrido en temperatura
    n muestras
    1 fondo
    
   
    '''
    t = t - t[0] #Muestra 
    t_r = t.copy() #y su referencia
    y = polaridad*v     #muestra (magnetizacion)
    y_r = v_r_m        #referencia muestra (campo)
    N = len(v)
    N_r_m = len(v_r_m)

    #Para que el largo de los vectores coincida
    if len(t)<len(y): #alargo t
        t = np.pad(t,(0,delta_t*(len(y)-len(t))),mode='linear_ramp',end_values=(0,max(t)+delta_t*(len(y)-len(t))))
    elif len(t)>len(y):#recorto t    
        t=np.resize(t,len(y))

    #Idem referencias
    if len(t_r)<len(y_r): #alargo t
        t_r = np.pad(t_r,(0,delta_t*(len(y_r)-len(t_r))),mode='linear_ramp',end_values=(0,max(t_r)+delta_t*(len(y_r)-len(t_r))))
    elif len(t_r)>len(y_r):#recorto t    
        t_r=np.resize(t_r,len(y_r))

    #Aplico transformada de Fourier
    f = rfftfreq(N,d=delta_t) #obtengo la mitad de los puntos, porque uso rfft
    f_HF = f.copy() 
    #f_HF = f_HF[np.nonzero(f>=frec_limite)] #aca estan el resto 
    
    f = f[np.nonzero(f<=frec_limite)] #limito frecuencias 
    g_aux = fft(y,norm='forward')     #“forward” applies the 1/n factor on the forward tranform
    g = abs(g_aux)                    #magnitud    
    fase = np.angle(g_aux)
    
    #Idem p/ Referencia
    f_r = rfftfreq(N_r_m, d=delta_t)
    f_r = f_r[np.nonzero(f_r<=frec_limite)]
    g_r_aux = fft(y_r,norm='forward')
    g_r = abs(g_r_aux)
    fase_r = np.angle(g_r_aux)
     
    #Recorto vectores hasta frec_limite
    g_HF = g.copy() 
    #g_HF = g_HF[np.nonzero(f>=frec_limite)]#magnitud de HF
    g = np.resize(g,len(f))
    g_r = np.resize(g_r,len(f_r))
    g_HF = np.resize(g_HF,len(f_HF))
    g_HF[np.argwhere(f_HF<=frec_limite)]=0 #Anulo LF

    #Obtengo frecuencias cuya intensidad relativa supera umbral dado por el filtro
    indices,_=find_peaks(abs(g),threshold=max(g)*filtro)
    indices_r,_=find_peaks(abs(g_r),threshold=max(g_r)*filtro)

    # anulo armonico fundamental descomentando siguiente linea
    #indices = np.delete(indices,0)

    for elem in indices:
        if f[elem]<0.9*f_r[indices_r[0]]:
            print('ATENCION: detectada subfrecuencia anómala en el espectro de la señal de muestra {:.2f} Hz\n'.format(f[elem]))
            indices = np.delete(indices,0)
            
    armonicos = f[indices]
    amplitudes = g[indices]
    fases = fase[indices]

    armonicos_r = f_r[indices_r]
    amplitudes_r = g_r[indices_r]
    fases_r = fase_r[indices_r]

    #Frecuencias/indices multiplo impar/par de la fundamental
    frec_multip = []
    indx_impar = []
    indx_par=[]

    for n in range(int(frec_limite//int(armonicos[0]))):
        frec_multip.append((2*n+1)*armonicos[0]/1000)
        if (2*n+1)*indices[0]<=len(f):
            indx_impar.append((2*n+1)*indices[0])
            indx_par.append((2*n)*indices[0])

    f_impar= f[indx_impar] #para grafico 1.0
    amp_impar= g[indx_impar]
    fases_impar= fase[indx_impar]
    del indx_par[0]
    f_par= f[indx_par] 
    amp_par= g[indx_par]
    fases_par= fase[indx_par]
    
    fases_impar_correg=[]
    
    if correccion_fase_espuria==True:
        print('Correccion de fase espuria')
        for i,e in enumerate(fases_impar):
            fases_impar_correg.append(e+np.mod(correccion_fase_por_frecuencia_kHz(f_impar[i]/1000),2*np.pi))
    else:
        print('Fase espuria sin corregir')
            
        
    #Imprimo tabla 
    print('''Espectro de la señal de Campo:\nFrecuencia (Hz) - Intensidad rel - Fase (rad)''')
    for i in range(len(indices_r)):
        print(f'{armonicos_r[i]:^10.2f}|{amplitudes_r[i]/max(amplitudes_r):^12.2f}|{fases_r[i]:^12.4f}')
    
    print('''\nEspectro de la señal de muestra:\nFrecuencia_Hz|Intensidad rel|Fase_rad''')
    for i in range(len(indices)):
        print(f'{armonicos[i]:^10.2f}|{amplitudes[i]/max(amplitudes):^12.2f}|{fases[i]:^12.4f}')    
    
    print('''\nEspectro de la señal de muestra reconstruida (impar):\nFrecuencia (Hz) - Intensidad rel - Fase (rad)  - Fase corregid (rad)''')
    for i in range(len(f_impar)):
        print(f'{f_impar[i]:^10.2f}|{amp_impar[i]/max(amp_impar):^12.2f}|{fases_impar[i]:^12.4f}|{fases_impar_correg[i]:^12.4f}')    
    
    #Reconstruyo señal IMPAR con ifft
    
    h_aux_impar = np.zeros(len(f),dtype=np.cdouble)
    h_aux_impar_bis = np.zeros(len(f),dtype=np.cdouble) #para corregir fase
    
    for W in indx_impar:
        fase_espuria = correccion_fase_por_frecuencia_kHz(f[W]/1000)
        h_aux_impar[W]=g_aux[W]
        h_aux_impar_bis[W]=g_aux[W]*np.exp(fase_espuria*1j)    #aca compenso el defasaje en el fundamental  

    rec_impares = irfft(h_aux_impar,n=len(t),norm='forward')
    rec_impares_bis = irfft(h_aux_impar_bis,n=len(t),norm='forward')
    
    
    #Reconstruyo señal PAR con ifft
    h_aux_par = np.zeros(len(f),dtype=np.cdouble)
    for Z in indx_par:
        h_aux_par[Z]=g_aux[Z] 
    rec_pares = irfft(h_aux_par,n=len(t),norm='forward')

    #Reconstruyo señal limitada con ifft
    #g_aux = np.resize(g_aux,len(f))
    #rec_limitada = irfft(g_aux,n=len(t),norm='forward')

    #Reconstruyo señal de alta frecuencia
    rec_HF = irfft(g_HF,n=len(t),norm='forward')
    #Resto HF a la señal original y comparo con reconstruida impar
    resta = y - rec_HF

    #Grafico 1.0 (Muestra): 
    fig = plt.figure(figsize=(8,12),constrained_layout=True)
    plt.suptitle('Análisis Espectral Muestra',fontsize=20)

    #Señal Orig + Ref
    ax1 = fig.add_subplot(3,1,1)
    ax1.plot(t,y/max(y),'.-',lw=0.9,label='Muestra')
    ax1.plot(t_r,y_r/max(y_r),'.-',c='tab:red',lw=0.9,label='Referencia')
    ax1.set_xlabel('t (s)')
    ax1.set_xlim(0,2/armonicos[0])
    ax1.axvspan(0,1/armonicos[0],color='g',alpha=0.3)
    ax1.set_title('Muestra y referencia - '+str(name), loc='left', fontsize=13)
    ax1.legend(loc='best')
    ax1.grid()  

    #Espectro de Frecuencias 
    ax2 = fig.add_subplot(3,1,2)
    ax2.plot(f/1000,g,'.-',lw=0.9)
    #ax2.scatter(armonicos/1000,amplitudes,c='r',marker='+',label='armónicos')
    for item in frec_multip:
        ax2.axvline(item,0,1,color='r',alpha=0.4,lw=0.9)   
    ax2.scatter(f_impar/1000,amp_impar,marker='x',c='tab:orange',label='armónicos impares',zorder=2.5)
    #ax2.scatter(f_par/1000,amp_par,marker='+',c='tab:green',label='armónicos pares',zorder=2.5)
    ax2.set_title('Espectro de frecuencias - {}% - frec max: {:.0f} kHz'.format(filtro*100,frec_limite/1e3), loc='left', fontsize=13)
    ax2.set_xlabel('Frecuencia (kHz)')
    ax2.set_ylabel('|F{$\epsilon$}|')   
    ax2.set_xlim(0,max(f)/1000)
    ax2.legend(loc='best')

    #  Espectro de Fases 
    ax3 = fig.add_subplot(3,1,3)
    ax3.vlines(armonicos/1000,ymin=0,ymax=fases)
    ax3.stem(armonicos/1000,fases,basefmt=' ')
    ax3.scatter(f_impar/1000,fases_impar,marker='x',color='tab:orange',label='armónicos impares',zorder=2.5)
    #ax3.scatter(f_par/1000,fases_par,marker='+',color='tab:green',label='armónicos pares',zorder=2.5)    
    #ax3.vlines(f_impar/1000, ymin=0, ymax=fases_impar,color='tab:orange')
    ax3.set_ylim(-np.pi-0.5,np.pi+0.5)
    ax3.set_yticks([-np.pi,-3*np.pi/4,-np.pi/2,-np.pi/4,0,np.pi/4,np.pi/2,3*np.pi/4,np.pi])
    ax3.set_yticklabels(['-$\pi$','','$-\pi/2$','','0','','$\pi/2$','','$\pi$'])
    ax3.axhline(0,0,max(armonicos)/1000,c='k',lw=0.8)
    for item in frec_multip:
        ax3.axvline(item,.1,0.92,color='r',alpha=0.4,lw=0.9)  
    #ax3.scatter(armonicos/1000,theta,label='fases redef')    
    ax3.set_ylabel('Fase')
    ax3.set_xlabel('Frecuencia (kHz)')
    #ax3.legend(loc='best')
    ax3.set_title('Espectro de fases',loc='left', fontsize=13)
    ax3.set_xlim(0,max(f)/1000)
    ax3.grid(axis='y')
    ax3.legend()

    
    #Redefino angulos p/ Fasorial impar
    r0 = 1
    theta_0 = 0
    r = amp_impar/max(amp_impar)
    defasaje_m =  fases_r-fases_impar
    
    #Grafico 2.0: Espectro Impar, Fasorial, Original+Rec_impar (Muestra)
    fig2 = plt.figure(figsize=(8,12),constrained_layout=True)
    plt.suptitle('Reconstruccion impar',fontsize=20)
    
    # Señal Original + Reconstruida impar
    ax1=fig2.add_subplot(2,1,1)
    ax1.plot(t,y,'.-',lw=0.9,label='Señal original')
    ax1.plot(t,rec_impares,'-',lw=1.3,label='Componentes impares')
    ax1.plot(t,rec_pares,'-',lw=1.1,label='Componentes pares')
    ax1.set_xlabel('t (s)')
    ax1.set_xlim(0,3/armonicos[0])
    ax1.axvspan(0,1/armonicos[0],color='g',alpha=0.3)
    ax1.set_title(str(name))
    ax1.grid() 
    ax1.legend(loc='best')
    ax1.set_ylim(-max(rec_impares)*10,max(rec_impares)*10)
    
    # Espectro en fases impares
    ax2=fig2.add_subplot(2,1,2)
    ax2.scatter(f_impar/1000,amp_impar,marker='o',c='tab:blue',label='Armónicos impares',zorder=2.5)
    ax2.vlines(f_impar/1000, ymin=0, ymax=amp_impar)
    ax2.axvline(armonicos_r/1000, ymin=0, ymax=1,c='tab:red',label='Referencia',lw=1,alpha=0.8)
    ax2.set_title('Espectro de la señal reconstruida', loc='left', fontsize=13)
    ax2.set_xlabel('Frecuencia (kHz)')
    ax2.set_ylabel('|F{$\epsilon$}|')   
    ax2.set_xlim(0,max(f)/1000)
    ax2.set_ylim(0,max(amp_impar)*1.1)
    ax2.grid()
    
    #inset
    axin = ax2.inset_axes([0.4, 0.35, 0.57, 0.6])
    axin.scatter(f_impar/1000,fases_impar,label='Fases')
    axin.vlines(f_impar/1000, ymin=0, ymax=fases_impar)
    axin.scatter(armonicos_r/1000,fases_r,label='Fases ref',c='tab:red')
    axin.vlines(armonicos_r/1000, ymin=0, ymax=fases_r,color='tab:red')
    axin.set_xlabel('Frecuencia (kHz)', fontsize=8)
    axin.set_ylim(-np.pi-0.5,np.pi+0.5)
    axin.set_yticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
    axin.set_yticklabels(['-$\pi$','$-\pi/2$','0','$\pi/2$','$\pi$'])
    axin.axhline(0,0,max(f_impar)/1000,c='k',lw=0.8)
    #axin.set_ylabel('fase')
    #axin.legend(loc='best')
    axin.grid()
    axin.set_title(' Espectro de fases',loc='left', y=0.87, fontsize=10)
    axin.set_xlim(0,max(f_impar)/1000)
    
    defase_fundamental = defasaje_m[0]
    return fig, fig2, rec_impares,rec_impares_bis,defase_fundamental,fases_impar,fases_impar_correg

#18 Abr 23
def fourier_señales_5(t,v,v_r_m,delta_t,polaridad,filtro,frec_limite,name,figuras=0):
    '''
    Toma señales de muestra, calibracion y referencia obtieniendo via fft frecuencias y fases.
    frec_muestreo = sample rate 1/delta_t (tipicamente 1e8 o 5e7).    
    Las señales indefectiblemente deben estar recortadas a N ciclos.
    Establecer frec limite permite filtrar la interferencia de alta señal del generador RF\n
    Se conoce la polaridad de la señal(del ajuste lineal sobre el ciclo paramagnetico). 
    
    Adaptada para usarse en procesador_ciclos_descong.py para medididas con barrido en temperatura
    n muestras
    1 fondo
    
    Retorna: señal reconstruida impar, figuras , espectro:(f,magnitud,fase)
    '''
    t = t - t[0] #Muestra 
    t_r = t.copy() #y su referencia
    y = polaridad*v     #muestra (magnetizacion)
    y_r = v_r_m        #referencia muestra (campo)
    N = len(v)
    N_r_m = len(v_r_m)

    #Para que el largo de los vectores coincida
    if len(t)<len(y): #alargo t
        t = np.pad(t,(0,delta_t*(len(y)-len(t))),mode='linear_ramp',end_values=(0,max(t)+delta_t*(len(y)-len(t))))
    elif len(t)>len(y):#recorto t    
        t=np.resize(t,len(y))

    #Idem referencias
    if len(t_r)<len(y_r): #alargo t
        t_r = np.pad(t_r,(0,delta_t*(len(y_r)-len(t_r))),mode='linear_ramp',end_values=(0,max(t_r)+delta_t*(len(y_r)-len(t_r))))
    elif len(t_r)>len(y_r):#recorto t    
        t_r=np.resize(t_r,len(y_r))

    #Aplico transformada de Fourier
    f = rfftfreq(N,d=delta_t) #obtengo la mitad de los puntos, porque uso rfft
    f_HF = f.copy() 
    #f_HF = f_HF[np.nonzero(f>=frec_limite)] #aca estan el resto 
    
    f = f[np.nonzero(f<=frec_limite)] #limito frecuencias 
    g_aux = fft(y)     #“forward” applies the 1/n factor on the forward tranform
    g = abs(g_aux)                    #magnitud    
    fase = np.angle(g_aux)
    
    #Idem p/ Referencia
    f_r = rfftfreq(N_r_m, d=delta_t)
    f_r = f_r[np.nonzero(f_r<=frec_limite)]
    g_r_aux = fft(y_r)
    g_r = abs(g_r_aux)
    fase_r = np.angle(g_r_aux)
     
    #Recorto vectores hasta frec_limite
    g_HF = g.copy() 
    #g_HF = g_HF[np.nonzero(f>=frec_limite)]#magnitud de HF
    g = np.resize(g,len(f))
    g_r = np.resize(g_r,len(f_r))
    g_HF = np.resize(g_HF,len(f_HF))
    g_HF[np.argwhere(f_HF<=frec_limite)]=0 #Anulo LF

    #Obtengo frecuencias cuya intensidad relativa supera umbral dado por el filtro
    indices,_=find_peaks(abs(g),threshold=max(g)*filtro)
    indices_r,_=find_peaks(abs(g_r),threshold=max(g_r)*filtro)

    # anulo armonico fundamental descomentando siguiente linea
    #indices = np.delete(indices,0)

    for elem in indices:
        if f[elem]<0.9*f_r[indices_r[0]]:
            print('ATENCION: detectada subfrecuencia anómala en el espectro de la señal de muestra {:.2f} Hz\n'.format(f[elem]))
            indices = np.delete(indices,0)
            
    armonicos = f[indices]
    amplitudes = g[indices]
    fases = fase[indices]

    armonicos_r = f_r[indices_r]
    amplitudes_r = g_r[indices_r]
    fases_r = fase_r[indices_r]
    
    #Frecuencias/indices multiplo impar/par de la fundamental
    frec_multip = []
    indx_impar = []
    indx_par=[]

    for n in range(int(frec_limite//int(armonicos[0]))):
        frec_multip.append((2*n+1)*armonicos[0]/1000)
        if (2*n+1)*indices[0]<=len(f):
            indx_impar.append((2*n+1)*indices[0])
            indx_par.append((2*n)*indices[0])

    f_impar= f[indx_impar] #para grafico 1.0
    amp_impar= g[indx_impar]
    fases_impar= fase[indx_impar]
    #09 Feb 24 Unwrapeo las fases
    #16 Feb 24 Tambbien las divido mod 2pi 
    fases_unw = np.unwrap(fases%(2*np.pi))
    fases_impar_unw = np.unwrap(fases_impar%(2*np.pi))
    fases_r = np.unwrap(fases_r%(2*np.pi))

    del indx_par[0]
    f_par= f[indx_par] 
    amp_par= g[indx_par]
    fases_par= fase[indx_par]
    
    # fases_impar_correg=[]
    
        
    #Imprimo tabla 
    print('\nAnalisis armonico sobre fem de referencia (campo) y muestra:')
    print('''Espectro de la señal de Campo:\nFrecuencia (Hz) - Intensidad rel - Fase (rad)''')
    for j in range(len(indices_r)):
        print(f'{armonicos_r[j]:^13.2f}|{amplitudes_r[j]/max(amplitudes_r):^12.2f}|{fases_r[j]:^12.4f}')
    
    print('''\nEspectro de la señal de muestra:\nFrecuencia_Hz|Intensidad rel|Dphi_relativo''')
    for i in range(len(indices)):
        print(f'{armonicos[i]:^13.2f}|{amplitudes[i]/max(amplitudes):^14.2f}|{fases_r[j] - fases[i]:^12.4f}')    
    
    # print('''\nEspectro de la señal de muestra reconstruida (impar):\nnFrecuencia_Hz - Intensidad rel - Fase (rad)''')
    # for i in range(len(f_impar)):
    #     print(f'{f_impar[i]:^13.2f}|{amp_impar[i]/max(amp_impar):^12.2f}|{fases_impar[i]:^12.4f}')    
    
    #Reconstruyo señal IMPAR con ifft
    
    h_aux_impar = np.zeros(len(f),dtype=np.cdouble)
    # h_aux_impar_bis = np.zeros(len(f),dtype=np.cdouble) #para corregir fase
    
    for W in indx_impar:
        # fase_espuria = correccion_fase_por_frecuencia_kHz(f[W]/1000)
        h_aux_impar[W]=g_aux[W]
        # h_aux_impar_bis[W]=g_aux[W]*np.exp(fase_espuria*1j)    #aca compenso el defasaje en el fundamental  

    rec_impares = irfft(h_aux_impar,n=len(t))
    # rec_impares_bis = irfft(h_aux_impar_bis,n=len(t))
    
    
    #Reconstruyo señal PAR con ifft
    h_aux_par = np.zeros(len(f),dtype=np.cdouble)
    for Z in indx_par:
        h_aux_par[Z]=g_aux[Z] 
    rec_pares = irfft(h_aux_par,n=len(t))

    #Reconstruyo señal limitada con ifft
    #g_aux = np.resize(g_aux,len(f))
    #rec_limitada = irfft(g_aux,n=len(t))

    #Reconstruyo señal de alta frecuencia
    rec_HF = irfft(g_HF,n=len(t))
    #Resto HF a la señal original y comparo con reconstruida impar
    resta = y - rec_HF

    #Grafico 1.0 (Muestra): 
    fig = plt.figure(figsize=(8,12),constrained_layout=True)
    plt.suptitle('Análisis Espectral Muestra',fontsize=20)

    #Señal Orig + Ref
    ax1 = fig.add_subplot(3,1,1)
    ax1.plot(t,y/max(y),'.-',lw=0.9,label='Muestra')
    ax1.plot(t_r,y_r/max(y_r),'.-',c='tab:red',lw=0.9,label='Referencia')
    ax1.set_xlabel('t (s)')
    ax1.set_xlim(0,2/armonicos[0])
    ax1.axvspan(0,1/armonicos[0],color='g',alpha=0.3)
    ax1.set_title('Muestra y referencia - '+str(name), loc='left', fontsize=13)
    ax1.legend(loc='best')
    ax1.grid()  

    #Espectro de Frecuencias 
    ax2 = fig.add_subplot(3,1,2)
    ax2.plot(f/1000,g,'.-',lw=0.9)
    #ax2.scatter(armonicos/1000,amplitudes,c='r',marker='+',label='armónicos')
    for item in frec_multip:
        ax2.axvline(item,0,1,color='r',alpha=0.4,lw=0.9)   
    ax2.scatter(f_impar/1000,amp_impar,marker='x',c='tab:orange',label='armónicos impares',zorder=2.5)
    #ax2.scatter(f_par/1000,amp_par,marker='+',c='tab:green',label='armónicos pares',zorder=2.5)
    ax2.set_title('Espectro de frecuencias - {}% - frec max: {:.0f} kHz'.format(filtro*100,frec_limite/1e3), loc='left', fontsize=13)
    ax2.set_xlabel('Frecuencia (kHz)')
    ax2.set_ylabel('|F{$\epsilon$}|')   
    ax2.set_xlim(0,max(f)/1000)
    ax2.legend(loc='best')

    #  Espectro de Fases 
    ax3 = fig.add_subplot(3,1,3)
    #ax3.vlines(armonicos/1000,ymin=0,ymax=fases_unw)
    ax3.stem(armonicos/1000,fases_unw,basefmt=' ')
    ax3.scatter(f_impar/1000,fases_impar_unw,marker='s',color='tab:orange',label='armónicos impares',zorder=2.5)
    #ax3.scatter(f_par/1000,fases_par,marker='+',color='tab:green',label='armónicos pares',zorder=2.5)    
    #ax3.vlines(f_impar/1000, ymin=0, ymax=fases_impar,color='tab:orange')
    #ax3.set_ylim(-np.pi-0.5,np.pi+0.5)
    #ax3.set_yticks([-np.pi,-3*np.pi/4,-np.pi/2,-np.pi/4,0,np.pi/4,np.pi/2,3*np.pi/4,np.pi])
    #ax3.set_yticklabels(['-$\pi$','','$-\pi/2$','','0','','$\pi/2$','','$\pi$'])
    ax3.axhline(0,0,max(armonicos)/1000,c='k',lw=0.8)
    for item in frec_multip:
        ax3.axvline(item,.1,0.92,color='r',alpha=0.4,lw=0.9)  
    #ax3.scatter(armonicos/1000,theta,label='fases redef')    
    ax3.set_ylabel('Fase unw')
    ax3.set_xlabel('Frecuencia (kHz)')
    #ax3.legend(loc='best')
    ax3.set_title('Espectro de fases',loc='left', fontsize=13)
    ax3.set_xlim(0,max(f)/1000)
    ax3.grid(axis='y')
    ax3.legend()

    
    #Redefino angulos p/ Fasorial impar
    r0 = 1
    theta_0 = 0
    r = amp_impar/max(amp_impar)
    defasaje_m =  fases_r -fases_impar
    
    #Grafico 2.0: Espectro Impar, Fasorial, Original+Rec_impar (Muestra)
    fig2 = plt.figure(figsize=(8,12),constrained_layout=True)
    plt.suptitle('Reconstruccion impar',fontsize=20)
    
    # Señal Original + Reconstruida impars
    ax1=fig2.add_subplot(3,1,1)
    ax1.plot(t,y,'.-',lw=0.9,label='Señal original')
    ax1.plot(t,rec_impares,'-',lw=1.3,label='Componentes impares')
    ax1.plot(t,rec_pares,'-',lw=1.1,label='Componentes pares')
    ax1.set_xlabel('t (s)')
    ax1.set_xlim(0,3/armonicos[0])
    ax1.axvspan(0,1/armonicos[0],color='g',alpha=0.3)
    ax1.set_title(str(name))
    ax1.grid() 
    ax1.legend(loc='best')
    ax1.set_ylim(-1.4*max(rec_impares),max(rec_impares)*1.4)
    
    # Espectro en fases impares
    ax2=fig2.add_subplot(3,1,2)
    ax2.scatter(f_impar/1000,amp_impar,marker='o',c='tab:blue',label='Armónicos impares',zorder=2.5)
    ax2.vlines(f_impar/1000, ymin=0, ymax=amp_impar)
    ax2.axvline(armonicos_r/1000, ymin=0, ymax=1,c='tab:red',label='Referencia',lw=1,alpha=0.8)
    ax2.set_title('Espectro de la señal reconstruida', loc='left', fontsize=13)
    ax2.set_xlabel('Frecuencia (kHz)')
    ax2.set_ylabel('|F{$\epsilon$}|')   
    ax2.set_xlim(0,max(f)/1000)
    ax2.set_ylim(0,max(amp_impar)*1.1)
    ax2.grid()
    
    # Fase
    ax3=fig2.add_subplot(3,1,3)
    # ax3 = ax2.inset_axes([0.4, 0.35, 0.57, 0.6])
    ax3.scatter(f_impar/1000,fases_impar_unw,label='Fases')
    ax3.vlines(f_impar/1000, ymin=0, ymax=fases_impar_unw)
    ax3.scatter(armonicos_r/1000,fases_r,label='Fases ref',c='tab:red')
    #ax3.vlines(armonicos_r/1000, ymin=0, ymax=fases_r,color='tab:red')
    ax3.set_xlabel('Frecuencia (kHz)', fontsize=8)
    #ax3.set_ylim(-np.pi-0.5,np.pi+0.5)
    #ax3.set_yticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
    #ax3.set_yticklabels(['-$\pi$','$-\pi/2$','0','$\pi/2$','$\pi$'])
    ax3.axhline(0,0,max(f_impar)/1000,c='k',lw=0.8)
    #ax3.set_ylabel('fase')
    #ax3.legend(loc='best')
    ax3.grid()
    ax3.set_title(' Espectro de fases',loc='left', y=0.87, fontsize=10)
    ax3.set_xlim(0,max(f)/1000)
    
    
    phi_0 = defasaje_m[0]
    
    f_0 = f_impar[0] 
    amp_0 = amp_impar[0]
    fase_0 = fases_impar[0]
    
    espectro_f_amp_fase_fem = (f_impar,amp_impar,fases_impar)
    espectro_f_amp_fase_ref = (armonicos_r[0],amplitudes_r[0],fases_r[0])
    return fig, fig2, rec_impares, phi_0,f_0,amp_0,fase_0,espectro_f_amp_fase_fem,espectro_f_amp_fase_ref#,h_aux_impar,g_r_aux[indices_r]

#%% lector de temperaturas 
def lector_templog(directorio):
    '''
    Busca archivo *templog.csv en directorio especificado.
    muestras = False plotea solo T(dt). 

    muestras = True plotea T(dt) con las muestras superpuestas

    Retorna arrys timestamp,temperatura y plotea el log completo 
    '''
    #levanto el log de temperaturas en .csv

    if fnmatch.filter(os.listdir(directorio),'*templog*'):
        #toma el 1ero en orden alfabetico
        dir_templog = os.path.join(directorio,fnmatch.filter(os.listdir(directorio),'*templog*')[0])

        data = pd.read_csv(dir_templog,sep=';',header=5,
                            names=('Timestamp','Temperatura_ambiente','Temperatura'),usecols=(0,1,2),
                            decimal=',',engine='python') 
        
        temperatura = pd.Series(data['Temperatura']).to_numpy(dtype=float)
        temp_ambiente= pd.Series(data['Temperatura_ambiente']).to_numpy(dtype=float)
        timestamp=np.array([datetime.strptime(date,'%Y/%m/%d %H:%M:%S') for date in data['Timestamp']]) 
        
        # fig, ax = plt.subplots(figsize=(10,5))
        # ax.plot(timestamp,temperatura,'.-',label=dir_templog.split('_')[-1]+' CH1' )
        # ax.plot(timestamp,temp_ambiente,'.-',label=dir_templog.split('_')[-1]+ ' CH2')
        # plt.grid()
        # plt.ylabel('Temperatura (ºC)')
        # fig.autofmt_xdate()
        # plt.legend(loc='best')  
        # plt.tight_layout()
        # plt.xlim(timestamp[0],timestamp[-1])
        # #plt.show()
        return timestamp,temperatura, temp_ambiente
    else:
        print('No se encuentra archivo templog.csv en el directorio:',directorio)


def lector_templog_2(directorio):
    '''
    Busca archivo *templog.csv en directorio especificado.
    Retorna arrys timestamp,temperatura_CH1,temperatura_CH2 y plotea el log completo 
    '''
    #levanto el log de temperaturas en .csv

    if fnmatch.filter(os.listdir(directorio),'*templog*'):
        #toma el 1ero en orden alfabetico
        dir_templog = os.path.join(directorio,fnmatch.filter(os.listdir(directorio),'*templog*')[0])

        data = pd.read_csv(dir_templog,sep=';',header=5,
                            names=('Timestamp','T_CH1','T_CH2'),usecols=(0,1,2),
                            decimal=',',engine='python') 
        
        temp_CH1 = pd.Series(data['T_CH1']).to_numpy(dtype=float)
        temp_CH2= pd.Series(data['T_CH2']).to_numpy(dtype=float)
        timestamp=np.array([datetime.strptime(date,'%Y/%m/%d %H:%M:%S') for date in data['Timestamp']]) 
        
        #fig, ax = plt.subplots(figsize=(10,5))
        #ax.plot(timestamp,temp_CH1,'.-',label=dir_templog.split('_')[-1]+' CH1' )
        #ax.plot(timestamp,temp_CH2,'.-',label=dir_templog.split('_')[-1]+ ' CH2')
        #plt.grid()
        #plt.ylabel('Temperaura (ºC)')
        #fig.autofmt_xdate()
        #plt.legend(loc='best')  
        #plt.tight_layout()
        #plt.xlim(timestamp[0],timestamp[-1])
        #plt.show()
        

        return timestamp,temp_CH1, temp_CH2
    else:
        print('No se encuentra archivo templog.csv en el directorio:',directorio)

#%%
def correccion_fase_por_frecuencia_kHz(f):
    '''Retorna el defasaje espurio, en rad, de acuerdo a la frecuencia en kHz.
    Basado en respuesta lineal.
    f: frecuencia en kHz
    Esto se basa en medidas del Dy2O3 en el rango [98; 300] kHz, y [11; 52] kA/m
    Analisis realizados el 01 Dic 22 '''
    
    m = ufloat(7.016550e-04,3.043333e-08)
    n = ufloat(-5.755735e-02,1.476908e-03) 
    d_phi = m.nominal_value*f+n.nominal_value
    return d_phi
#%%
#%% Susceptibilidad a M = 0
def susceptibilidad_M_0(campo,magnetizacion,label,Hc_mean):
    def lineal(x,m,n):
        return m*x+n
  
    for i in range(len(campo)-1):
        if magnetizacion[i]*magnetizacion[i+1]<0 and (campo[i]>0):
            indice_cruce=i
            print(f'Indice: {indice_cruce}')
            print(f'Campo en indice: {magnetizacion[i+1]}')

    mag_ajuste = magnetizacion[indice_cruce-2:indice_cruce+4]
    campo_ajuste = campo[indice_cruce-2:indice_cruce+4]
    popt,pcvo=curve_fit(lineal,campo_ajuste,mag_ajuste)
    x_aux = np.linspace(0,3.2*Hc_mean,1000)
    y_aux = x_aux*popt[0]+popt[1]


    for i in range(len(campo)-1):
        if magnetizacion[i]*magnetizacion[i+1]<0 and (campo[i]<0):
            indice_cruce_2=i
            print(f'Indice: {indice_cruce_2}')
            print(f'Campo en indice: {magnetizacion[i+1]}')

    mag_ajuste_2 = magnetizacion[indice_cruce_2-2:indice_cruce_2+4]
    campo_ajuste_2 = campo[indice_cruce_2-2:indice_cruce_2+4]
    popt_2,pcvo_2=curve_fit(lineal,campo_ajuste_2,mag_ajuste_2)
    x_aux_2 = np.linspace(-3.2*Hc_mean,0,1000)
    y_aux_2 = x_aux_2*popt_2[0]+popt_2[1]
    suscept_a_M_0 = np.mean([popt[0],popt_2[0]])
    print(f'\nSusceptibilidad a M=0: {suscept_a_M_0}')

    # fig , ax =plt.subplots(figsize=(7,5.5), constrained_layout=True)    
    # ax.plot(campo,magnetizacion,'o-',label=label)
    # ax.plot(x_aux,y_aux,label='AL1')
    # ax.plot(x_aux_2,y_aux_2,label='AL2')

    # plt.legend(loc='best')
    # plt.grid()
    # plt.xlabel('H $(A/m)$')
    # plt.ylabel('M $(A/m)$')
    # plt.title('Susceptibilidad a M=0')
    # plt.text(0.22,0.5,f'Susceptibilidad a M=0\n{suscept_a_M_0:.3e}',fontsize=13,bbox=dict(color='tab:blue',alpha=0.6),
    #             va='center',ha='center',transform=ax.transAxes)
    return suscept_a_M_0#, fig    
    
#%%

if __name__ != '__main__':
    print('Modulo importado')


# %%
