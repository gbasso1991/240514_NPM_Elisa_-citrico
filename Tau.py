#%% Analisis para calculo de tau - levantando de archivo resultados.txt y de ciclo
import numpy as np
import matplotlib.pyplot as plt
import fnmatch
import os
import pandas as pd
import chardet 
import re
from scipy.interpolate import interp1d
from uncertainties import ufloat, unumpy 
#%% LECTOR RESULTADOS
def lector_resultados(path): 
    '''
    Para levantar archivos de resultados con columnas :
    Nombre_archivo	Time_m	Temperatura_(ºC)	Mr_(A/m)	Hc_(kA/m)	Campo_max_(A/m)	Mag_max_(A/m)	f0	mag0	dphi0	SAR_(W/g)	Tau_(s)	N	xi_M_0
    '''
    with open(path, 'rb') as f:
        codificacion = chardet.detect(f.read())['encoding']
        
    # Leer las primeras 6 líneas y crear un diccionario de meta
    meta = {}
    with open(path, 'r', encoding=codificacion) as f:
        for i in range(6):
            line = f.readline()
            if i == 0:
                match = re.search(r'Rango_Temperaturas_=_([-+]?\d+\.\d+)_([-+]?\d+\.\d+)', line)
                if match:
                    key = 'Rango_Temperaturas'
                    value = [float(match.group(1)), float(match.group(2))]
                    meta[key] = value
            else:
                match = re.search(r'(.+)_=_([-+]?\d+\.\d+)', line)
                if match:
                    key = match.group(1)[2:]
                    value = float(match.group(2))
                    meta[key] = value
                    
    # Leer los datos del archivo
    data = pd.read_table(path, header=14,
                         names=('name', 'Time_m', 'Temperatura',
                                'Remanencia', 'Coercitividad','Campo_max','Mag_max',
                                'frec_fund','mag_fund','dphi_fem',
                                'SAR','tau',
                                'N','xi_M_0'),
                         usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13),
                         decimal='.',
                         engine='python',
                         encoding=codificacion)
        
    files = pd.Series(data['name'][:]).to_numpy(dtype=str)
    time = pd.to_datetime(data['Time_m'][:],dayfirst=True)
    # delta_t = np.array([dt.total_seconds() for dt in (time-time[0])])
    temperatura = pd.Series(data['Temperatura'][:]).to_numpy(dtype=float)
    
    Mr = pd.Series(data['Remanencia'][:]).to_numpy(dtype=float)
    Hc = pd.Series(data['Coercitividad'][:]).to_numpy(dtype=float)
    campo_max = pd.Series(data['Campo_max'][:]).to_numpy(dtype=float)
    mag_max = pd.Series(data['Mag_max'][:]).to_numpy(dtype=float)
    
    xi_M_0=  pd.Series(data['xi_M_0'][:]).to_numpy(dtype=float)
     
    SAR = pd.Series(data['SAR'][:]).to_numpy(dtype=float)
    tau = pd.Series(data['tau'][:]).to_numpy(dtype=float)
   
    frecuencia_fund = pd.Series(data['frec_fund'][:]).to_numpy(dtype=float)
    dphi_fem = pd.Series(data['dphi_fem'][:]).to_numpy(dtype=float)
    magnitud_fund = pd.Series(data['mag_fund'][:]).to_numpy(dtype=float)
    
    N=pd.Series(data['N'][:]).to_numpy(dtype=int)
    return meta, files, time,temperatura,  Mr, Hc, campo_max, mag_max, xi_M_0, frecuencia_fund, magnitud_fund , dphi_fem, SAR, tau, N
#%% LECTOR CICLOS
def lector_ciclos(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()[:6]

    metadata = {'filename': os.path.split(filepath)[-1],
                'Temperatura':float(lines[0].strip().split('_=_')[1]),
        "Concentracion_g/m^3": float(lines[1].strip().split('_=_')[1].split(' ')[0]),
            "C_Vs_to_Am_M": float(lines[2].strip().split('_=_')[1].split(' ')[0]),
            "ordenada_HvsI ": float(lines[4].strip().split('_=_')[1].split(' ')[0]),
            'frecuencia':float(lines[5].strip().split('_=_')[1].split(' ')[0])}
    
    data = pd.read_table(os.path.join(os.getcwd(),filepath),header=7,
                        names=('Tiempo_(s)','Campo_(kA/m)','Magnetizacion_(A/m)'),
                        usecols=(0,3,4),
                        decimal='.',engine='python',
                        dtype={'Tiempo_(s)':'float','Campo_(kA/m)':'float','Magnetizacion_(A/m)':'float'})  
    t= pd.Series(data['Tiempo_(s)']).to_numpy()
    H = pd.Series(data['Campo_(kA/m)']).to_numpy(dtype=float)*1000 #A/m
    M= pd.Series(data['Magnetizacion_(A/m)']).to_numpy(dtype=float)#A/m
    
    return t,H,M,metadata
#%% TAU PROMEDIO
def Tau_promedio(filepath,recorto_extremos=20):
    '''Dado un path, toma archivo de ciclo M vs H
     Calcula Magnetizacion de Equilibrio, y Tau pesado con dM/dH
     '''
    t,H,M,meta=lector_ciclos(filepath)
     
    indx_max= np.nonzero(H==max(H))[0][0]
    t_mag = t[recorto_extremos:indx_max-recorto_extremos]
    H_mag = H[recorto_extremos:indx_max-recorto_extremos]
    M_mag = M[recorto_extremos:indx_max-recorto_extremos]

    H_demag = H[indx_max+recorto_extremos:-recorto_extremos] 
    # H_demag = np.concatenate((H_demag[:],H_mag[0:1]))

    M_demag = M[indx_max+recorto_extremos:-recorto_extremos]
    # M_demag = np.concatenate((M_demag[:],M_mag[0:1]))

    #INTERPOLACION de M 
    # Verificar que H_mag esté dentro del rango de H_demag
    #H_mag = H_mag[(H_mag >= min(H_demag)) & (H_mag <= max(H_demag))]

    # INTERPOLACION de M solo para los valores dentro del rango
    interpolador = interp1d(H_demag, M_demag,fill_value="extrapolate")
    M_demag_int = interpolador(H_mag)

    # interpolador=interp1d(H_demag, M_demag)
    # M_demag_int = interpolador(H_mag) 
    
    # Derivadas
    dMdH_mag = np.gradient(M_mag,H_mag)
    dMdH_demag_int = np.gradient(M_demag_int,H_mag)
    dHdt= np.gradient(H_mag,t_mag)

    Meq = (M_mag*dMdH_demag_int + M_demag_int*dMdH_mag)/(dMdH_mag+ dMdH_demag_int)
    dMeqdH = np.gradient(Meq,H_mag)

    Tau = (Meq - M_mag)/(dMdH_mag*dHdt )

    Tau_prom = np.sum(Tau*dMeqdH)/np.sum(dMdH_mag)
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    #%paso a kA/m y ns
    H_mag/=1e3
    H_demag/=1e3
    Tau *=1e9
    Tau_prom*=1e9
    print(meta['filename'])
    print(Tau_prom,'s')

    fig,(ax1,ax2) = plt.subplots(nrows=2,figsize=(7,6),constrained_layout=True)
    #ax1.plot(H,Tau,'-',label='U')
    ax1.plot(H_mag,Tau,'.-')
    ax1.grid()
    ax1.set_xlabel('H (kA/m)')
    ax1.set_ylabel(r'$\tau$ (s)')
    ax1.text(1/2,1/7,rf'<$\tau$> = {Tau_prom:.1f} ns',ha='center',va='center',
             bbox=dict(alpha=0.8),transform=ax1.transAxes,fontsize=11)

    ax1.grid()
    ax1.set_xlabel('H (A/m)')
    ax1.set_ylabel('$\\tau$ (ns)')
    ax1.set_title(r'$\tau$ vs H', loc='left')
    ax1.grid()

    ax2.plot(H_mag,Meq,'-',label='M$_{equilibrio}$')
    ax2.plot(H_mag,M_mag,label='Mag')
    ax2.plot(H_demag,M_demag,label='Demag')
    ax2.grid()
    ax2.legend()
    ax2.set_title('M vs H', loc='left')
    ax2.set_xlabel('H (kA/m)')
    ax2.set_ylabel('M (A/m)')

    axins = ax2.inset_axes([0.6, 0.12, 0.39, 0.4])
    axins.plot(H_mag,Meq,'.-')
    axins.plot(H_mag, M_mag,'.-')
    axins.plot(H_demag,M_demag,'.-')
    axins.set_xlim(-0.1*max(H_mag),0.1*max(H_mag)) 
    axins.set_ylim(-0.1*max(M_mag),0.1*max(M_mag))
    ax2.indicate_inset_zoom(axins, edgecolor="black")
    axins.grid()
    plt.suptitle(meta['filename'])

    return Meq , H_mag, max(H)/1000, Tau , Tau_prom , fig
#%% 135 10 
identif_1='135_10'
dir = os.path.join(os.getcwd(),identif_1)
archivos_resultados = [f for f in os.listdir(dir) if  fnmatch.fnmatch(f, '*resultados*')]
archivos_resultados.sort()
filepaths = [os.path.join(dir,f) for f in archivos_resultados]
meta_1,files_1,time_1,temperatura_1,Mr_1,Hc_1,campo_max_1,mag_max_1,xi_M_0_1,frecuencia_fund_1,magnitud_fund_1,dphi_fem_1,SAR_1_1,tau_1_1,N1 = lector_resultados(filepaths[0])
meta_2,files_2,time_2,temperatura_2,Mr_2,Hc_2,campo_max_2,mag_max_2,xi_M_0_2,frecuencia_fund_2,magnitud_fund_2,dphi_fem_2,SAR_1_2,tau_1_2,N2 = lector_resultados(filepaths[1])
meta_3,files_3,time_3,temperatura_3,Mr_3,Hc_3,campo_max_3,mag_max_3,xi_M_0_3,frecuencia_fund_3,magnitud_fund_3,dphi_fem_3,SAR_1_3,tau_1_3,N3 = lector_resultados(filepaths[2])

taus_1=np.array([ufloat(np.mean(tau_1_1),np.std(tau_1_1)),ufloat(np.mean(tau_1_2),np.std(tau_1_2)),ufloat(np.mean(tau_1_3),np.std(tau_1_3))])*1e9
SARs_1=np.array([ufloat(np.mean(SAR_1_1),np.std(SAR_1_1)),ufloat(np.mean(SAR_1_2),np.std(SAR_1_2)),ufloat(np.mean(SAR_1_3),np.std(SAR_1_3))])

for i,ar in enumerate(archivos_resultados):
    print('File:',ar,f'- tau: {taus_1[i]:.2e} s',f'- SAR: {SARs_1[i]:.1f} W/g')
ufloat(np.mean([t.nominal_value for t in taus_1]),np.std([t.nominal_value for t in taus_1]))

print(f'\nPromedio de las {len(taus_1)} medidas:')
tau1 = np.mean(unumpy.uarray([np.mean(tau_1_1),np.mean(tau_1_2),np.mean(tau_1_3)],[np.std(tau_1_1),np.std(tau_1_2),np.std(tau_1_3)]))*1e9
print(f' tau = {tau1} ns')
SAR1 = ufloat(np.mean([S.nominal_value for S in SARs_1]),np.std([S.nominal_value for S in SARs_1]))
print(f' SAR = {SAR1:.2uf} W/g')

fig,ax= plt.subplots()
ax.plot(tau_1_1,'.-',label='1')
ax.plot(tau_1_2,'.-',label='2')
ax.plot(tau_1_3,'.-',label='3')
ax.text(0.95,0.1,rf'<$\tau$> = {tau1:.1uf} ns',bbox=dict(alpha=0.8),transform=ax.transAxes,ha='right', va='bottom')
plt.legend()
plt.grid()
plt.ylabel(r'$\tau$ (s)')
plt.xlabel('Indx')
plt.title(r'$\tau$ - '+ identif_1)
plt.show()

archivos_ciclos = [f for f in os.listdir(dir) if  fnmatch.fnmatch(f, '*promedio*')]
archivos_ciclos.sort()
filepaths = [os.path.join(dir,f) for f in archivos_ciclos]
for ac in archivos_ciclos:
    print(ac)

fig1,ax1=plt.subplots(constrained_layout=True)

for i,fp in enumerate(filepaths):
    t,H,M,metadata=lector_ciclos(fp)
    ax1.plot(H,M,label=f'{SARs_1[i]:1f} W/g')

ax1.text(0.95,0.1,f'<SAR> = {SAR1:.2uf} W/g',bbox=dict(alpha=0.8),transform=ax.transAxes,ha='right', va='bottom')
ax1.set_ylabel('M (A/m)')
ax1.set_xlabel('H (A/m)')
ax1.legend()
ax1.grid()
plt.title('Ciclos promedio - '+identif_1)
plt.savefig('ciclos_promedio_'+identif_1+'.png',dpi=300)
plt.show()

#%% 135 15 
identif_2='135_15'
dir = os.path.join(os.getcwd(),identif_2)
archivos_resultados = [f for f in os.listdir(dir) if  fnmatch.fnmatch(f, '*resultados*')]
archivos_resultados.sort()
filepaths = [os.path.join(dir,f) for f in archivos_resultados]
meta_1,files_1,time_1,temperatura_1,Mr_1,Hc_1,campo_max_1,mag_max_1,xi_M_0_1,frecuencia_fund_1,magnitud_fund_1,dphi_fem_1,SAR_2_1,tau_2_1,N1 = lector_resultados(filepaths[0])
meta_2,files_2,time_2,temperatura_2,Mr_2,Hc_2,campo_max_2,mag_max_2,xi_M_0_2,frecuencia_fund_2,magnitud_fund_2,dphi_fem_2,SAR_2_2,tau_2_2,N2 = lector_resultados(filepaths[1])
meta_3,files_3,time_3,temperatura_3,Mr_3,Hc_3,campo_max_3,mag_max_3,xi_M_0_3,frecuencia_fund_3,magnitud_fund_3,dphi_fem_3,SAR_2_3,tau_2_3,N3 = lector_resultados(filepaths[2])

taus_2=np.array([ufloat(np.mean(tau_2_1),np.std(tau_2_1)),ufloat(np.mean(tau_2_2),np.std(tau_2_2)),ufloat(np.mean(tau_2_3),np.std(tau_2_3))])
SARs_2=np.array([ufloat(np.mean(SAR_2_1),np.std(SAR_2_1)),ufloat(np.mean(SAR_2_2),np.std(SAR_2_2)),ufloat(np.mean(SAR_2_3),np.std(SAR_2_3))])

for i,ar in enumerate(archivos_resultados):
    print('File:',ar,f'- tau: {taus_2[i]:.2e} s',f'- SAR: {SARs_2[i]:.1f} W/g')
ufloat(np.mean([t.nominal_value for t in taus_2]),np.std([t.nominal_value for t in taus_2]))

print(f'\nPromedio de las {len(taus_2)} medidas:')
tau2 = np.mean(unumpy.uarray([np.mean(tau_2_1),np.mean(tau_2_2),np.mean(tau_2_3)],[np.std(tau_2_1),np.std(tau_2_2),np.std(tau_2_3)]))*1e9
print(f' tau = {tau2} ns')
SAR2 = ufloat(np.mean([S.nominal_value for S in SARs_2]),np.std([S.nominal_value for S in SARs_2]))
print(f' SAR = {SAR2:.2uf} W/g')

fig,ax= plt.subplots()
ax.plot(tau_2_1,'.-',label='1')
ax.plot(tau_2_2,'.-',label='2')
ax.plot(tau_2_3,'.-',label='3')
ax.text(0.95,0.1,rf'<$\tau$> = {tau2:.1uf} ns',bbox=dict(alpha=0.8),transform=ax.transAxes,ha='right', va='bottom')
plt.legend()
plt.grid()
plt.ylabel(r'$\tau$ (s)')
plt.xlabel('Indx')
plt.title(r'$\tau$ - '+ identif_2)
plt.show()

archivos_ciclos = [f for f in os.listdir(dir) if  fnmatch.fnmatch(f, '*promedio*')]
archivos_ciclos.sort()
filepaths = [os.path.join(dir,f) for f in archivos_ciclos]
for ac in archivos_ciclos:
    print(ac)

fig1,ax1=plt.subplots(constrained_layout=True)

for i,fp in enumerate(filepaths):
    t,H,M,metadata=lector_ciclos(fp)
    ax1.plot(H,M,label=f'{SARs_2[i]:1f} W/g')

ax1.text(0.95,0.1,f'<SAR> = {SAR2:.2uf} W/g',bbox=dict(alpha=0.8),transform=ax.transAxes,ha='right', va='bottom')
ax1.set_ylabel('M (A/m)')
ax1.set_xlabel('H (A/m)')
ax1.legend()
ax1.grid()
plt.title('Ciclos promedio - '+identif_2)
plt.savefig('ciclos_promedio_'+identif_2+'.png',dpi=300)
plt.show()
#%% 265 10
identif_3='265_10'
dir = os.path.join(os.getcwd(),identif_3)
archivos_resultados = [f for f in os.listdir(dir) if  fnmatch.fnmatch(f, '*resultados*')]
archivos_resultados.sort()
filepaths = [os.path.join(dir,f) for f in archivos_resultados]

meta_1,files_1,time_1,temperatura_1,Mr_1,Hc_1,campo_max_1,mag_max_1,xi_M_0_1,frecuencia_fund_1,magnitud_fund_1,dphi_fem_1,SAR_3_1,tau_3_1,N1 = lector_resultados(filepaths[0])
meta_2,files_2,time_2,temperatura_2,Mr_2,Hc_2,campo_max_2,mag_max_2,xi_M_0_2,frecuencia_fund_2,magnitud_fund_2,dphi_fem_2,SAR_3_2,tau_3_2,N2 = lector_resultados(filepaths[1])
meta_3,files_3,time_3,temperatura_3,Mr_3,Hc_3,campo_max_3,mag_max_3,xi_M_0_3,frecuencia_fund_3,magnitud_fund_3,dphi_fem_3,SAR_3_3,tau_3_3,N3 = lector_resultados(filepaths[2])

taus_3=np.array([ufloat(np.mean(tau_3_1),np.std(tau_3_1)),ufloat(np.mean(tau_3_2),np.std(tau_3_2)),ufloat(np.mean(tau_3_3),np.std(tau_3_3))])*1e9
SARs_3=np.array([ufloat(np.mean(SAR_3_1),np.std(SAR_3_1)),ufloat(np.mean(SAR_3_2),np.std(SAR_3_2)),ufloat(np.mean(SAR_3_3),np.std(SAR_3_3))])

for i,ar in enumerate(archivos_resultados):
    print('File:',ar,f'- tau: {taus_3[i]:.2e} s',f'- SAR: {SARs_3[i]:.1f} W/g')
ufloat(np.mean([t.nominal_value for t in taus_3]),np.std([t.nominal_value for t in taus_3]))

print(f'\nPromedio de las {len(taus_3)} medidas:')
tau3 = np.mean(unumpy.uarray([np.mean(tau_3_1),np.mean(tau_3_2),np.mean(tau_3_3)],[np.std(tau_3_1),np.std(tau_3_2),np.std(tau_3_3)]))*1e9
print(f' tau = {tau3} ns')
SAR3 = ufloat(np.mean([S.nominal_value for S in SARs_3]),np.std([S.nominal_value for S in SARs_3]))
print(f' SAR = {SAR3:.2uf} W/g')

fig,ax= plt.subplots()
ax.plot(tau_3_1,'.-',label='1')
ax.plot(tau_3_2,'.-',label='2')
ax.plot(tau_3_3,'.-',label='3')
ax.text(0.95,0.1,rf'<$\tau$> = {tau3:.1uf} ns',bbox=dict(alpha=0.8),transform=ax.transAxes,ha='right', va='bottom')
plt.legend()
plt.grid()
plt.ylabel(r'$\tau$ (s)')
plt.xlabel('Indx')
plt.title(identif_3)
#plt.savefig('tau_'+identif_3+'.png',dpi=300)
plt.show()
#CICLOS
archivos_ciclos = [f for f in os.listdir(dir) if  fnmatch.fnmatch(f, '*promedio*')]
archivos_ciclos.sort()
filepaths = [os.path.join(dir,f) for f in archivos_ciclos]
for ac in archivos_ciclos:
    print(ac)

fig2,ax2=plt.subplots(constrained_layout=True)

for i,fp in enumerate(filepaths):
    t,H,M,metadata=lector_ciclos(fp)
    ax2.plot(H,M,label=f'{SARs_3[i]:1f} W/g')
ax2.text(0.95,0.1,f'<SAR> = {SAR3:.2uf} W/g',bbox=dict(alpha=0.8),transform=ax.transAxes,ha='right', va='bottom')
ax2.set_ylabel('M (A/m)')
ax2.set_xlabel('H (A/m)')
ax2.legend()
ax2.grid()
plt.title('Ciclos promedio - '+identif_3)
plt.savefig('ciclos_promedio_'+identif_3+'.png',dpi=300)

plt.show()

#%% 265 15
identif_4='265_15'
dir = os.path.join(os.getcwd(),identif_4)
archivos_resultados = [f for f in os.listdir(dir) if  fnmatch.fnmatch(f, '*resultados*')]
archivos_resultados.sort()
filepaths = [os.path.join(dir,f) for f in archivos_resultados]

meta_1,files_1,time_1,temperatura_1,Mr_1,Hc_1,campo_max_1,mag_max_1,xi_M_0_1,frecuencia_fund_1,magnitud_fund_1,dphi_fem_1,SAR_4_1,tau_4_1,N1 = lector_resultados(filepaths[0])
meta_2,files_2,time_2,temperatura_2,Mr_2,Hc_2,campo_max_2,mag_max_2,xi_M_0_2,frecuencia_fund_2,magnitud_fund_2,dphi_fem_2,SAR_4_2,tau_4_2,N2 = lector_resultados(filepaths[1])
meta_3,files_3,time_3,temperatura_3,Mr_3,Hc_3,campo_max_3,mag_max_3,xi_M_0_3,frecuencia_fund_3,magnitud_fund_3,dphi_fem_3,SAR_4_3,tau_4_3,N3 = lector_resultados(filepaths[2])

taus_4=np.array([ufloat(np.mean(tau_4_1),np.std(tau_4_1)),ufloat(np.mean(tau_4_2),np.std(tau_4_2)),ufloat(np.mean(tau_4_3),np.std(tau_4_3))])*1e9
SARs_4=np.array([ufloat(np.mean(SAR_4_1),np.std(SAR_4_1)),ufloat(np.mean(SAR_4_2),np.std(SAR_4_2)),ufloat(np.mean(SAR_4_3),np.std(SAR_4_3))])

for i,ar in enumerate(archivos_resultados):
    print('File:',ar,f'- tau: {taus_4[i]:.2e} s',f'- SAR: {SARs_4[i]:.1f} W/g')
ufloat(np.mean([t.nominal_value for t in taus_4]),np.std([t.nominal_value for t in taus_4]))

print(f'\nPromedio de las {len(taus_4)} medidas:')
tau4 = np.mean(unumpy.uarray([np.mean(tau_4_1),np.mean(tau_4_2),np.mean(tau_4_3)],[np.std(tau_4_1),np.std(tau_4_2),np.std(tau_4_3)]))*1e9
print(f' tau = {tau4} ns')
SAR4 = ufloat(np.mean([S.nominal_value for S in SARs_4]),np.std([S.nominal_value for S in SARs_4]))
print(f' SAR = {SAR4:.2uf} W/g')

fig,ax= plt.subplots()
ax.plot(tau_4_1,'.-',label='1')
ax.plot(tau_4_2,'.-',label='2')
ax.plot(tau_4_3,'.-',label='3')
ax.text(0.95,0.1,rf'<$\tau$> = {tau4:.1uf} ns',bbox=dict(alpha=0.8),transform=ax.transAxes,ha='right', va='bottom')
plt.legend()
plt.grid()
plt.ylabel(r'$\tau$ (s)')
plt.xlabel('Indx')
plt.title(identif_4)
#plt.savefig('tau_'+identif_4+'.png',dpi=300)
plt.show()
#CICLOS
archivos_ciclos = [f for f in os.listdir(dir) if  fnmatch.fnmatch(f, '*promedio*')]
archivos_ciclos.sort()
filepaths = [os.path.join(dir,f) for f in archivos_ciclos]
for ac in archivos_ciclos:
    print(ac)

fig2,ax2=plt.subplots(constrained_layout=True)

for i,fp in enumerate(filepaths):
    t,H,M,metadata=lector_ciclos(fp)
    ax2.plot(H,M,label=f'{SARs_4[i]:1f} W/g')
ax2.text(0.95,0.1,f'<SAR> = {SAR4:.2uf} W/g',bbox=dict(alpha=0.8),transform=ax.transAxes,ha='right', va='bottom')
ax2.set_ylabel('M (A/m)')
ax2.set_xlabel('H (A/m)')
ax2.legend()
ax2.grid()
plt.title('Ciclos promedio - '+identif_4)
plt.savefig('ciclos_promedio_'+identif_4+'.png',dpi=300)

plt.show()
#%% PLOT ALL

fig,((ax1,ax2),(ax3,ax4))=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)

ax1.plot(tau_1_1*1e9,'.-',label='1')
ax1.plot(tau_1_2*1e9,'.-',label='2')
ax1.plot(tau_1_3*1e9,'.-',label='3')
ax1.axhline(tau1.nominal_value,0,1,c='tab:red',label=f'{tau1} ns')
ax1.axhspan(tau1.nominal_value-tau1.std_dev,tau1.nominal_value+tau1.std_dev,alpha=0.5,color='tab:red')
ax1.set_title(identif_1)

ax2.plot(tau_2_1*1e9,'.-',label='1')
ax2.plot(tau_2_2*1e9,'.-',label='2')
ax2.plot(tau_2_3*1e9,'.-',label='3')
ax2.axhline(tau2.nominal_value,0,1,c='tab:red',label=f'{tau2} ns')
ax2.axhspan(tau2.nominal_value-tau2.std_dev,tau2.nominal_value+tau2.std_dev,alpha=0.5,color='tab:red')
ax2.set_title(identif_2)

ax3.plot(tau_3_1*1e9,'.-',label='1')
ax3.plot(tau_3_2*1e9,'.-',label='2')
ax3.plot(tau_3_3*1e9,'.-',label='3')
ax3.axhline(tau3.nominal_value,0,1,c='tab:red',label=f'{tau3} ns')
ax3.axhspan(tau3.nominal_value-tau3.std_dev,tau3.nominal_value+tau3.std_dev,alpha=0.5,color='tab:red')
ax3.set_title(identif_3)

ax4.plot(tau_4_1*1e9,'.-',label='1')
ax4.plot(tau_4_2*1e9,'.-',label='2')
ax4.plot(tau_4_3*1e9,'.-',label='3')
ax4.axhline(tau4.nominal_value,0,1,c='tab:red',label=f'{tau4} ns')
ax4.axhspan(tau4.nominal_value-tau4.std_dev,tau4.nominal_value+tau4.std_dev,alpha=0.5,color='tab:red')

ax4.set_title(identif_4)
#ax4.text(0.25,1/5,rf'$\tau$ = {tau} s',bbox=dict(alpha=0.8),transform=ax.transAxes,ha='center', va='center')

for ax in [ax1,ax2,ax3,ax4]:
    ax.set_xlabel('Indx')
    ax.set_ylabel(r'$\tau$ (s)')
    ax.legend()
    ax.grid()
plt.suptitle('Ni=0 @citrato - 1ra sintesis',fontsize=14)
plt.savefig('tau_Ni0_C_recubierta.png',dpi=300,facecolor='w')

# #%% PLOTEO TODOS LOS CICLOS PROMEDIO
# # Copiar el contenido de la primera figura al nuevo subplot
# new_fig, (new_ax1, new_ax2) = plt.subplots(1, 2, figsize=(12, 5))

# for line in ax1.get_lines():
#     new_ax1.plot(line.get_xdata(), line.get_ydata(), label=line.get_label())
#     new_ax1.set_title(ax1.get_title())
#     new_ax1.legend(loc='best')

# # Copiar el contenido de la segunda figura al nuevo subplot
# for line in ax2.get_lines():
#     new_ax2.plot(line.get_xdata(), line.get_ydata(), label=line.get_label())
#     new_ax2.set_title(ax2.get_title())
#     new_ax2.legend(loc='best')

# # Ajustar los diseños
# plt.tight_layout()
# plt.show()
# %%
