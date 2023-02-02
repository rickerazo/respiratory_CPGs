from __future__ import division
# breath.py
## 	Ricardo Erazo-Toscano, PhD
##	Center for Integrative Brain Research, Seattle Children's, Seattle
## 	JMR Lab
##	Winter 2022
##	Here, we implement a biophysically accurate neuron to simulate a network
##	of the VRG, comprised of rhythmically active, tonic spiking, silent neurons, and conditionally spiking neurons.
####################################################################################################
import random
import matplotlib.pyplot as plt
import numpy as np
import os
# import sys
import pandas as pd
import time

from scipy.integrate import solve_ivp
workspace0 = 'spikes/'
# workspace1 = 'ramps/Kex_ctrl/'
workspace1 = 'network/synapses_ramp/'
workspace2 = 'network/v01/'
os.makedirs(workspace0, exist_ok=True)
os.makedirs(workspace1, exist_ok=True)
####################################################################################################
############################### timestamp
from datetime import datetime
dt = datetime.now()
ts = datetime.timestamp(dt)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 
# numpy.random.default_rng(seed=0)
########################################################################################## graphic stuff
import matplotlib as mpl
mpl.rcParams['axes.linewidth']=0
font = {'weight':'normal','size':20}
plt.rc('font', **font)
# plt.rcParams['agg.path.chunksize'] = 10000
########################################################################################## set up equations and params
np.random.default_rng(seed=1)
def compute_characteristics(spike_times):
	spike_lag = np.diff(spike_times)

	# last spike in bursts -> mechanism: identify spiking frequency, then when the interspike interval is greater than the mean spiking frequency + 4 std deviations
	# interspike interval tolerance for burst discrimination:
	interspike_tolerance = np.mean(spike_lag)+ np.std(spike_lag)

	p1 = np.nonzero(spike_lag>interspike_tolerance)
	p1 = p1[0]
	last_spike = np.append(p1,np.size(spike_times)-1)

	# first spike in bursts
	first_spike = 0
	first_spike = np.append(first_spike,p1+1)

	events1 = np.nonzero((last_spike-first_spike>=min_spikes)) #minimum five spikes to be considered a burst
	first_spike = first_spike[events1]
	last_spike = last_spike[events1]

	ap = np.zeros((1,np.size(last_spike)))-13 #plotting arctifact
	ap=ap[0]

	# Id median spikes, burst by burst
	middle_spike = np.zeros((1,np.size(first_spike)))
	middle_spike = middle_spike[0]
	mean_Hz = np.zeros((1,np.size(first_spike)))
	mean_Hz = mean_Hz[0]
	for i in range(0,np.size(first_spike)):
		burst = spike_times[first_spike[i]:last_spike[i]+1]
		middle_spike[i] = np.median(burst)
		mean_Hz[i] = np.mean(1/np.diff(burst))


	burst_duration = spike_times[last_spike] - spike_times[first_spike]
	interburst_interval = spike_times[first_spike[1:np.size(first_spike)]] - spike_times[last_spike[0:-1]]
	cycle_period = np.diff(middle_spike)

	burst_duration = burst_duration[~np.isnan(burst_duration)]
	interburst_interval = interburst_interval[~np.isnan(interburst_interval)]
	cycle_period = cycle_period[~np.isnan(cycle_period)]
	mean_Hz = mean_Hz[~np.isnan(mean_Hz)]

	return burst_duration, interburst_interval, cycle_period, mean_Hz
## ODE
def sparse_coupling_exponentialDecay(nr_neurons, delta1, sigma1,gsyn,probability_of_connection):
	neuronSpace = np.arange(0,nr_neurons*delta1,delta1)
	
	J = np.zeros((nr_neurons,nr_neurons))
	for i in np.arange(0,nr_neurons,1):
		J[:,i] = abs(neuronSpace - neuronSpace[i])
	J = delta1*np.exp(-J/sigma1)/2/sigma1
	J = J-np.diag(np.diag(J))
	for i in np.arange(0,nr_neurons,1):
		J[i,i:nr_neurons] = 0
		for k in range(0,nr_neurons):
			rand_gen = np.ceil(np.random.rand()*100)
			# print(rand_gen)
			if rand_gen>=probability_of_connection: J[i,k]=0
	W = gsyn*J


	return W

def gNaP_exponentialDecay(nr_neurons, delta1, sigma1,gNaP):
	neuronSpace = np.arange(0,nr_neurons*delta1,delta1)
	
	J = np.exp(-neuronSpace/sigma1)
	# print(J,neuronSpace)
	# J = J-np.diag(np.diag(J))
	# for i in np.arange(0,nr_neurons,1):
	# 	J[i,i:nr_neurons] = 0
	gNaP_gradient = J*gNaP

	return neuronSpace, gNaP_gradient

def breath_SN(t,y0):
	# print(t)
	v,Nai,hNa,mK,mNaP,mCaT,hCaT,mCAN,mh,Ca,K,glut,vgat=y0
		
	if Ca<0: Ca=Ca_min
	if K<0: K=K_min
	if Nai<0: Nai=Na_min

	ENa = 25.26*np.log(Nae/Nai)
	ECa = (8.314*308/(2*96.485))*np.log(Ca_out/Ca)
	EK = 26.7*np.log(K/K_in)

	random_noise=np.random.random()
	# random_noise=np.random.randn()

	INaF=gNaF*1.0/(1.0+np.exp((v-VmNaF)/kmNaF))*1.0/(1.0+np.exp((v-VmNaF)/kmNaF))*1.0/(1.0+np.exp((v-VmNaF)/kmNaF))*hNa*(v-26.45*np.log(Nae/Nai))  # //Fast Sodium
	INaP=gNaP*mNaP*(v-26.45*np.log(Nae/Nai))
	IK=gK*mK*mK*mK*mK*(v-EK)	# //Potassium
	Ipump=Ipmpmx/(1.0+np.power(Naih*np.exp(0.09*nhillNa*v/-26.45)/Nai,nhillNa))  #  //Pump current from Barmashenko et al. 1999
	ICaT=gCaT*mCaT*mCaT*hCaT*(v-26.45*np.log(Cae/mCAN)) # //T-type Ca2+
	ICAN=(gCAN*(ECAN-EK)/(ENa-EK)*1.0/(1.0+np.power(CaCAN/mCAN,nhillCa))*(v-26.45*np.log(Nae/Nai)) # //Na+ component of ICAN
		+gCAN*(ECAN-ENa)/(EK-ENa)*1.0/(1.0+np.power(CaCAN/mCAN,nhillCa))*(v-EK)) # //K+ component of ICAN
	Ih=gh*mh*mh*(v-Eh)

	Iglut=g_glut*glut*(v-Vrev_glut)
	Ivgat=g_vgat*vgat*(v-Vrev_vgat)

	dv=-(INaF+INaP+IK+Ipump+ICaT+ICAN+Ih
		+gL*(EL-EK)/(ENa-EK)*(v-26.45*np.log(Nae/Nai))   #   //Sodium leak
		+gL*(EL-ENa)/(EK-ENa)*(v-EK)   #   //Potassium leak
		)/capac 

	dNai = (-1.0/(vlm*frdy))*(gL*(EL-EK)/(ENa-EK)*(v-26.45*np.log(Nae/Nai))#//ILNa
		+INaF+INaP+ICAN+3.0*Ipump)

	dhNa= (1.0/(1.0+np.exp((v-VhNaF)/khNaF))-hNa)/(tauhNaF/np.cosh((v-VhNaF)/12.8))
	dmK= (1.0/(1.0+np.exp((v-VmK)/kmK))-mK)/(taumK/(np.exp((v-(VmK-25.0))/40.0)+np.exp((v-(VmK-25.0))/-50.0)))

	dmNaP = (1.0/(1.0+ np.exp((v-VmNaP)/kmNaP))-mNaP)/taumNaP
	dmCaT = (1.0/(1.0+np.exp((v-VmCaT)/kmCaT))-mCaT)/(0.001/(np.exp((v+132.0)/-16.7)+np.exp((v+16.8)/18.2))+0.000612)

	dhCaT = (1.0/(1.0+np.exp((v-VhCaT)/khCaT))-hCaT)/(0.001*np.exp((v+22.0)/-10.5)+0.028)

	dmCAN= -1.0*alphaCa*gCaT*mCaT*mCaT*hCaT*(v-26.45*np.log(Cae/mCAN))-(mCAN-Ca_min)/tauCaPump

	mh_inf = 1/(1+np.exp((v+A_h)/B_h))
	dmh = (mh_inf -mh)/tau_h

	dCa = -alphaCa*(ICaT+P_Ca)-(Ca-Ca_min)/tauCaPump

	# K_bath=4.
	K_extracellular_bath=K_bath+.6*random_noise

	Idiff = 1/tau_diff*(K-K_extracellular_bath)
	Iglia = G_max_rate/(1+np.exp(zK*(v2k-K)))
	dK=gamma*beta*IK-Idiff -Iglia

	glut_ss = 1.0/(1.+np.exp((v-v2_glut)/k_glut))
	vgat_ss = 1.0/(1.+np.exp((v-v2_vgat)/k_vgat))
	dvglut = (glut_ss - glut)/tau_glut
	# dvglut = 0
	dvgat = (vgat_ss - vgat)/tau_vgat
	# dvgat = 0

	# ff=dv,dNai,dhNa,dmK,dmNaP,dmCaT,dhCaT,dmCAN,dmh,dCa,dK
	# print(dv,dNai,dhNa,dmK,dmNaP,dmCaT,dhCaT,dmCAN,dmh,dCa,dK)

	# f = np.concatenate((dv,dNai,dhNa,dmK,dmNaP,dmCaT,dhCaT,dmCAN,dmh,dCa,dK),axis=0)
	f=dv,dNai,dhNa,dmK,dmNaP,dmCaT,dhCaT,dmCAN,dmh,dCa,dK,dvglut,dvgat
	return f


def breath_Network(t,y0):
	uu=np.reshape(y0,(13,nr_neurons))
	# print(y0,uu)
	s_glut=uu[-2,:]
	s_vgat=uu[-1,:]
	dy=np.zeros((nr_neurons,13))
	# print(y0,uu,s_glut)

	for i in range(0,nr_neurons):
		W0= W[i,:]
		W1= w[i,:]
		u=uu[0,i],uu[1,i],uu[2,i],uu[3,i],uu[4,i],uu[5,i],uu[6,i],uu[7,i],uu[8,i],uu[9,i],uu[10,i],uu[11,i],uu[12,i]
		# print(u)
		# print(t)
		v,Nai,hNa,mK,mNaP,mCaT,hCaT,mCAN,mh,Ca,K,glut,vgat=u
		# print(v)
		if Ca<0: Ca=Ca_min
		if K<0: K=K_min
		if Nai<0: Nai=Na_min

		ENa = 25.26*np.log(Nae/Nai)
		ECa = (8.314*308/(2*96.485))*np.log(Ca_out/Ca)
		EK = 26.7*np.log(K/K_in)

		random_noise=np.random.random()

		INaF=gNaF*1.0/(1.0+np.exp((v-VmNaF)/kmNaF))*1.0/(1.0+np.exp((v-VmNaF)/kmNaF))*1.0/(1.0+np.exp((v-VmNaF)/kmNaF))*hNa*(v-26.45*np.log(Nae/Nai))  # //Fast Sodium
		# INaP=gNaP*mNaP*(v-26.45*np.log(Nae/Nai))
		INaP=NaP_gradient[i]*mNaP*(v-26.45*np.log(Nae/Nai))
		IK=gK*mK*mK*mK*mK*(v-EK)	# //Potassium
		Ipump=Ipmpmx/(1.0+np.power(Naih*np.exp(0.09*nhillNa*v/-26.45)/Nai,nhillNa))  #  //Pump current from Barmashenko et al. 1999
		ICaT=gCaT*mCaT*mCaT*hCaT*(v-26.45*np.log(Cae/mCAN)) # //T-type Ca2+
		ICAN=(gCAN*(ECAN-EK)/(ENa-EK)*1.0/(1.0+np.power(CaCAN/mCAN,nhillCa))*(v-26.45*np.log(Nae/Nai)) # //Na+ component of ICAN
			+gCAN*(ECAN-ENa)/(EK-ENa)*1.0/(1.0+np.power(CaCAN/mCAN,nhillCa))*(v-EK)) # //K+ component of ICAN
		Ih=gh*mh*mh*(v-Eh)

		Iglut=g_glut*glut*(v-Vrev_glut)
		Ivgat=g_vgat*vgat*(v-Vrev_vgat)

		dv=-(INaF+INaP+IK+Ipump+ICaT+ICAN+Ih
			+gL*(EL-EK)/(ENa-EK)*(v-26.45*np.log(Nae/Nai))   #   //Sodium leak
			+gL*(EL-ENa)/(EK-ENa)*(v-EK)   #   //Potassium leak
			+np.dot(W0,s_glut)*(v-Vrev_glut)
			+np.dot(W1,s_vgat)*(v-Vrev_vgat)
			# +np.dot()
			)/capac 

		dNai = (-1.0/(vlm*frdy))*(gL*(EL-EK)/(ENa-EK)*(v-26.45*np.log(Nae/Nai))#//ILNa
			+INaF+INaP+ICAN+3.0*Ipump)

		dhNa= (1.0/(1.0+np.exp((v-VhNaF)/khNaF))-hNa)/(tauhNaF/np.cosh((v-VhNaF)/12.8))
		dmK= (1.0/(1.0+np.exp((v-VmK)/kmK))-mK)/(taumK/(np.exp((v-(VmK-25.0))/40.0)+np.exp((v-(VmK-25.0))/-50.0)))

		dmNaP = (1.0/(1.0+ np.exp((v-VmNaP)/kmNaP))-mNaP)/taumNaP
		dmCaT = (1.0/(1.0+np.exp((v-VmCaT)/kmCaT))-mCaT)/(0.001/(np.exp((v+132.0)/-16.7)+np.exp((v+16.8)/18.2))+0.000612)

		dhCaT = (1.0/(1.0+np.exp((v-VhCaT)/khCaT))-hCaT)/(0.001*np.exp((v+22.0)/-10.5)+0.028)

		dmCAN= -1.0*alphaCa*gCaT*mCaT*mCaT*hCaT*(v-26.45*np.log(Cae/mCAN))-(mCAN-Ca_min)/tauCaPump

		mh_inf = 1/(1+np.exp((v+A_h)/B_h))
		dmh = (mh_inf -mh)/tau_h

		dCa = -alphaCa*(ICaT+P_Ca)-(Ca-Ca_min)/tauCaPump

		# K_bath=4.
		K_extracellular_bath=K_bath+.6*random_noise

		Idiff = 1/tau_diff*(K-K_extracellular_bath)
		Iglia = G_max_rate/(1+np.exp(zK*(v2k-K)))
		dK=gamma*beta*IK-Idiff -Iglia

		glut_ss = 1.0/(1.+np.exp((v-v2_glut)/k_glut))
		vgat_ss = 1.0/(1.+np.exp((v-v2_vgat)/k_vgat))
		dvglut = (glut_ss - glut)/tau_glut
		# dvglut = 0
		dvgat = (vgat_ss - vgat)/tau_vgat
		# dvgat = 0

		# ff=dv,dNai,dhNa,dmK,dmNaP,dmCaT,dhCaT,dmCAN,dmh,dCa,dK
		# print(dv,dNai,dhNa,dmK,dmNaP,dmCaT,dhCaT,dmCAN,dmh,dCa,dK)

		# f = np.concatenate((dv,dNai,dhNa,dmK,dmNaP,dmCaT,dhCaT,dmCAN,dmh,dCa,dK),axis=0)
		dy[i,0:13]=dv,dNai,dhNa,dmK,dmNaP,dmCaT,dhCaT,dmCAN,dmh,dCa,dK,dvglut,dvgat
	dv_vec = dy[:,0]
	dNai_vec = dy[:,1]
	dhNa_vec = dy[:,2]
	dmK_vec = dy[:,3]
	dmNaP_vec = dy[:,4]
	dmCaT_vec = dy[:,5]
	dhCaT_vec = dy[:,6]
	dmCAN_vec = dy[:,7]
	dmh_vec = dy[:,8]
	dCa_vec = dy[:,9]
	dK_vec = dy[:,10]
	dvglut_vec = dy[:,11]
	dvgat_vec = dy[:,12]
	# print(np.shape(dv_vec),np.shape(dvgat_vec))
	f=np.concatenate((dv_vec,dNai_vec,dhNa_vec,dmK_vec,dmNaP_vec,dmCaT_vec,dhCaT_vec,dmCAN_vec,dmh_vec,dCa_vec,dK_vec,dvglut_vec,dvgat_vec),axis=0)
	return f

def detect_spikes(t,y0): return y0[0]-VT
detect_spikes.terminal= False
detect_spikes.direction = 1

### Params
VT=-30
capac = 0.028 #//Koizumi et al. 2008 in nF

gNaP = 2.0
gL = 2.75
gK = 98.0 
gNaF = 150.0 

VmNaF = -43.8
VhNaF = -68.0
kmNaF = -6.6 
khNaF = 10.8 
# tauhNaF = 0.0352 #//Rybak et al. 2003
tauhNaF = 8.46e-3 #// Phillips et al. 2022

VmNaP = -47.1
kmNaP = -3.1
taumNaP = 0.001 ## 1 ms

VmK = -45.0
kmK = -11.0
taumK = 0.0075

EL = -68.0
frdy = 9.6485e13	#//Faraday's constant in pC/mmol
vlm = 500e-15 	#//Tomlinson and Coupland 1990, m^3

Ipmpmx = 80.0
Naih = 7.5
# Naih = 6.5
Nae = 120.0
nhillNa = 1.5
Na_min=1e-6

gCaT = 5.0 	#//Elsen & Ramirez 1998
VmCaT = -59.05 	#//Elsen & Ramirez 1998
kmCaT = -2.36 	# //Elsen & Ramirez 1998
VhCaT = -80.72 	# //Elsen & Ramirez 1998
khCaT = 5.31 	# //Elsen & Ramirez 1998
Cae = 4.0 	# //Phillips et al. 2019

gCAN = 1.0
ECAN = 0.0
nhillCa = 0.97 		# //Phillips et al. 2019
CaCAN = 0.00074 	# //Phillips et al. 2019

Ca_min = 0.000005 	# //Phillips et al. 2019
tauCaPump = 0.5 	# //Phillips & Rubin 2022
alphaCa = 0.000025 	# //Phillips et al. 2019
P_Ca = 0.01 ### probability that synaptic current carries Ca2+ ions
Ca_out=4
### Extracellular K+ dynamics
tau_diff = 750e-3
K_bath = 4
zK = 6
v2k = 5
G_max_rate= 10

gamma = 7.214e-3
beta = 14.555

K_in = 140
K_min= 10e-7
#####

gh = 2.0
Eh = -40

B_h = 7.0
A_h = -78.
tau_h= 500e-3


### synaptic currents params
tau_glut = 5e-3 		### Jasisnki et al., 2013 
k_glut= -3.
Vrev_glut = 0.
v2_glut =0.
g_glut=2.

tau_vgat = 30e-3 
k_vgat= -3.
Vrev_vgat = -80.
v2_vgat =0.
g_vgat=1.

######################## gnap gradient
nr_neurons=10
						  # gNaP_exponentialDecay(nr_neurons, delta1, sigma1,gNaP)
neuronSpace, gNaP_gradient= gNaP_exponentialDecay(nr_neurons, 1e-4, 1e-2,gNaP)
print(gNaP_gradient)

NaP_gradient=np.zeros((1,nr_neurons))[0]
for i in range(0,nr_neurons):NaP_gradient[i]= random.choice(gNaP_gradient)
print(NaP_gradient)

# W_glut=sparse_coupled_network(nr_neurons, delta1, sigma1,gsyn,probability_of_connection)
# sparse_coupled_network(nr_neurons, delta1, sigma1,gsyn,probability_of_connection)

# km_h = -6.6 
## initial conditions
# v,Nai,hNa,mK,mNaP,mCaT,hCaT,mCAN,mh,Ca,K
# v,Nai,hNa,mK,mNaP,mCaT,hCaT,mCAN,mh,Ca,K,glut,vgat=y0

initconds = np.zeros((1,nr_neurons*13))
initconds[0,0*nr_neurons:nr_neurons*1] = -0.019399333769278947						## V
initconds[0,1*nr_neurons:nr_neurons*2] = 0.008090615680578466			## Nai
initconds[0,2*nr_neurons:nr_neurons*3] = 0.9746892752219183				## hNa
initconds[0,3*nr_neurons:nr_neurons*4] = 0.26842472007330115			## mK
initconds[0,4*nr_neurons:nr_neurons*5] = 0.26842472007330115			## mNaP
initconds[0,5*nr_neurons:nr_neurons*6] = 7.83630549230382e-12			## mCaT
initconds[0,6*nr_neurons:nr_neurons*7] = 0.8006377696554691				## hCaT
initconds[0,7*nr_neurons:nr_neurons*8] = 0.001522942885107103			## mCAN
initconds[0,8*nr_neurons:nr_neurons*9] = 0.011161274311291531			## mh
initconds[0,9*nr_neurons:nr_neurons*10] = 1e-6							## Ca
initconds[0,10*nr_neurons:nr_neurons*11] = 1e-6							## K
initconds[0,11*nr_neurons:nr_neurons*12] = .3							## mS vglut
initconds[0,12*nr_neurons:nr_neurons*13] = .3							## mS vgat
### ODE settings
# tfin=100
tfin=50
tspan=np.array((0,tfin))
tev = np.arange(tspan[0],tspan[1],1e-3)

[p0,nvars]=np.shape(initconds)
### simulate

# neuron = solve_ivp(breath_SN, tspan,initconds[0],method='RK45',t_eval = tev,atol=1e-7,rtol=1e-5)

# continue_=np.load('continue_.npy')
gK=86.0
gNaP=2.0
Naih = 6.0
Ipmpmx=80.
gh=2.0
K_bath=4.
# continue_=initconds[0]
continue_=np.load(workspace2+'continue_.npy')

for g in np.arange(0,5,.5):
	for gg in np.arange(0.,2,.2):
		
		# sparse_coupling_exponentialDecay(nr_neurons, delta1, sigma1,gsyn,probability_of_connection)
		# g=1
		# gg=0.5
		W=sparse_coupling_exponentialDecay(nr_neurons, 1e-4, 1e-2,g,2)
		w=sparse_coupling_exponentialDecay(nr_neurons, 1e-4, 1e-3,gg,1)

		# var1=stochastic_factor
		# label1 = '{:.2f}'.format(var1)
		# label2 = 'stochastic_factor_'+str(label1)
		label2 = 'Net_'		
		# print(label1)

		tic = time.perf_counter()


		# neuron = solve_ivp(breath_Network, tspan,initconds[0],method='RK45',t_eval = tev,atol=1e-6,rtol=1e-4)

		# initconds=np.load(workspace1+'continue_.npy')
		# neuron = solve_ivp(breath_Network, tspan,initconds,method='RK45',t_eval = tev,atol=1e-6,rtol=1e-4)

		neuron = solve_ivp(breath_Network, tspan,continue_,method='RK45',t_eval = tev,atol=1e-6,rtol=1e-4)


		toc = time.perf_counter()
		print(label2)
		print(f"Simulated the network rhythm in {(toc-tic)/60:0.4f} minutes \n")

		# np.save(workspace1+'continue_.npy',neuron.y[:,-1],allow_pickle=True)
		continue_=neuron.y[:,-1]
		# np.save(workspace1+'continue_.npy',continue_,allow_pickle=True)
		print(np.shape(initconds),np.shape(continue_))


		min_spikes=5
		# burst_duration, interburst_interval, cycle_period, mean_Hz = compute_characteristics(neuron.t_events[0])
		# print('burst_duration=',np.median(burst_duration),'\n interburst_interval=',np.median(interburst_interval),'\n cycle_period=',np.median(cycle_period),'\n mean_spike_f=',np.median(mean_Hz),'\n duty cycle=',str(np.median(burst_duration)/np.median(cycle_period)))

		v=neuron.y[nr_neurons*0:nr_neurons*1]
		Nai=neuron.y[nr_neurons*1:nr_neurons*2]
		hNa=neuron.y[nr_neurons*2:nr_neurons*3]
		mK=neuron.y[nr_neurons*3:nr_neurons*4]
		mNaP=neuron.y[nr_neurons*4:nr_neurons*5]
		mCaT=neuron.y[nr_neurons*5:nr_neurons*6]
		hCAT=neuron.y[nr_neurons*6:nr_neurons*7]
		mCAN=neuron.y[nr_neurons*7:nr_neurons*8]
		mh=neuron.y[nr_neurons*8:nr_neurons*9]
		Ca=neuron.y[nr_neurons*9:nr_neurons*10]
		K=neuron.y[nr_neurons*10:nr_neurons*11]

		ENa = 25.26*np.log(Nae/Nai)
		ECa = (8.314*308/(2*96.485))*np.log(Ca_out/Ca)
		EK = 26.7*np.log(K/K_in)

		IK=gK*mK*mK*mK*mK*(v-EK)	# //Potassium
		Idiff = 1/tau_diff*(K-K_bath)
		Iglia = G_max_rate/(1+np.exp(zK*(v2k-K)))
		Ih=gh*mh*mh*(v-Eh)
		INaP=gNaP*mNaP*(v-26.45*np.log(Nae/Nai))

		Ipump=Ipmpmx/(1.0+np.power(Naih*np.exp(0.09*nhillNa*v/-26.45)/Nai,nhillNa))  #  //Pump current from Barmashenko et al. 1999

		### plot simulation Currents:
		f1 = plt.figure(figsize=(30,20))
		ax1 = f1.add_subplot(111)
		# ax1 = f1.add_subplot(211)
		# ax2 = f1.add_subplot(212)
		# ax3 = f1.add_subplot(313)
		for j in range(0,nr_neurons):
			ax1.plot(neuron.t, v[j]-1e+1*j)
			# ax2.plot(neuron.t, Ipump[j]-1e-2*j,'g')
			# ax2.plot(neuron.t, Ih[j]-1e-2*j,'r')
			# ax2.plot(neuron.t, INaP[j]-1e-2*j,'orange')
			# ax2.plot(neuron.t, IK[j]-1e-2*j,'k')
		ax1.set_xlabel('time(s)')
		ax1.set_ylabel('voltage(mV)')
		ax1.set_title(r'$g_e$ = '+str(g)+r', $g_i$ = '+str(gg))
		f1.tight_layout()

		f1.savefig(workspace1+label2+'g_'+str(g)+'_gg_'+str(gg)+'_currents.png')
		ax1.set_xlim((50,65))

		f1.savefig(workspace1+label2+'g_'+str(g)+'_gg_'+str(gg)+'_currents_zoom.png')


		f0=plt.figure(figsize=(20,10))
		a0=f0.add_subplot(111)
		for j in range(0,nr_neurons):
			a0.plot(neuron.t,gamma*beta*IK[j]-1e-2*j,'orange')
			a0.plot(neuron.t,Idiff[j]-1e-2*j,'green')
			a0.plot(neuron.t,Iglia[j]-1e-2*j,'blue')
			a0.plot(neuron.t,Ipump[j]-1e-2*j,'red',linewidth=4)

		f0.tight_layout()

		f0.savefig(workspace1+label2+'g_'+str(g)+'_gg_'+str(gg)+'_K.png')

		plt.close('all')


		##############

		tau_glut = 15e-3 
		k_glut= -6.
		Vrev_glut = 0.
		v2_glut =0.
		g_glut=2

		tau_vgat = 30e-3 
		k_vgat= -2.
		Vrev_vgat = -80.
		v2_vgat =0.
		g_vgat=1.

		vv=np.arange(-90,10,1)

		glut_ss = 1.0/(1.+np.exp((vv-v2_glut)/k_glut))
		vgat_ss = 1.0/(1.+np.exp((vv-v2_vgat)/k_vgat))

		p0=plt.figure(figsize=(10,10))
		a0=p0.add_subplot(111)
		a0.plot(vv,glut_ss,'green')
		a0.plot(vv,vgat_ss,'red',':')
		p0.savefig(workspace1+'g_'+str(g)+'_gg_'+str(gg)+'syn_activation.png')


		#######################
		W=sparse_coupling_exponentialDecay(100, 1e-4, 1e-3,25,2)
		f1=plt.figure(figsize=(10,10))
		a1=f1.add_subplot(111)
		cm = a1.imshow(W)
		f1.colorbar(cm)
		f1.savefig(workspace1+'g_'+str(g)+'_gg_'+str(gg)+'W.png')

		##### implemented W connectivity matrix function correctly. - spatial decay gsyn
		# need to implement W function with finite support - spatial constant gsyn
		# also need to make gradients of INaP, and assign them randomly - non-space associated
		# then we can start to code the network