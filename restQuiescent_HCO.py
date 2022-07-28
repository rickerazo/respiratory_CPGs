# swallow_HCO.py
# this model simulates a HCO, that is silent at rest but responds to an inhibitory pulse by creating one oscillation in each neuron
##
## 	Ricardo Erazo-Toscano, PhD
##	Center for Integrative Brain Research, Seattle Children's Hospital, Seattle
## 	JMR Lab
##	Summer 2022
####################################################################################################

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.integrate import solve_ivp

####################################################################################################
v1= -0.01
v2= 0.15
# v2= 0.15
v3= 0.1
v4= 0.145
phi= 1/3

EK = -0.7
# Eleak= -0.65
Eleak= -0.565	## <------- near bifurcation point to get Saddle Node Invariant Circle phenomena
gK = 2
gleak = 0.75
gCa = 1.25
gP = 2
gh= 1
gsyn=4

k_h=1
theta_h = .65

theta_syn = -0.3
k_syn = 0.05

## ODE
def swallow_HCO(t,y):
	sm_a = 0.5*(1+np.tanh((y[0]+theta_syn)/k_syn))
	sm_b = 0.5*(1+np.tanh((y[2]+theta_syn)/k_syn))

	minf_a = 0.5*(1+np.tanh((y[0]-v1)/v2))
	ninf_a = 0.5*(1+np.tanh((y[0]-v3)/v4))
	lamn_a = phi*np.cosh((y[0]-v3)/(2*v4))
	hinf_a = 0.5*(1-np.tanh((y[0]+theta_h)/k_h))

	# 0.5*(1-np.tanh((x+.65)/.3))

	dV_a = gleak*(Eleak-y[0])+gK*y[1]*(EK-y[0])-gCa*minf_a*(y[0]-1)+gsyn*sm_b*(y[0]-1)-gh*hinf_a*hinf_a*(y[0]-2)+Iapp_a
	dw_a = lamn_a*(ninf_a-y[1])

	minf_b = 0.5*(1+np.tanh((y[2]-v1)/v2))
	ninf_b = 0.5*(1+np.tanh((y[2]-v3)/v4))
	lamn_b = phi*np.cosh((y[2]-v3)/(2*v4))
	hinf_b = 0.5*(1-np.tanh((y[2]+theta_h)/k_h))

	dV_b = gleak*(Eleak-y[2])+gK*y[3]*(EK-y[2])-gCa*minf_b*(y[2]-1)+gsyn*sm_a*(y[2]-0.9)-gh*hinf_b*hinf_b*(y[2]-2)+Iapp_b
	dw_b = lamn_b*(ninf_b-y[3])

	f= dV_a, dw_a, dV_b, dw_b
	return f


### ODE settings
tab0 = pd.read_csv('baseline_ic.csv')
init_conds = np.array([tab0.V_A.values[0], tab0.w_A.values[0], tab0.V_B.values[0],tab0.w_B.values[0]])

####################################################################################################
### simulate
t0 = 0
t1 = 10
Iapp_a=0.
Iapp_b=0.
neuron1 = solve_ivp(swallow_HCO, [t0, t1], init_conds)

### apply current
t0 = neuron1.t[-1]
t1 = t0+0.5

# Iapp_a=0.8
# Iapp_b=0.

Iapp_a=0.
Iapp_b=-0.8

neuron2 = solve_ivp(swallow_HCO, [t0, t1], np.array([neuron1.y[0,-1],neuron1.y[1,-1],neuron1.y[2,-1],neuron1.y[3,-1]]))

### evolve
t0 = neuron2.t[-1]
t1 = t0+600
Iapp_a=0.0
Iapp_b=0.0
neuron3 = solve_ivp(swallow_HCO, [t0, t1], np.array([neuron2.y[0,-1],neuron2.y[1,-1],neuron2.y[2,-1],neuron2.y[3,-1]]))


####################################################################################################
####################################### 	Data Visualization
####################################################################################################
####################################### 	Traces
f1 = plt.figure(figsize=(10,10))
ax1 = f1.add_subplot(211)
ax2 = f1.add_subplot(212)

ax1.plot(neuron1.t, neuron1.y[0],c='c')
ax2.plot(neuron1.t, neuron1.y[2],'--',c='orangered')

ax1.plot(neuron2.t, neuron2.y[0],c='c')
ax2.plot(neuron2.t, neuron2.y[2],'--',c='orangered')

ax1.plot(neuron3.t, neuron3.y[0],c='c')
ax2.plot(neuron3.t, neuron3.y[2],'--',c='orangered')

ax1.set_ylim([-0.9,0.3])
ax2.set_ylim([-0.9,0.3])
### output data
output_path = 'prototype_HCO/'
os.makedirs(output_path, exist_ok=True)
ax1.legend()
f1.savefig(output_path+'traces_.png')
############## save and iterate
tab1 = pd.DataFrame()
nrow = {'V_A':neuron3.y[0,-1],'V_B':neuron3.y[2,-1],'w_A':neuron3.y[1,-1],'w_B':neuron3.y[3,-1]}
tab1 = tab1.append(nrow, ignore_index=True)
tab1.to_csv('baseline_ic.csv')
print(nrow)

V = np.arange(-1,1,0.01)

####################################### 	Currents

f4 = plt.figure(figsize=(10,10))
ax4=f4.add_subplot(211)
ax5=f4.add_subplot(212)

time0 = np.append(neuron1.t,neuron2.t)
time1 = np.append(time0, neuron3.t)

VA0 = np.append(neuron1.y[0,:],neuron2.y[0,:])
VA1 = np.append(VA0, neuron3.y[0,:])

VB0 = np.append(neuron1.y[2,:],neuron2.y[2,:])
VB1 = np.append(VB0, neuron3.y[2,:])

sm_a = 0.5*(1+np.tanh((VA1+theta_syn)/k_syn))
sm_b = 0.5*(1+np.tanh((VB1+theta_syn)/k_syn))

minf_a = 0.5*(1+np.tanh((VA1-v1)/v2))
ninf_a = 0.5*(1+np.tanh((VA1-v3)))
lamn_a = phi*np.cosh((VA1-v3)/(2*v4))
hinf_a = 0.5*(1-np.tanh((VA1+theta_h)/k_h))

ax4.plot(time1, gleak*(Eleak-VA1),c='b',label='leak')
# ax4.plot(time1, +gK*y[1]*(EK-VA1))
ax4.plot(time1, -gCa*minf_a*(VA1-1), c='g',label='Ca')
ax4.plot(time1, +gsyn*sm_b*(VA1-0.7), c='r', label='Syn')
ax4.plot(time1, -gh*hinf_a*(VA1-1)+Iapp_b, c='c', label='ih')

minf_b = 0.5*(1+np.tanh((VB1-v1)/v2))
ninf_b = 0.5*(1+np.tanh((VB1-v3)))
lamn_b = phi*np.cosh((VB1-v3)/(2*v4))
hinf_b = 0.5*(1-np.tanh((VB1+theta_h)/k_h))

ax5.plot(time1, gleak*(Eleak-VB1),c='b',label='leak')
ax5.plot(time1, -gCa*minf_b*(VB1-1), c='g',label='Ca')
ax5.plot(time1, +gsyn*sm_a*(VB1-0.7), c='r', label='Syn', linewidth=2)
ax5.plot(time1, -gh*hinf_b*(VB1-1), c='c', label='ih')

ax5.legend()
ax4.set_ylim([-0.42, 1.2])
ax5.set_ylim([-0.42, 1.2])
f4.savefig(output_path+'currents_.png')
##########################################
####################################### 	Activation curves


f8 = plt.figure(figsize=(10,10))
ax8=f8.add_subplot(211)
ax9=f8.add_subplot(212)

ax8.plot(V, 0.5*(1-np.tanh((V+theta_h)/k_h)), c='k',linewidth=0.5)
ax8.plot(VA1, hinf_a, c='c', label='ih')
ax8.plot(V, 0.5*(1+np.tanh((V-v1)/v2)), c='k', linewidth=0.5)
ax8.plot(VA1, 0.5*(1+np.tanh((VA1-v1)/v2)), c='g')


# ax8.plot(VA1, phi*np.cosh((VA1-v3)/(2*v4)), c='brown')
# ax8.plot(V, phi*np.cosh((V-v3)/(2*v4)), c='k')
ax9.plot(V, 0.5*(1-np.tanh((V+theta_h)/k_h)), c='k',linewidth=0.5)
ax9.plot(VB1, hinf_b, c='c', label='ih')
ax9.plot(V, 0.5*(1+np.tanh((V-v1)/v2)), c='k', linewidth=0.5)
ax9.plot(VB1, 0.5*(1+np.tanh((VB1-v1)/v2)), c='g',label='Ica')
# ax9.plot(VB1, phi*np.cosh((VB1-v3)/(2*v4)), c='brown')
# ax9.plot(V, phi*np.cosh((V-v3)/(2*v4)), c='k')
ax9.legend()
f8.savefig(output_path+'activation_curves_')