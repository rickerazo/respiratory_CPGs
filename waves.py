from __future__ import division
#### Ricardo Erazo, PhD
#### Seattle Children's Research Institute
#### Integrative Brain Research
#### 2022. Open Source Software.

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

nr_neurons = 30
delta1 = 1e-2
sigma1 = delta1*nr_neurons/2
gsyn= 15
domain = np.arange(0,nr_neurons)*delta1
domain = np.array([domain])
Vt = -0.01
tau2 = 0.05
# tspan = [0,13]
tspan = [0,30]
delta2 = 1.

neuronSpace = np.arange(0,nr_neurons*delta1,delta1)

J = np.zeros((nr_neurons,nr_neurons))
for i in np.arange(0,nr_neurons,1):
	J[:,i] = abs(neuronSpace - neuronSpace[i])
J = delta1*np.exp(-J/sigma1)/2/sigma1
J = J-np.diag(np.diag(J))
for i in np.arange(0,nr_neurons,1):
	J[i,i:nr_neurons] = 0
W = gsyn*J
W1 = W[0,:]

#Experimental parameters
Htheta= -0.021
K2theta= 0.0285

#Cell parameters
IpumpMax=0.8
Nais=0.0016
Naih=0.012
Nao=0.115   ###### what's the molar concentration of Na in aCSF?
volume=500e-15
faraday_const = 9.6485e13
gNaP = 6.0
gNa= 105.
ENa= 0.045
gK2= 98.
EK=-0.07
gH= 1.
EH= -0.021
gl= 2.
El= -0.06

Vrev= -0.062
Vglut= 0.02
v2glut= 0.03
################################## excitability gradient gNaP
NaP_vec = np.zeros((1,nr_neurons))[0]+gNaP
for i in range(0,nr_neurons):
    NaP_vec[i]=np.exp(-i*sigma1*0.5)*gNaP

#time scales
Cm= 0.028
tau_h= 0.04050
tau_m= 0.1
tau_k= 2
tau_s= 0.1
tau1= Cm;
int_time = 50

# ODE
#initial conditions
## rest
v0= -0.0421996958
h0= 0.99223221
Na0= 0.001
m0= 0.15
n0= 0.0152445526
mCAT0=0
hCAT0=0
mCAN0=0
## shocked
# v2 = -4.4203449e-2
v2 = -0.44203449
h2 = 9.92877246e-1
m2 = 3.02937855e-1
n2 = 1.52351209e-2

#initial conditions
# y0 = np.array((v0,h0,m0,n0,0))
y0 = np.array((v0,Na0,m0,n0,0))
y2 = np.array((v2,h2,m2,n2,0))

initconds = np.zeros((1,nr_neurons*5))
initconds[0,0:nr_neurons] = v0
initconds[0,nr_neurons:nr_neurons*2] = 0.0001
initconds[0,nr_neurons*2:nr_neurons*3] = m0
initconds[0,nr_neurons*3:nr_neurons*4] = n0
initconds[0,nr_neurons*4:nr_neurons*5] = 0
initconds[0,0] = v2
initconds[0,nr_neurons] = Na0
initconds[0,nr_neurons*2]= m2
initconds[0,nr_neurons*3] = n2
initconds[0,nr_neurons*4] = 0

def HCO(t, y0):
    u = np.reshape(y0, (5,nr_neurons))
    s = u[4,:]
    dy = np.zeros((nr_neurons,5))
    for i in range(0,nr_neurons):
        W0 = W[i,:]
        neuron = u[0:4,i]
        v,Na,m,n= neuron

        ENa= 0.02526*np.log(Nao/Na)

        Ipump = IpumpMax/(1+np.exp((Naih-Na)/Nais))

        mNaPss=1./(1.+np.exp(-120*(v+0.04)))
        mK2ss=1./(1.+np.exp(-83.*(v+K2theta)))
        mSss=1./(1+np.exp(-150*(v+v2glut)))

        # dv = (-1/Cm) * (gNaP*mNaPss*(v-ENa)+ Ipump+gK2*n*n*(v-EK) + gH*m*m*(v-EH) + gl*(v-El) +np.dot(W0,s)*(v-Vglut))
        dv = (-1/Cm) * (NaP_vec[i]*mNaPss*(v-ENa)+ Ipump+gK2*n*n*(v-EK) + gH*m*m*(v-EH) + gl*(v-El) +np.dot(W0,s)*(v-Vglut))
        dNa = -(gNaP*mNaPss*(v-ENa)+3*Ipump)*(1/(volume*faraday_const))
        dm = (1/(1+2*np.exp(1.023*180*(v+Htheta))+np.exp(1.023*500*(v+Htheta))) -m)/tau_m
        dn = (mK2ss - n)/tau_k


        ds = (mSss-s[i])/tau2
        ff = dv,dNa,dm,dn,ds
        dy[i,0:5] = ff
    dv_vec = dy[:,0]
    dNa_vec = dy[:,1]
    dm_vec = dy[:,2]
    dn_vec = dy[:,3]
    ds_vec = dy[:,4]
    temp1 = np.concatenate((dv_vec,dNa_vec),axis=0)
    temp2 = np.concatenate((dm_vec,dn_vec,ds_vec),axis=0)
    f = np.concatenate((temp1,temp2),axis=0)
    return f
def Vt_cross_ctr(t,y0): return y0[ctr]-Vt
Vt_cross_ctr.terminal= False
Vt_cross_ctr.direction = 1
Vt_cross_ctr.terminal= False

hmin = 1e-4
tev = np.arange(tspan[0],tspan[1],hmin)
spike_times = np.zeros((1,nr_neurons))
ctr = 0

# neural_net = solve_ivp(HCO, tspan,initconds[0],method='RK45',t_eval = tev,events =Vt_cross_ctr ,atol=1e-7,rtol=1e-5)
continue_=np.load('initconds.npy')
neural_net = solve_ivp(HCO, tspan,continue_,method='RK45',t_eval = tev,events =Vt_cross_ctr ,atol=1e-7,rtol=1e-5)
np.save('initconds',neural_net.y[:,-1])
fgsz = 9
plt.figure(figsize=(fgsz,fgsz))
for i in range(0,nr_neurons):
        plt.plot(neural_net.t,neural_net.y[i,:]-0.05*i)
		# plt.plot(neural_net.t,neural_net.y[i,:])
plt.xlim([tspan[0],tspan[1]])
plt.savefig('V_g'+str(gsyn)+'.png')
f2=plt.figure(figsize=(fgsz,fgsz))
ax1=f2.add_subplot(211)
ax2=f2.add_subplot(212)
ax1.plot(neural_net.t, neural_net.y[0])
ax2.plot(neural_net.t, neural_net.y[31])
f2.savefig('test.png')
f3=plt.figure(figsize=(4,4))
ax3=f3.add_subplot(111)
connectivity=ax3.imshow(W)
f3.colorbar(connectivity)
f3.savefig('W.png')
plt.close('all')

########## view dynamics
f0=plt.figure(figsize=(24,12))
a0=f0.add_subplot(311)
a1=f0.add_subplot(312)
a2=f0.add_subplot(313)
## neuron 10
choose_neuron=10
v=neural_net.y[0+choose_neuron,:]
na=neural_net.y[1*nr_neurons+1+choose_neuron,:]
s=neural_net.y[4*nr_neurons+1+choose_neuron,:]
# print(np.median(s))
# W0=np.dot(W[choose_neuron,:],s)*(v-Vrev)
ENa= 0.02526*np.log(Nao/na)
Ipump=IpumpMax/(1+np.exp((Naih-na)/Nais))
mNaPss=1./(1.+np.exp(-(1/3.1)*(v+0.052)))
INaP=gNaP*mNaPss*(v-ENa)
a0.plot(neural_net.t,v)
a1.plot(neural_net.t,na)
# a2.plot(neural_net.t,gK2**neural_net.y[2*nr_neurons+1,:]*(v-EK),label='IK')
# a2.plot(neural_net.t,gH**neural_net.y[3*nr_neurons+1,:]*(v-EH),label='IH')
a2.plot(neural_net.t,s*(v-Vglut),label='Isyn')
# a2.plot(neural_net.t,Ipump, label='Ipump')
# a2.plot(neural_net.t,INaP, label='INaP')
a2.legend()
a0.set_xlim([tspan[0],tspan[1]])
a1.set_xlim([tspan[0],tspan[1]])
a2.set_xlim([tspan[0],tspan[1]])
f0.savefig('dynamics.png')