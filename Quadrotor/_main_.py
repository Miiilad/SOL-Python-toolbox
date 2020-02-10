
from __future__ import division
import numpy as np
from vpython import *

import matplotlib.pyplot as plt

from math import atan2
from scipy.integrate import solve_ivp
import os
from shutil import copy

from model import Quadrotor
import MBRLtool
import graphics
import utils

import tkinter as tk
#Choose the Identification algorithm from {'SINDy','RLS'}
select_ID_algorithm={'SINDy':0,'RLS':1} #options={'SINDy','RLS'}
select_output_graphs={'states':1,'value':1,'P':1,'error':1}
#To be used for labeling the output results' directory
version=3
script_path=os.path.dirname(os.path.realpath(__file__))
output_dir_path=script_path+'/Results_{}_{}'.format(version,list(select_ID_algorithm.keys())[list(select_ID_algorithm.values()).index(1)])
try:
    os.makedirs(output_dir_path)
except FileExistsError:
    print('The directory already exists')
#save a copy of the script together with the results
copy(__file__, output_dir_path)



#system specifications
Model=Quadrotor()
n=Model.dim_n
m=Model.dim_m
measure_dim=Model.measure_dim
#choosing and initaializing the Librery of bases
chosen_bases=['1','x']

Lib=MBRLtool.Library(chosen_bases,measure_dim,m)
p=Lib._Phi_dim
Theta_dim=(m+1)*p
print('Set of Bases:',Lib._Phi_lbl)
#Building a Database
db_dim=2000
Database=MBRLtool.Database(db_dim=db_dim,Theta_dim=Theta_dim,output_dir_path=output_dir_path,Lib=Lib,load=0,save=0)
X_buf=np.zeros((measure_dim,5))
U_buf=np.zeros((m,3))
#Define and initialize the Identification System Model
SysID_Weights=np.ones((measure_dim,Theta_dim))*0.001
SysID=MBRLtool.SysID(select_ID_algorithm,Database,SysID_Weights,Lib)

#dx_dt= partial(double_pend_cart.dynamics, u=0)
num_of_episode=100
#duration of simulation
t_end=10
#sampling time
h=0.01 #0.01
#time grid points
t=np.arange(0,t_end+h,h)

#To characterise the performance measure
R=np.diag([1,1,1,1])*4e-4
R=np.diag([1,1,1,1])*0.5e-7
Q=np.diag([200,200,200,50,50,50,1,1,1,5,5,2])*1.5e3
Q=np.diag([0,0,100,0,0,0,0,0,0,1,1,1])
gamma=0.5
# R=np.diag([1,1,1,1])*4e-4
# Q=np.diag([500,500,500,10,10,10,1,1,1,30,30,30])*1.5e3
# gamma=0.00001
Obj=MBRLtool.Objective(Q,R,gamma)


#Tracking reference
x_ref=np.zeros((measure_dim))
x_ref[0],x_ref[1],x_ref[2]=0,0,3
x_ref[11]=0
#Define the controller
Controller=MBRLtool.Control(h,Objective=Obj,Lib=Lib,P_init=np.zeros((p,p)))
u_lim=20000

#Simulation results
Sim=MBRLtool.SimResults(t,Lib,Database,SysID,Controller,select=select_output_graphs,output_dir_path=output_dir_path)
Kb=MBRLtool.Knowledge_base(u_lim)

for j in range(num_of_episode):

    u=np.zeros((m))
    x_ref[:3]=np.random.uniform(low=0, high=1.5, size=(3))
    x_init=np.zeros((measure_dim))
    x_init[:3]=np.random.uniform(low=-1, high=2.5, size=(3))
    # while abs(x_init[1])<np.pi/180*2:
    #     x_init[1]=np.random.uniform(low=-np.pi/180*5, high=np.pi/180*5, size=(1))
    # x_init[2]=np.random.uniform(low=-np.pi/180*2, high=np.pi/180*2, size=(1))
    #x_init=np.array([0,0.05,0.05,0,0,0])
    #initialize the model
    Model.x=np.zeros((n))
    Model.x[:9]=x_init[:9]
    Model.x[9]=1
    Model.x[13]=1
    Model.x[17]=1
    #initialize the 3D simulation
    error=np.zeros((len(t)))
    graphic=graphics.Sim3D(j,Database,x_init,len(t),u_lim,x_ref,measure_dim,realtime=1)

    x_dot_approx=np.zeros((measure_dim))
    x_dot_hat=np.zeros((measure_dim))
    #uncoment this if want to fix parameteres after converging so that a fixed controller can be obtained
    # if j>=1:
    #     Controller.update_P=0
    # else:
    #     Controller.P=np.zeros((p,p))
    bias=np.random.uniform(low=0, high=20000, size=(1))
    len_t=len(t)


    pre_run=5
    for i in range(len(t)-2):


        Model.integrate(u,[t[i],t[i+1]])

        sample=Model.Read_sensor()
        x_s=(sample-x_ref)
        #x_s=Model.Read_sensor()-x_ref
        x_s[:3]=x_s[:3]*(np.ones((3))+np.random.normal(0, 0.005, 3))
        #x_s[3:6]=x_s[3:6]*(np.ones((3))+np.random.normal(0, 0.01, 3))



        #x_dif=sample-x_ref-x_init

        X_buf=np.roll(X_buf,-1)
        X_buf[:,-1]=x_s
        U_buf=np.roll(U_buf,-1)
        U_buf[:,-1]=u


        if i>3:
            x_dot_approx=utils.x_dot_approx(X_buf,h)
            x_dot_hat=SysID.evaluate(X_buf[:,2],U_buf[:,0])

        error[i]=np.linalg.norm(x_dot_approx-x_dot_hat)
        if (i>3):# & (np.linalg.norm([X_buf[4:,2]])<3) & (abs(X_buf[1,2])<(np.pi/8)) & (abs(X_buf[2,2])<(np.pi/8)) & (abs(u[0])<u_lim):
            #print('hi')
            if (error[i]>0.5*np.mean(error[:i])) & (error[i]>0.2) & (abs(x_s[9])<10) & (abs(x_s[10])<10): #(error[i]>0.5*np.mean(error[:i]))
                Database.add(X_buf[:,2],x_dot_approx,U_buf[:,0]) #Database.add(X_buf[:,2],x_dot_approx,1e-3*U_buf[:,0])
                SysID_Weights=SysID.update(X_buf[:,2],x_dot_approx,U_buf[:,0])
                SysID_Weights[:,p:]=SysID_Weights[:,p:]#SysID_Weights[:,p:]=1e-3*SysID_Weights[:,p:]


        if j>=pre_run:
            Controller.integrate_P_dot(x_s,SysID_Weights,k=2,sparsify=False)
            Sim.record(i,Model.Read_sensor(),u,Controller.P,Controller.value(),error[i])
            if i%2==0:u=Controller.calculate(x_s,SysID_Weights,u_lim)

        #if (j>=pre_run) & (i%2==0):u=Controller.calculate(x_s,SysID_Weights,u_lim)#+np.array([35000,35000,35000,35000])

        if i%15==0:
            print(u)
            print('progress={:4.1f}%'.format(i/len_t*100))
            print(error[i])

        if (j<pre_run) & (i%2==0):u=np.random.uniform(low=-500, high=500, size=(4))+np.ones((4))*bias

        #Kb.mode_update(x_s,u)
        #u=Kb.enforce()


        #if i%2==0:u=np.random.uniform(low=-500, high=500, size=(4))+np.ones((4))*bias
        # if i<150:u=np.array([60100,60100,60000,60000])*0.8
        # if i>150:u=np.array([59900,59900,60000,60000])*0.8
        # u=np.array([60000,60000,60000,60000])*0.8
        #print('pitch',i, '-',x_s[9:])


        #x_init=np.array(sample-x_ref)
        graphic.update_realtime(i,sample,u+np.array([35000,35000,35000,35000]),sample[6:9]-x_ref[6:9],h)


        if graphic.collision_check() |  (abs(x_s[9])>40) | (abs(x_s[10])>40):
            print('pitch',x_s[9:11])
            break


        # if np.linalg.norm(Model.Read_sensor()[:3]-x_ref[:3])<0.16:
        #     x_ref[:3]=np.random.uniform(low=-0.5, high=2.5, size=(3))
        #     graphic.ref=vector(x_ref[0],x_ref[2],x_ref[1])



    if not graphic.realtime:
        root = tk.Tk()
        root.title("Countinue simulation...ready?")
        button = tk.Button(root, text='Yes', width=25, command=root.destroy)
        button.pack()
        root.mainloop()

    graphic.run(i,h)
    graphic.reset()
    Database.DB_save()
    SysID.save()
    Sim.graph(j,i)
    Sim.printout(j)
