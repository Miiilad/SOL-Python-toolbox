
from __future__ import division
import numpy as np
from vpython import *

import matplotlib.pyplot as plt

from math import atan2
from scipy.integrate import solve_ivp
import os
from shutil import copy

from model import HydroTrans
import MBRLtool
import graphics
import utils

#Choose the Identification algorithm from {'SINDy','RLS'}
select_ID_algorithm={'SINDy':1,'RLS':0} #options={'SINDy','RLS'}
select_output_graphs={'states':1,'value':0,'P':0,'error':0}
#To be used for labeling the output results' directory
version=1
script_path=os.path.dirname(os.path.realpath(__file__))
output_dir_path=script_path+'/Results_{}_{}'.format(version,list(select_ID_algorithm.keys())[list(select_ID_algorithm.values()).index(1)])
try:
    os.makedirs(output_dir_path)
except FileExistsError:
    print('The directory already exists')
#save a copy of the script together with the results
copy(__file__, output_dir_path)



#system specifications
Model=HydroTrans()
n=Model.dim_n
m=Model.dim_m

#choosing and initaializing the Librery of bases
chosen_bases=['1','x','xx']
measure_dim=n
Lib=MBRLtool.Library(chosen_bases,measure_dim,m)
p=Lib._Phi_dim
Theta_dim=(m+1)*p
print('Set of Bases:',Lib._Phi_lbl)
#Building a Database
db_dim=1000
Database=MBRLtool.Database(db_dim=db_dim,Theta_dim=Theta_dim,output_dir_path=output_dir_path,Lib=Lib,load=True,save=True)
X_buf=np.zeros((measure_dim,5))
U_buf=np.zeros((m,3))
#Define and initialize the Identification System Model
SysID_Weights=np.ones((measure_dim,Theta_dim))*0.001
SysID=MBRLtool.SysID(select_ID_algorithm,Database,SysID_Weights,Lib)
#dx_dt= partial(double_pend_cart.dynamics, u=0)
num_of_episode=100
#duration of simulation
t_end=5
#sampling time
h=0.005 #0.01
#time grid points
t=np.arange(0,t_end+h,h)

#To characterise the performance measure
R=np.diag([5,0.1])*1e-2
Q=np.diag([0,20,1,0])
gamma=0.3
Obj=MBRLtool.Objective(Q,R,gamma)

#Tracking reference
x_ref=np.zeros((n))
x_ref[0],x_ref[1],x_ref[2],x_ref[3]=0,0.5,0.3,0

#Define the controller
Controller=MBRLtool.Control(h,Objective=Obj,Lib=Lib,P_init=np.zeros((p,p)))
base_rod_l=20
u_lim=10

#Simulation results
Sim=MBRLtool.SimResults(t,Lib,Database,SysID,Controller,select=select_output_graphs,output_dir_path=output_dir_path)
for j in range(num_of_episode):

    u=np.zeros((m))

    x_init=np.zeros((measure_dim))
    # while abs(x_init[1])<np.pi/180*2:
    #     x_init[1]=np.random.uniform(low=-np.pi/180*5, high=np.pi/180*5, size=(1))
    # x_init[2]=np.random.uniform(low=-np.pi/180*2, high=np.pi/180*2, size=(1))
    #x_init=np.array([0,0.05,0.05,0,0,0])
    #initialize the model
    Model.x[:9]=x_init[:9]
    #initialize the 3D simulation


    x_dot_approx=np.zeros((measure_dim))
    x_dot_hat=np.zeros((measure_dim))

    Controller.P=np.zeros((p,p))

    for i in range(len(t)-2):
        #print(scene.camera.pos)
        #u=np.clip(u, -u_lim, u_lim)
        #u=np.array([60020,60000,60000,60000])*0.75


        Model.integrate(u,[t[i],t[i+1]])
        x_s=Model.Read_sensor()-x_ref




        x_dif=x_s-x_init

        X_buf=np.roll(X_buf,-1)
        X_buf[:,-1]=x_s
        U_buf=np.roll(U_buf,-1)
        U_buf[:,-1]=u


        if i>3:
            x_dot_approx=utils.x_dot_approx(X_buf,h)
            x_dot_hat=SysID.evaluate(X_buf[:,2],U_buf[:,0])

        error=np.linalg.norm(x_dot_approx-x_dot_hat)
        if (i>3) & (error>0.01):# & (np.linalg.norm([X_buf[4:,2]])<3) & (abs(X_buf[1,2])<(np.pi/8)) & (abs(X_buf[2,2])<(np.pi/8)) & (abs(u[0])<u_lim):
            #print('hi')
            Database.add(X_buf[:,2],x_dot_approx,U_buf[:,0])
            SysID_Weights=SysID.update()

        Controller.integrate_P_dot(x_s,SysID_Weights,k=2,sparsify=False)
        Sim.record(i,Model.Read_sensor(),u,Controller.P,Controller.value(),error)
        u=Controller.calculate(x_s,SysID_Weights,u_lim)


        x_init=np.array(x_s)



    Database.DB_save()
    Sim.graph(j,i)
    Sim.printout(j)
