import numpy as np
import utils
from functools import partial
from scipy.integrate import solve_ivp

class HydroTrans():
    def __init__(self):

        self.dim_n=4
        self.dim_m=2
        self.x=np.zeros((self.dim_n))
        self.x_dot=np.zeros((self.dim_n))

        self.tav_up=0.13
        self.tav_uM=0.22
        self.kp=241.67
        self.kM=283.33
        self.CH=1840.8
        self.Vmaxp=145
        self.VmaxM=170
        self.kleak=0.14
        self.ig=-6.12
        self.ia=-23.3
        self.eta_g=0.98
        self.eta_mh=0.697
        self.d_vc=0.33
        self.TLw=0 ###Arbitrarily chosen
        self.wp=105   ###Arbitrarily chosen
        self.Jv=16512

    def integrate(self,u,t_interval):
        dx_dt=partial(self.dynamics, u=u)
        sol=solve_ivp(dx_dt,t_interval, self.x, method='RK45', t_eval=None,rtol=1e-6, atol=1e-6, dense_output=False, events=None, vectorized=False)
        self.x=sol.y[...,-1]
        #return partial(self.dynamics, u=u)
    def dynamics(self,t,x,u):
        x1,x2,x3,x4=x
        u1,u2=u*1e-3

        x1_d=-1/self.tav_up*x1+self.kp/self.tav_up *u1
        x2_d=-1/self.tav_uM*x2+self.kM/self.tav_uM *u2
        x3_d= 10/self.CH *(self.Vmaxp*self.wp*x1-self.VmaxM*x2*x4-self.kleak*x3)
        x4_d=((self.ig**2)*(self.ia**2)*self.eta_g*self.eta_mh*self.Vmaxp*1e-4*x2*x3-self.d_vc*(self.ia**2)*x4-self.TLw*self.ig*self.ia)/self.Jv

        return [x1_d,x2_d,x3_d,x4_d]
    def Read_sensor(self):
        return self.x
