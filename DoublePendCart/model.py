import numpy as np
from functools import partial
from scipy.integrate import solve_ivp

class DoublePendCart():
    def __init__(self,m=6,m1=3,m2=1,l1=1,l2=2,d1=10,d2=1,d3=0.5,g=9.8):
        self.m=m
        self.m1=m1
        self.m2=m2
        self.l1=l1
        self.l2=l2
        self.g=g
        self.d1=d1
        self.d2=d2
        self.d3=d3
        self.dim_n=6
        self.dim_m=1
        self.x=np.zeros((self.dim_n))
    def integrate(self,u,t_interval):
        dx_dt=partial(self.dynamics, u=u)
        sol=solve_ivp(dx_dt,t_interval, self.x, method='RK45', t_eval=None,rtol=1e-6, atol=1e-6, dense_output=False, events=None, vectorized=False)
        self.x=sol.y[...,-1]
        #return partial(self.dynamics, u=u)
    def dynamics(self,t,x,u):
        y,theta1,theta2,y_d,theta1_d,theta2_d=x

        M=[[self.m+self.m1+self.m2,  self.l1*(self.m1+self.m2)*np.cos(theta1), self.m2*self.l2*np.cos(theta2)],\
           [self.l1*(self.m1+self.m2)*np.cos(theta1), (self.l1**2)*(self.m1+self.m2), self.l1*self.l2*self.m2*np.cos(theta1-theta2)],\
           [self.l2*self.m2*np.cos(theta2),  self.l1*self.l2*self.m2*np.cos(theta1-theta2), (self.l2**2)*self.m2]]
        f=[[self.l1*(self.m1+self.m2)*(theta1_d**2)*np.sin(theta1)+self.m2*self.l2*(theta2_d**2)*np.sin(theta2)-self.d1*y_d+30*u],\
           [-self.l1*self.l2*self.m2*(theta2_d**2)*np.sin(theta1-theta2)+self.g*(self.m1+self.m2)*self.l1*np.sin(theta1)-self.d2*theta1_d],\
           [self.l1*self.l2*self.m2*(theta1_d**2)*np.sin(theta1-theta2)+self.g*self.l2*self.m2*np.sin(theta2)-self.d3*theta2_d]]
        x_dd=np.matmul(np.linalg.inv(M),f)

        return np.concatenate(([y_d,theta1_d,theta2_d],np.array(x_dd).flatten()))
    def Read_sensor(self):
        return self.x
