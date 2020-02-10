import numpy as np
import utils
from functools import partial
from scipy.integrate import solve_ivp

class Pendulum():
    def __init__(self):

        self.dim_n=2
        self.dim_m=1
        self.x=np.zeros((self.dim_n))
        self.x_dot=np.zeros((self.dim_n))


    def integrate(self,u,t_interval):
        dx_dt=partial(self.dynamics, u=u)
        sol=solve_ivp(dx_dt,t_interval, self.x, method='RK45', t_eval=None,rtol=1e-6, atol=1e-6, dense_output=False, events=None, vectorized=False)
        self.x=sol.y[...,-1]
        #return partial(self.dynamics, u=u)
    def dynamics(self,t,x,u):
        m=0.1
        l=0.5
        k=0.1
        x1,x2=x

        x1_d=-x2
        x2_d=-(9.8/l)*np.sin(x1)-(k/m)*x2+(1/(m*l**2))*u
        return [x1_d,x2_d]

    def Read_sensor(self):
        return self.x#self.x[0]
