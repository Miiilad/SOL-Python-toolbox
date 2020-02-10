import numpy as np
import math

def SINDy(dXdt,Theta,lam=0.01):
    W = _reg_(Theta,dXdt) # initial guess: Least-squares
    for k in range(10):
        #find small coefficients
        smallinds = np.absolute(W)<lam
        # threshold
        W[smallinds]=0
        for ind in range(dXdt.shape[0]):
            # n is state dimension
            biginds =np.invert(smallinds[ind,...])
            #Regress dynamics onto remaining terms to find sparse Xi
            #print(np.where(biginds)[0])
            #print(biginds)
            W[ind,biginds]= _reg_(Theta[np.where(biginds)[0].T,...],dXdt[ind,...])
    return W

def identification_RLS(W,S,dXdt,Theta,forget_factor=0.9):
    g=S*Theta*(1/(forget_factor+Theta.T*S*Theta))

    W=W+(dXdt-W*Theta)*g.T
    S=(1/forget_factor)*(S-g*Theta.T*S);
    # %omega_tilda1=omega+S*Theta*(dXdt-omega'* Theta)';
    # %S_tilda=1/(lambda+Theta'*S*Theta) *(S-S*Theta*Theta'*S);
    return [W,S]


def _reg_(a, b):
    return np.linalg.lstsq(a.T, b.T,rcond=None)[0].T


def rad_regu(angle):
    angle=angle%(2*np.pi)
    if angle>np.pi:
        angle=angle-2*np.pi

    return angle
def x_dot_approx(X_buf,h):
    return (X_buf[:,2]-X_buf[:,1])/(h)#double_pend_cart(0, X_buf[...,2], U[i-2],u_lim)# #double_pend_cart(0, X_buf[...,2], U[i-2],u_lim)#    (X_buf[...,0]-8*X_buf[...,1]+8*X_buf[...,3]-X_buf[...,4])/(12*h)  #(X_buf[...,0]-8*X_buf[...,1]+8*X_buf[...,3]-X_buf[...,4])/(12*h) #X_dict[...,m_dict-2]-4*X_dict[...,m_dict-1]+3*X_dict[...,m_dict] (-X_dict[...,m_dict-1]+X_dict[...,m_dict])/(h)


def rot_to_euler(R):
    # %gamma   beta   alpha
    # % x       y      z
    # %Roll   Pitch   Yaw
    # %Phi    Theta   Sai
    psi=math.atan2(R[1,0],R[0,0])*180/np.pi
    theta=math.atan2(-R[2,0],math.sqrt(R[2,1]**2+R[2,2]**2))*180/np.pi
    phi=math.atan2(R[2,1],R[2,2])*180/np.pi

    return [phi, theta, psi]
