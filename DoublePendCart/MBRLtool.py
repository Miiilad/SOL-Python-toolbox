import numpy as np
import utils
from functools import partial
from scipy.integrate import solve_ivp
import os
import matplotlib.pyplot as plt

class Library():
    def __init__(self,chosen_bases,n):
        self.chosen_bases=chosen_bases
        self.n=n
        #library of bases
        self.lib={'1':lambda x:[1],\
                 'x':lambda x:x,\
                 'x^2':lambda x: x**2,\
                 'x^3':lambda x: x**3,\
                 'sinx':lambda x:np.sin(x),\
                 '(sinx)^2':lambda x:np.sin(x)**2,\
                 'cosx':lambda x:np.cos(x),\
                 '(cosx)^2':lambda x:np.cos(x)**2,\
                 'xx':lambda x:self.build_product(x)}
        #library of the corresponding gradients
        self.plib={'1':lambda x:x*0,\
                  'x':lambda x:np.diag(x**0),\
                  'x^2':lambda x: np.diag(2*x),\
                  'x^3':lambda x: np.diag(3*(x**2)),\
                  'sinx':lambda x:np.diag(np.cos(x)),\
                  '(sinx)^2':lambda x:np.diag(np.multiply(2*np.sin(x),np.cos(x))),\
                  'cosx':lambda x:np.diag(-np.sin(x)),\
                  '(cosx)^2':lambda x:np.diag(np.multiply(-2*np.cos(x),np.sin(x))),\
                  'xx':lambda x:self.build_pproduct(x)}
        #library of the corresponding labels
        self.lib_labels={'1':'1',\
                        'x':self.build_lbl('x'),\
                        'x^2':self.build_lbl('x^2'),\
                        'x^3':self.build_lbl('x^3'),\
                        'sinx':self.build_lbl('sinx'),\
                        '(sinx)^2':self.build_lbl('(sinx)^2'),\
                        'cosx':self.build_lbl('cosx'),\
                        '(cosx)^2':self.build_lbl('(cosx)^2'),\
                        'xx':self.build_lbl_product('xx')}
        self.lib_dims={'1':1,\
                        'x':self.n,\
                        'x^2':self.n,\
                        'x^3':self.n,\
                        'sinx':self.n,\
                        '(sinx)^2':self.n,\
                        'cosx':self.n,\
                        '(cosx)^2':self.n,\
                        'xx':(self.n**2-self.n)/2}

        self._Phi_lbl=[]
        for i in self.chosen_bases:
            self._Phi_lbl.extend(self.lib_labels[i])
        self._Phi_dim=len(self._Phi_lbl)
        #reserve the memeory required to evaluate Phi
        self._Phi_res=np.zeros((self._Phi_dim))
        #reserve the memeory required to evaluate pPhi
        self._pPhi_res=np.zeros((self._Phi_dim,self.n))

    def build_product(self,x):
        function=np.zeros((int((self.n**2-self.n)/2)))
        ind=0
        for i in range(self.n):
            for j in range(i+1,self.n):
                  function[ind]=x[i]*x[j]
                  ind+=1
        return function
    def build_pproduct(self,x):
        g=np.zeros((int((self.n**2-self.n)/2),self.n))
        ind=0
        for i in range(self.n):
            for j in range(i+1,self.n):
                g[ind][i]=x[j]
                g[ind][j]=x[i]
                ind+=1
        return g
    def build_lbl(self,func_name):
        lbl=[]
        for i in range(self.n):
            index=func_name.find('x')
            lbl.append(func_name[:index+1]+'({})'.format(i+1)+func_name[index+1:])
        return lbl
    def build_lbl_product(self,func_name):
        lbl=[]
        for i in range(self.n):
            for j in range(i+1,self.n):
                index1=func_name.find('x')
                index2=func_name.find('x',index1+1)
                lbl.append(func_name[:index1+1]+'({})'.format(i+1)+func_name[index1+1:index2+1]+'({})'.format(j+1)+func_name[index2+1:])
        return lbl
    def _Phi_(self,x):
        i=0
        for key in self.chosen_bases:
            temp=int(self.lib_dims[key])
            self._Phi_res[i:i+temp]=self.lib[key](x)
            i+=temp
        return self._Phi_res

    def _pPhi_(self,x):
        i=0
        for key in self.chosen_bases:
            temp=int(self.lib_dims[key])
            self._pPhi_res[i:i+temp,:]=self.plib[key](x)
            i+=temp
        return self._pPhi_res




class Control():
    def __init__(self,h,Objective,Lib,P_init):
        self.Objective=Objective
        self.Lib=Lib
        self.Qb=np.zeros((self.Lib._Phi_dim,self.Lib._Phi_dim))
        self.Qb[1:self.Lib.n+1,1:self.Lib.n+1]=self.Objective.Q
        self.P=P_init
        self.h=h

    def integrate_P_dot(self,x,Wt,k,sparsify):
        self.Phi=self.Lib._Phi_(x)
        self.pPhi=self.Lib._pPhi_(x)
        dp_dt=partial(self.P_dot, x=x,Wt=Wt)
        sol=solve_ivp(dp_dt,[0, k*self.h], self.P.flatten(), method='RK45', t_eval=None,rtol=1e-6, atol=1e-6, dense_output=False, events=None, vectorized=False)
        self.P=sol.y[...,-1].reshape((self.Lib._Phi_dim,self.Lib._Phi_dim))
        if (sparsify):
            # print('Pk_{} is:'.format(j))
            # print(Pk)
            #sparsification of Pk
            absPk=np.absolute(self.P)
            maxP=np.amax(absPk)
            small_index = absPk<(0.001*maxP) #np.logical_and(absPk<0.1 , absPk<(0.0001*maxP))
            self.P[small_index]=0
            #print("number of non zero elements in Pk:",np.count_nonzero(self.P))
        #return partial(self.P_dot, x=x,Wt=Wt)
    def P_dot(self,t,P,x,Wt):
        P=P.reshape((self.Lib._Phi_dim,self.Lib._Phi_dim))
        x[1]=utils.rad_regu(x[1])
        x[2]=utils.rad_regu(x[2])
        W=Wt[:,:self.Lib._Phi_dim]
        Wc=Wt[:,self.Lib._Phi_dim:]

        P_pPhi_Wc_Phi=np.matmul(np.matmul(P,self.pPhi),np.matmul(Wc,self.Phi))
        P_pPhi_W=np.matmul(np.matmul(P,self.pPhi),W)
        return (self.Qb-1/self.Objective.R[0]*np.outer(P_pPhi_Wc_Phi,P_pPhi_Wc_Phi)+P_pPhi_W+P_pPhi_W.T-self.Objective.gamma*P).flatten()

    def calculate(self,x,Wt):
        return -(1/self.Objective.R[0])*np.matmul(self.Phi,np.matmul(np.matmul(self.P,self.Lib._pPhi_(x)),np.matmul(Wt[:,self.Lib._Phi_dim:],self.Phi)))
    def value(self):
        return np.matmul(np.matmul(self.Phi,self.P),self.Phi)

class Objective():
    def __init__(self,Q,R,gamma):
        self.gamma=gamma
        self.Q=Q
        self.R=R
    def stage_cost(self,x,u):
        return np.matmul(np.matmul(x,Q),x)+np.matmul(np.matmul(u,R),u)
class Database():
    def __init__(self,db_dim,Theta_dim,output_dir_path,Lib,load=True,save=True):
        self.db_dim=db_dim
        self.output_dir_path=output_dir_path
        self.Lib=Lib
        self.n=self.Lib.n
        self._Phi_dim=self.Lib._Phi_dim
        self.Theta_dim=Theta_dim
        self.save=save

        if load & os.path.exists(self.output_dir_path+'/db_Theta.npy'):
            self.db_Theta=np.load(self.output_dir_path+'/db_Theta.npy')
            self.db_X_dot=np.load(self.output_dir_path+'/db_X_dot.npy')
            self.db_overflow=np.load(self.output_dir_path+'/db_overflow.npy')
            print(self.db_X_dot[1,:10].T)
            print('Theta_dict:',self.db_Theta)
            self.db_index=np.load(output_dir_path+'/db_index.npy')
            self.db_overflow=np.load(output_dir_path+'/db_overflow.npy')
        else:
            self.Lib=Lib
            self.db_Theta=np.zeros((Theta_dim,self.db_dim))
            self.db_X_dot=np.zeros((self.n,self.db_dim))
            self.db_overflow=False
            self.db_index=0
    def add(self,x,x_dot,u):
        self.db_X_dot[:,self.db_index]=x_dot
        _Phi_=self.Lib._Phi_(x)
        self.db_Theta[:self._Phi_dim,self.db_index]=_Phi_
        self.db_Theta[self._Phi_dim:,self.db_index]=_Phi_*u
        self.db_index+=1
        if self.db_index>(self.db_dim-1):
            self.db_overflow=True
            self.db_index=0
        #print(self.db_index)
    def read(self):
        if self.db_overflow:
            db=[self.db_Theta,self.db_X_dot]
        else:
            db=[self.db_Theta[:,:self.db_index],self.db_X_dot[:,:self.db_index]]
        return db
    def DB_save(self):
        if self.save:
            np.save(self.output_dir_path+'/db_Theta.npy', self.db_Theta)
            np.save(self.output_dir_path+'/db_X_dot.npy', self.db_X_dot)
            np.save(self.output_dir_path+'/db_index.npy', self.db_index)
            np.save(self.output_dir_path+'/db_overflow.npy', self.db_overflow)

class SysID():
    def __init__(self,select_ID_algorithm,Database,Weights,Lib):
        self.ID_alg=select_ID_algorithm
        self.DB=Database
        self.Weights=Weights
        self.Lib=Lib
        self.Theta=np.zeros((self.DB.Theta_dim))
    def update(self):
        if self.ID_alg['SINDy']:
            if self.DB.db_overflow:
                self.Weights=(utils.SINDy(self.DB.db_X_dot,self.DB.db_Theta))
            else:
                self.Weights=(utils.SINDy(self.DB.db_X_dot[:,:self.DB.db_index],self.DB.db_Theta[:,:self.DB.db_index]))
            return self.Weights
    def evaluate(self,x,u):
        _Phi_=self.Lib._Phi_(x)
        self.Theta[:self.Lib._Phi_dim]=_Phi_
        self.Theta[self.Lib._Phi_dim:]=_Phi_*u
        return np.matmul(self.Weights,self.Theta)

class SimResults():
    def __init__(self,t,Lib,DB,SysID,Ctrl,output_dir_path,select={'states':1,'value':1,'P':1,'error':1}):
        self.t=t
        len_t=len(t)
        self.Lib=Lib
        self.DB=DB
        self.SysID=SysID
        self.Ctrl=Ctrl
        self.x_s_history=np.zeros((self.Lib.n,len_t))
        self.u_history=np.zeros((1,len_t))
        self.P_history=np.zeros((len_t,self.Lib._Phi_dim,self.Lib._Phi_dim))
        self.V_history=np.zeros((len_t))
        self.error_history=np.zeros((len_t))
        self.select=select
        self.output_dir_path=output_dir_path
    def record(self,i,x_s,u,P,V,error):
        self.x_s_history[:,i]=x_s
        self.u_history[:,i]=u
        self.P_history[i]=P
        self.V_history[i]=V
        self.error_history[i]=error
    def graph(self,j,i):
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>PLOT<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        #plot: the components of matrix 'Pk' vs. 'time' #######################################
        px=self.Lib._Phi_dim
        if self.select['P']:
            fig = plt.figure()
            for ii in range(px):
                for jj in range(px):
                    plt.plot(self.t[:i],self.P_history[:i,ii,jj],'g')

            plt.savefig(self.output_dir_path+'/fig_P{}.png'.format(j))
            plt.close(fig)
            plt.show()

        #plot the 'control' + 'states' of the system vs. 'time' ################################
        if self.select['states']:
            fig1 = plt.figure()
            plt.plot(self.t[:i],self.u_history[0,:i],'c--')
            plt.plot(self.t[:i], self.x_s_history[0,:i], 'r')
            plt.plot(self.t[:i], self.x_s_history[1,:i], 'b')
            plt.plot(self.t[:i], self.x_s_history[2,:i], 'g')
            plt.plot(self.t[:i], self.x_s_history[3,:i], 'm')
            plt.plot(self.t[:i], self.x_s_history[4,:i], '#E67E22')
            plt.plot(self.t[:i], self.x_s_history[5,:i], '#1F618D')
            plt.legend(["Control","Cart position (m)","Angle 1 (rad)","Angle 2 (rad)","Cart velocity (m/sec)","Angular velocity 1(rad/sec)","Angular velocity 2(rad/sec)"], loc=1)
            plt.xlabel('t (sec)')
            plt.ylabel('States and Control')
            #plt.tight_layout()
            plt.ylim((-10, 10))

            plt.grid(color='k', linestyle=':', linewidth=1)
            plt.savefig(self.output_dir_path+'/fig_states_control{}.pdf'.format(j),format='pdf')
            plt.close(fig1)
            plt.show()

            #plot the 'value' + 'parameters' and error of the system vs. 'time' ################################
            fig0, axs = plt.subplots(3, 1)
            b1,=axs[0].plot(self.t[:i],self.V_history[:i],'b')
            #axs[0].set_xlabel('time')
            axs[0].set_ylabel('Value')
            axs[0].grid(color='k', linestyle=':', linewidth=1)

            for ii in range(px):
                for jj in range(px):
                    axs[1].plot(self.t[:i],self.P_history[:i,ii,jj],'g')
            #axs[0].set_xlabel('time')
            axs[1].set_ylabel('Parameters')
            axs[1].grid(color='k', linestyle=':', linewidth=1)

            b1,=axs[2].plot(self.t[:i],self.error_history[:i],'r')
            #axs[0].set_xlabel('time')
            axs[2].set_ylabel('Error')
            axs[2].set_ylim([0, 5])
            axs[2].grid(color='k', linestyle=':', linewidth=1)

            plt.tight_layout()
            plt.savefig(self.output_dir_path+'/fig_states_control_Value_Param_Error{}.pdf'.format(j),format='pdf')
            plt.close(fig0)
            plt.show()
        #plot: the pridiction error ###########################################################
        if self.select['error']:
            fig2 = plt.figure()
            plt.plot(self.t[:i],self.error_history[:i],'g')
            plt.ylim((0,200))
            plt.savefig(self.output_dir_path+'/fig_error{}.png'.format(j))
            plt.close(fig2)
            plt.show()

        #plot: Value ###########################################################
        if self.select['value']:
            fig3 = plt.figure()
            plt.plot(self.t[:i],self.V_history[:i],'b')
            plt.tight_layout()

            plt.savefig(self.output_dir_path+'/fig_value{}.png'.format(j))
            plt.close(fig3)
            plt.show()


        #plot: ROA using the obtained lyapunov function V(x)=Phi'*P*Phi
        # [x_green,x_red]=ROA([-4, 4, -4, 4],[0.1,0.1], Wt, Pk, R, px)
        # fig3=plt.figure()
        # plt.scatter(x_green[0],x_green[1],20,edgecolors='none', c='green')
        # plt.scatter(x_red[0],x_red[1],20,edgecolors='none', c='red')
        # plt.scatter(x_green_LQR[0],x_green_LQR[1],5,edgecolors='none', c='yellow')
        # plt.tight_layout()
        # plt.savefig(output_dir+'/fig_ROA{}.png'.format(j))
        # plt.close(fig3)
        # plt.show()


    def printout(self,j):
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>PRINT<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        #print: identified system
        print('Episode {}:'.format(j+1))
        if self.DB.db_overflow:
            print('Number of samples in database : ',self.DB.db_dim)
        else:
            print('Number of samples in database : ',self.DB.db_index)

        # print('Initial and final step values:')
        # print(initial_stage_value)
        # print(final_stage_value)
        chosen_basis_label=self.Lib._Phi_lbl
        for ii in range(self.Lib.n):
            handle_str='x_dot({}) = '.format(ii+1)
            for jj in range(self.DB.Theta_dim):
                if self.SysID.Weights[ii,jj]!=0:
                    if jj<self.Lib._Phi_dim:
                        handle_str=handle_str+(' {:7.3f}*{} '.format(self.SysID.Weights[ii,jj],chosen_basis_label[jj]))
                    elif jj>=self.Lib._Phi_dim:
                        handle_str=handle_str+(' {:7.3f}*{}*u '.format(self.SysID.Weights[ii,jj],chosen_basis_label[jj-self.Lib._Phi_dim]))
            print(handle_str)
        #print: obtained value function
        handle_str='V(x) = '
        for ii in range(self.Lib._Phi_dim):
            for jj in range(ii+1):
                if (self.Ctrl.P[ii,jj]!=0):
                    if (ii==jj):
                        handle_str=handle_str+'{:7.3f}*{}^2'.format(self.Ctrl.P[ii,jj],chosen_basis_label[jj])
                    else:
                        handle_str=handle_str+'{:7.3f}*{}*{}'.format(2*self.Ctrl.P[ii,jj],chosen_basis_label[ii],chosen_basis_label[jj])
        print(handle_str)
        print("% of non-zero elements in P: {:4.1f} %".format(100*np.count_nonzero(self.Ctrl.P)/(self.Lib._Phi_dim**2)))
