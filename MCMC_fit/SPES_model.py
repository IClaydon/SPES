import numpy 
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pylab import sqrt, log, log10, pi, sum, mean, interp, sort, arange, tan
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pickle
from scipy.integrate import odeint,ode
from scipy.integrate import quad
from scipy.integrate import simps
from scipy.special import gammainc as gl
from scipy.special import gammaincc as gu
from scipy.special import gamma as g
from scipy.special import kv
import sys
class SPES:
	def __init__(self, rmax,phi0,eta,B,delta,res,mm):
		self.r = np.array([0])
		self.C = 1 - ((1-B)/eta**2)
                self.mmb=1
                self.mmpe=mm
		#print self.C
		#delta=rhm
		# The 2 variables in the Poisson equation: phi, U, where U = -GM(<r)
		self._y = [phi0, 0]  
		self.phi0 = phi0
		init = [phi0,0]
		self.G = 9/(4*pi)

		# Solve
		sol = ode(self._odes)
		#sol.set_integrator('dopri5',nsteps=1e6,max_step=0.1)
		if res == 0:
			sol.set_integrator('dopri5')#,nsteps=1e6,atol=1e-10,rtol=1e-10)
		if res==1:
			sol.set_integrator('dopri5',max_step=0.1,nsteps=1e6,atol=1e-10,rtol=1e-10)
		sol.set_f_params([phi0,eta,B])
		sol.set_solout(self._logcheck)
		sol.set_initial_value(numpy.array([phi0,0]),0.0001)
		sol.integrate(rmax)

		
		# Save phi, rho and M from Poisson solver
		self.phi = self._y[0] 
		#print self.phi
		phi_in = self.phi[self.phi>0]
		phi_out = self.phi[self.phi<0]
		
		###Checks for infinitys in density and dispersion
		inf1 = (np.exp(init[0]/(eta**2))*gu(1.5,init[0]/(eta**2)))*g(1.5)
		if (inf1==np.inf) or (inf1!=inf1):
			inf1=(init[0]/eta**2)**0.5
		inf = np.exp(phi_in/(eta**2))*gu(1.5,phi_in/(eta**2))*g(1.5)
		inf2 =np.exp(phi_in/(eta**2))*gu(2.5,phi_in/(eta**2))*g(2.5)
		for i in range(len(inf)):
			if (inf[i]==np.inf) or (inf[i]!=inf[i]):
				inf[i] = (phi_in[i]/eta**2)**0.5
			if (inf2[i]==np.inf) or (inf2[i]!=inf2[i]):
				inf2[i] =  (phi_in[i]/eta**2)**1.5
		###Density and Velocity dispersion for r<r_t
		self.rho0 = ((np.exp(init[0])*g(1.5)*gl(1.5,init[0])-(2.0/3.0)*init[0]**(1.5)*B) - (4.0/15.0)*self.C*init[0]**(2.5)+ ((1-B)*eta**3 *inf1))#*kv(0.25,eta**2/init[0])))
		self.rho_b = self.mmb* (np.exp(phi_in)*g(1.5)*gl(1.5,phi_in)-(2.0/3.0)*phi_in**(1.5)*B - (4.0/15.0)*self.C*phi_in**(2.5))
		self.rho_pe = self.mmpe* ((1-B)*eta**3 *inf)#*kv(0.25,eta**2/phi_in))
		self.rho_in = self.rho_b + self.rho_pe
		self.pressure_in = (((np.exp(phi_in)*g(2.5)*gl(2.5,(phi_in))-0.4*phi_in**(2.5)*B) - (4.0/35.0)*self.C*phi_in**(3.5)+ ((1-B)*eta**5 *inf2*kv(0.25,eta**2/phi_in))))
		self.pressure_b = (((np.exp(phi_in)*g(2.5)*gl(2.5,(phi_in))-0.4*phi_in**(2.5)*B) - (4.0/35.0)*self.C*phi_in**(3.5)))
		self.pressure_pe=((1-B)*eta**5 *inf2)#*kv(0.25,eta**2/phi_in))
		self.sigma_b = np.sqrt( 2*self.pressure_b/self.rho_b)
		self.sigma_pe = np.sqrt( 2*self.pressure_pe/self.rho_pe)
		sigma_in = np.sqrt( 2*self.pressure_in/self.rho_in)
		###Density and velocity dispersion for r>r_t
		inf_neg = np.exp(phi_out/eta**2)*g(1.5)
		inf2_neg = np.exp(phi_out/(eta**2))*g(2.5)
		self.rho_out = self.mmpe*(((1-B)*eta**3 *inf_neg))#/self.rho0
		sigma_out = np.sqrt( 2*eta**2*(inf2_neg/inf_neg))
		###Total rho_hat and sigma
		#print len(self.rho_in),len(self.rho_out)
		self.rhohat = np.array(np.concatenate([self.rho_in,self.rho_out]))/np.array(self.rho0)
		self.sigma = np.concatenate([sigma_in,sigma_out])

		###dM/dE & f(E)
		self.phii = np.linspace(0,-phi0,100)
		self.N1=[0]*len(phi_in)
		self.N2=[0]*len(self.phii)
		self.fe = [0]*len(phi_in)
		self.fe_1 = [0]*len(phi_in)
		self.fe_2 = [0]*len(phi_in)
		A = 1./(4*np.pi*np.sqrt(2)*(np.exp(init[0])*g(1.5)*gl(1.5,init[0])-(2.0/3.0)*init[0]**(1.5)*B) - (4.0/15.0)*self.C*init[0]**(2.5) + ((1-B)*eta**3 *inf1))
		self.fe2=[0]*len(self.phii)
		self.fe2_1=[0]*len(self.phii)
		self.fe2_2=[0]*len(self.phii)
		for i in range(len(phi_in)):
			self.N1[i] = ((4*np.pi)*(np.exp(phi_in[i])-B-(self.C*phi_in[i]))*simps((self.r[0:(i+1)]**2 * np.sqrt((-phi_in[i] + phi_in[0:(i+1)]))),self.r[0:(i+1)])/self.rho0)
			self.fe[i] = A*(np.exp(phi_in[i])-B-(self.C*phi_in[i]))
			self.fe_1[i] = A*(np.exp(phi_in[i])-self.C)
			self.fe_2[i] = A*(np.exp(phi_in[i]))	
		for i in range(len(self.phii)):
			self.N2[i] = ((1-B) * (4*np.pi/np.sqrt(2))*np.exp(self.phii[i]/eta**2)*simps(self.r[self.phi>0]**2 * np.sqrt(2*(-self.phii[i] + phi_in)),self.r[self.phi>0])/self.rho0)
			self.fe2[i] = A*(1-B)*(np.exp(self.phii[i]/eta**2))
			self.fe2_1[i] = A*(1-B)*(np.exp(self.phii[i]/eta**2))/eta**2
			self.fe2_2[i] = A*(1-B)*(np.exp(self.phii[i]/eta**2))/eta**4
 
		###Scale values 
		count = 0
		Mss = []
		
		self.Mf = (4*np.pi*simps((self.r[0:len(phi_in)]**2)*self.rhohat[0:len(phi_in)],self.r[0:len(phi_in)]))
		for i in range(len(phi_in)-1):
			Mc =(4*np.pi*simps((self.r[0:i+1]**2)*self.rhohat[0:i+1],self.r[0:i+1]))
			if (Mc > 0.5*self.Mf) & (count==0):
				count = 1
				self.rhm = self.r[i]
				break
		self.rt = np.interp(0,-self.phi,self.r)
		self.Mt = (4*np.pi*simps((self.r[self.r<self.rt]**2)*self.rhohat[self.r<self.rt],self.r[self.r<self.rt]))
		#self.rs = (1.1*rlb)/(delta*self.rt)
		#self.Ms = Mt/(Mf)
		#self.vs = np.sqrt( (self.Ms/self.rs)*(0.0043*(np.pi*4.0/9.0)))
		###Scaled Parameters
		#self.rhohat_s = self.rhohat*(self.Ms)/(self.rs**3)
		#self.rhob_s = (self.rho_b/self.rho0)*(self.Ms)/(self.rs**3)
		#self.rhope_s = (self.rho_pe/self.rho0)*(self.Ms)/(self.rs**3)

		self.Mb =(4*np.pi*simps((self.r[self.phi>0]**2)*self.rho_b/self.rho0,self.r[self.phi>0]))#-M_bp# (simps(N,-phi))
		self.Mpe = 4*np.pi*simps((self.r[self.phi>0]**2)*self.rho_pe/self.rho0,self.r[self.phi>0])
		self.Mfrac = self.Mpe/(self.Mb+self.Mpe)

		
		#self.sigma_s = self.sigma*(self.vs)
		#self.N1_s = np.array(self.N1)*(self.Ms/(self.vs**2))
		#self.N2_s = np.array(self.N2)*(self.Ms/(self.vs**2))
		#self.phi_s = self.phi*self.vs**2
		#self.phii_s = self.phii*self.vs**2
		#self.r_s = self.r*self.rs
		#self.rt = self.rt*self.rs
		
		###Surface Density and projected velocity dispersion
		R = self.r[self.r<delta*self.rt]

		self.density = self.rhohat[self.r<delta*self.rt]
		self.dispersion = self.sigma[self.r<delta*self.rt]
		self.surface_density = np.zeros(len(self.density)-1)
		self.surface_dispersion = np.zeros(len(self.density)-1)
		self.rall = self.r
		self.phi = self.phi[self.r<delta*self.rt]
		self.r = self.r[self.r<delta*self.rt]
		#self.sigma_b = self.sigma_b[self.r<delta*self.rt]
		#self.rho_b = self.rho_b[self.r<delta*self.rt]
		#self.sigma_pe = self.sigma_pe[self.r<delta*self.rt]
		#self.rho_pe = self.rho_pe[self.r<delta*self.rt]
		
		for i in range(len(self.r)-1):
			c = (self.r >= R[i])
			r_2 = self.r[c]
			z = sqrt(abs(r_2**2 - R[i]**2)) # avoid small neg. values
			self.surface_density[i] = 2.0*abs(simps(self.density[c], x=z))
			self.surface_dispersion[i] = np.sqrt((abs(2.0*(simps(self.density[c]*(1.0/3.0)*self.dispersion[c]**2,x=z))))/self.surface_density[i])
		#print self.surface_dispersion
		#self.surface_density = self.surface_density*(self.Ms)/(self.rs**2)
		#self.surface_dispersion = self.surface_dispersion*(self.vs)
		self.R = self.r[0:len(self.r)-1]
	def _logcheck(self, t, y):
		""" Logs steps and checks for final values """
		#print y
		if (t>0)&(y[0]>-2*self.phi0): self.r, self._y = numpy.r_[self.r, t], numpy.c_[self._y, y]

		return 0 

	def _odes(self, x,y,args):
		derivs = [y[1]/x**2] if (x>0) else [0]
		derivs.append(-9.0*x**2*np.exp(y[0]))
		
		u = y[1]
		
		phi=y[0]
		phi0=args[0]
		eta=args[1]
		B = args[2]
	
		C = 1 - ((1-B)/eta**2)

		if phi>0:
			inf = gu(1.5,phi/eta**2)*np.exp(phi/eta**2)*g(1.5)
			if inf!=inf or inf==np.inf:
				inf = (phi/eta**2)**0.5
			inf2 = gu(1.5,phi0/eta**2)*np.exp(phi0/eta**2)*g(1.5)
			if inf2!=inf2 or inf2==np.inf:
				inf2 = (phi0/eta**2)**0.5

			dydt =[u/(x**2), -9*(x**2)*(self.mmb*(np.exp(phi)*g(1.5)*gl(1.5,phi)-(2.0/3.0)*phi**(1.5)*B- (4.0/15.0)*C*phi**(2.5)) + self.mmpe* ((1-B)*eta**3 *inf))/ (self.mmb*(np.exp(phi0)*g(1.5)*gl(1.5,phi0)-(2.0/3.0)*phi0**(1.5)*B- (4.0/15.0)*C*phi0**(2.5))+self.mmpe* ((1-B)*eta**3 *inf2))]

		if phi<0:
			inf_neg = np.exp(phi/eta**2)*g(1.5)
			inf2 = gu(1.5,phi0/eta**2)*np.exp(phi0/eta**2)*g(1.5)
			if inf2!=inf2 or inf2==np.inf:
				inf2 = (phi0/eta**2)**0.5
			dydt =[u/(x**2), -9*(x**2)*(self.mmpe*((1-B)*eta**3 *inf_neg))/(self.mmb*(np.exp(phi0)*g(1.5)*gl(1.5,phi0)-(2.0/3.0)*phi0**(1.5)*B- (4.0/15.0)*C*phi0**(2.5))+self.mmpe* ((1-B)*eta**3 *inf2))]
		return dydt
