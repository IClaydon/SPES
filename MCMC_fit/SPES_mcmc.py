#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This code fits parameters with EMCEE

# These top 2 lines are to ensure that plots can be created when on 
# the server without X11 support
import matplotlib     
matplotlib.use('Agg')                

import emcee                   # The MCMC code, download from: https://github.com/dfm/emcee
from limepy import limepy      # Our lowered isothermal model DF code
import corner as triangle      # This module can be found here: https://pypi.python.org/pypi/triangle_plot
from scipy.integrate import simps
import numpy as np
import numpy 
import matplotlib.pyplot as pl
import argparse, sys
from matplotlib.ticker import MaxNLocator
from pylab import sqrt, log, log10, pi, sum, mean, interp, sort, arange, tan, exp
from pylab import exp, sqrt, log, cos,pi
from scipy.integrate import odeint,ode
from scipy.integrate import quad
from scipy.integrate import simps
from scipy.special import gammainc as gl
from scipy.special import gammaincc as gu
from scipy.special import gamma as g
from scipy.special import gammainc 
from scipy.special import gammaincc 
from scipy.special import gamma
from scipy.optimize import fmin
import sys
import pickle
from shutil import copyfile
from SPES_model import SPES

restart = 0 # restarts from previous samplechain.pkl if restart=1
ss = 0.4 # set which snapshot to use (fraction of mass remaining)

if restart>0:
	
	copyfile("SPES_mcmc_output/samplerchain.pkl","SPES_mcmc_output/restart.pkl")


def read_final_chain():
    # This routine allows restart from a previous run
    file_Name = "SPES_mcmc_output/restart.pkl"
    fileObject = open(file_Name,'r')
    chain = pickle.load(fileObject)
    fileObject.close()
    lastchain = chain[:,-1,:]
    lnprob = numpy.loadtxt('SPES_mcmc_output/lnprob.txt')
    #print len(lastchain)
    #for i in range(len(lnprob)):

		#chain[:,-1,:][i] = chain[:,-1,:][lnprob.argmax()] + 1e-2*chain[:,-1,:][lnprob.argmax()]*numpy.random.randn(5)
		
    lastchain2 = chain[:,-1,:]
    return numpy.r_[lastchain2]


# Define the probability function as likelihood * prior.
def lnprior(theta):
    W0, eta, M,rh,B = theta
    C = 1 - ((1-B)/eta**2)
    if  1 < W0 < 14. and 0.01 < eta < 1  and 0.01 < M < 1 and 0<B<1 and -100<C<1 and 0<rh<10:
        return 0
    return -np.inf

ssi = [10,16]
def lnlike_ian(theta,data):
	#print theta
	W0, eta, Mt, rh, B = theta
	#print theta
	rho_dat = data[0]
	sigma_dat = data[1]
	#discrete_dat = data[2]

	#try:
	eta = eta
	G = np.pi*4.0/9.0
	init = W0,0.0
	C = 1 - ((1-B)/eta**2)

	params = [W0,eta,Mt,rh]

	if restart==0:
		m = SPES(1000,W0,eta,B,1,0,1)
	if restart==1:
		m = SPES(1000,W0,eta,B,1,1,1)
	r = m.r
	phi = m.phi

	minim = 0

	rt = m.rt	
	dispersion = m.dispersion
	density = m.density

	

	
	count = 0
	Mss = []
	Mf = (4*np.pi*np.trapz((r**2)*density,r))

	Mcs=[]
	for l in range(len(phi)-1):
		Mc=(4*np.pi*np.trapz((r[0:l+1]**2)*density[0:l+1],r[0:l+1]))
		if (Mc > 0.5*Mf) & (count==0):
			count = 1
			rs = rh/r[l]
			break

	Ms = Mt/(Mf)
	vs = np.sqrt( (Ms/rs)*(np.pi*4.0/9.0))

	density_all = np.array(density)*(Ms)/(rs**3)
	disp_all =  np.array(dispersion)*(vs)

	r_us = r
	r = r*rs

	
	prob = 0
	inf1 = (np.exp(W0/(eta**2))*gu(1.5,W0/(eta**2)))*g(1.5)
	
	A =(Ms/((vs**3)*(rs**3)))* 1./(4*np.pi*np.sqrt(2)*(np.exp(W0)*g(1.5)*gl(1.5,W0)-(2.0/3.0)*W0**(1.5)*B) - (4.0/15.0)*C*W0**(2.5) + ((1-B)*eta**3 *inf1))
	
	values = np.loadtxt('data/equalmass_32')
	
	sigma_mod_interp = np.interp(sigma_dat[0],r,disp_all,right=-np.inf)
	rho_mod_interp = np.interp(rho_dat[0],r,density_all,right=-np.inf)
	loglike = 0
	for i in range(len(rho_dat[0])):
	
			loglike -= (rho_mod_interp[i] - rho_dat[1][i])**2/(2*(rho_dat[2][i]**2))
			loglike -= log(sqrt(2*pi*rho_dat[2][i]**2))
		
	# Likelihood: velocity dispersion
	for i in range(len(sigma_dat[0])):
		
			loglike -= (sigma_mod_interp[i] - sigma_dat[1][i])**2/(2*(sigma_dat[2][i]**2))
			
			loglike -= log(sqrt(2*pi*sigma_dat[2][i]**2))

	if numpy.isnan(loglike) :
		print "nan"
		loglike = -np.inf
	return (loglike)
		



def lnprob_ian(theta, data):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf

    lnlikeval = lnlike_ian(theta, data)
    return lp + lnlikeval

# Read data
def get_data():
    # Personalise this: data is assumed to be m, ra, dec
    surface_brightness_file = 'data/rho_%i_equal.dat'%(ss*100)
    radial_velocity_file   = 'data/sigma_%i_equal.dat'%(ss*100)
    #phasespace_file   = 'data/%i.%i/discrete_%i.dat'%(lambd,lr,ss*100)
    
    surface_brightness_dat  = np.loadtxt(surface_brightness_file,skiprows=0).T
    radial_velocity_dat    = np.loadtxt(radial_velocity_file, skiprows=0).T
    #phasespace_dat = np.loadtxt(phasespace_file).T

    data = [surface_brightness_dat, radial_velocity_dat]#,phasespace_dat] 

    return data

def run_mcmc(data,ndim, nwalkers, burnin, nsteps, nout, nthreads, true,number):
    # Reproducible results!
    np.random.seed(123)
    number += 1
    print number
    #W0, eta, M, rh = theta

    # Set up the sampler.

    guess = [11,0.25,ss,1.05,0.95]


    # Initial positions of the walkers in parameter space

    pos = [guess + 1e-2*np.array(guess)*np.random.randn(ndim) for i in range(nwalkers)]

    # Or alternative restart from previous run
    if restart>0:
    	pos = read_final_chain()

    # Note that number of threads can be used: nwalkers/threads should be integer
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_ian, args=([data]), threads=nthreads)

    # Clear and run the production chain.
    print("Running MCMC...") 

    state=None

    while sampler.iterations < nsteps:
#        print " TEST ",sampler.iterations
        pos, lnp, state = sampler.run_mcmc(pos, nout, rstate0=state)
        output=" %4i %12.5e %12.5e %12.5e %12.5e"%(sampler.iterations, mean(pos[-nout:-1,0]), mean(pos[-nout:-1,1]), mean(pos[-nout:-1,2]), mean(lnp[:]))

        if sampler.iterations % nout == 0:
            make_plots(sampler, true, ndim, 2,lnp,pos)
            save_chain(sampler)
        print output
        """if (lnp_old/mean(lnp[:])>0.999) and (mean(lnp[:])==mean(lnp[:])):
            break
        else:
            lnp_old = mean(lnp[:])"""
    return sampler

def save_chain(sampler):
    file_Name = "SPES_mcmc_output/samplerchain.pkl"
    samples=sampler.chain
    fileObject = open(file_Name,'wb')
    pickle.dump(samples,fileObject)
    fileObject.close()

def make_plots_end(sampler, true, ndim, burnin):
	pl.clf()

	for i in range(ndim):
		true[i] = np.mean(sampler.chain[:,burnin:,i])

	fig, axes = pl.subplots(ndim, 1, sharex=True, figsize=(8, 9))
	label = ['$W_0$','$\eta$','$M$','$rh$','$B$'] 
	for i in range(ndim):
		axes[i].plot(sampler.chain[:, :, i].T, color="k", alpha=0.4)
		axes[i].yaxis.set_major_locator(MaxNLocator(5))
		axes[i].axhline(true[i], color="#888888", lw=2)
		axes[i].set_ylabel(label[i])

	fig.tight_layout(h_pad=0.0)
	fig.savefig("SPES_mcmc_output/line-time-end.png")

	# Make the triangle plot.
	samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))

	fig = triangle.corner(samples, labels=label,
		                  truths=true)
	fig.savefig("SPES_mcmc_output/line-triangle-end.png")

def make_plots(sampler, true, ndim, burnin,lnp,pos):
	pl.clf()

	for i in range(ndim):
		true[i] = np.mean(sampler.chain[:,burnin:,i])

	fig, axes = pl.subplots(ndim, 1, sharex=True, figsize=(8, 9))
	label = ['$W_0$','$\eta$','$M$','$rh$','$B$'] 
	for i in range(ndim):
		axes[i].plot(sampler.chain[:, :, i].T, color="k", alpha=0.4)
		axes[i].yaxis.set_major_locator(MaxNLocator(5))
		axes[i].axhline(true[i], color="#888888", lw=2)
		axes[i].set_ylabel(label[i])

	fig.tight_layout(h_pad=0.0)
	fig.savefig("SPES_mcmc_output/line-time.png")

	# Make the triangle plot.
	samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
	fig = triangle.corner(samples, labels=label,
		                  truths=true)
	fig.savefig("SPES_mcmc_output/line-triangle.png")
	abc = -500
	
	fig = pl.figure()
	pl.plot(pos[abc:-1,1],lnp[abc:-1],'k.')
	fig.savefig("SPES_mcmc_output/eta-likelihood.png")
	fig = pl.figure()
	pl.plot(pos[abc:-1,0],lnp[abc:-1],'k.')
	fig.savefig("SPES_mcmc_output/W0-likelihood.png")
	fig = pl.figure()
	pl.plot(pos[abc:-1,2],lnp[abc:-1],'k.')
	fig.savefig("SPES_mcmc_output/M-likelihood.png")
	fig = pl.figure()
	pl.plot(pos[abc:-1,3],lnp[abc:-1],'k.')
	fig.savefig("SPES_mcmc_output/rh-likelihood.png")
        fig = pl.figure()
	pl.plot(pos[abc:-1,4],lnp[abc:-1],'k.')
	fig.savefig("SPES_mcmc_output/B-likelihood.png")


	np.savetxt("SPES_mcmc_output/lnprob.txt",lnp[abc:-1])


# Compute the quantiles.

	W0_mcmc,  eta_mcmc, M_mcmc, rh_mcmc, B_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
		                                    zip(*np.percentile(samples, [16, 50, 84],
		                                                       axis=0)))
	f = open("SPES_mcmc_output/quantiles.txt",'w')
	f.write("""                                                                                                                                                                             
	W0  {0[0]} +{0[1]} -{0[2]} {1}                                                                                                                                                      
	eta   {2[0]} +{2[1]} -{2[2]} {3}                                                                                                                                                      
	M  {4[0]} +{4[1]} -{4[2]}{5}                                                                                                                                                      
	rh  {6[0]} +{6[1]} -{6[2]}
{7}
	B  {8[0]} +{8[1]} -{8[2]}
{9}


		                                                                                                                                                                                                                                                                                                    
	""".format(W0_mcmc, true[0], eta_mcmc, true[1], M_mcmc, true[2], rh_mcmc, true[3],B_mcmc,true[4]))
	f.close()


def main():
    # ndim     = number of parameters
    # nwalkers = number of walkers
    # burnin   = number of burnin in steps
    # nsteps   = total number of steps
    # nout     = output every nout
    # nthreads = number of parallel threads
    length=500
    if restart>0:
        length=800

    ndim, nwalkers, burnin, nsteps, nout, nthreads = 5, 100, 1, length, 5, 10
    number = 0
    data = get_data()
    
    true = np.zeros(ndim)

    sampler = run_mcmc(data, ndim, nwalkers, burnin, nsteps, nout, nthreads, true,number)

    make_plots_end(sampler, true, ndim, burnin)

if __name__ == '__main__':
	
	main() 
