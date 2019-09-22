#from limepy import limepy      # Our lowered isothermal model DF code
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
from limepy import limepy
import sys
from SPES_model import SPES

model = 1
mr = 0
lambd = 1
lr = 0
ss = 0.4

def read_final_chain(i):


	file_Name = "SPES_mcmc_output/samplerchain.pkl"

	fileObject = open(file_Name,'r')
	chain = pickle.load(fileObject)
	fileObject.close()
	lastchain = chain[:,-1,:]
	return lastchain

def read_final_chain_limepy(i):

	a = [0.8,0.6,0.4,0.2]
	a = i

	file_Name = "limepy_samplerchain.pkl"
	
	fileObject = open(file_Name,'r')
	chain = pickle.load(fileObject)
	fileObject.close()
	lastchain = chain[:,-1,:]
	return lastchain



name = ["pm","sis","pl"]
def poissonsolve(phi0,eta,Mt,rhm,B):
	eta = eta
	init = phi0,0.0

	C = 1 - ((1-B)/eta**2)
	if model==0:
		C=0

	params = [phi0,eta,Mt,rhm,B]


	inp = 1000
	
        m = SPES(inp,phi0,eta,B,1,0,1)
	r = m.r
	
	phi = m.phi
	rt = np.interp(0,-phi,r)
	
	while rt == inp:
		m = isothermal(inp*2,phi0,eta,B)
		r = m.r
		
		phi = m.phi
		rt = np.interp(0,-phi,r)
		inp = inp*2
	r2 = r
	
	if mr==0:
		r=r[phi>0]
		r2 = r2[phi>0]
		phi=phi[phi>0]
		
	if mr==1:
		phi=phi[r<1.5*rt]
		r2 = r2[r<1.5*rt]
		r=r[r<1.5*rt]
		

	inf1 = (np.exp(init[0]/(eta**2))*gu(1.5,init[0]/(eta**2)))*g(1.5)
		
	if (inf1==np.inf) or (inf1!=inf1):
		inf1=(init[0]/eta**2)**0.5
	
	inf = np.exp(phi/(eta**2))*gu(1.5,phi/(eta**2))*g(1.5)
	inf2 =np.exp(phi/(eta**2))*gu(2.5,phi/(eta**2))*g(2.5)
	for i in range(len(inf)):
		if (inf[i]==np.inf) or (inf[i]!=inf[i]):
			inf[i] = (phi[i]/eta**2)**0.5
			

	for i in range(len(inf2)):
		if (inf2[i]==np.inf) or (inf2[i]!=inf2[i]):
			inf2[i] =  (phi[i]/eta**2)**1.5
		

	density = np.zeros_like(phi)
	dispersion = np.zeros_like(phi)
	density_bound = np.zeros_like(phi)
	density_pe = np.zeros_like(phi)

	fe = [0]*len(phi[phi>0])
	fe_1 = [0]*len(phi[phi>0])
	fe_2 = [0]*len(phi[phi>0])
	phii = phi[phi<0]
	#print "len",len(fe)
	A = 1./(4*np.pi*np.sqrt(2)*(np.exp(init[0])*g(1.5)*gl(1.5,init[0])-(2.0/3.0)*init[0]**(1.5)*B) - (4.0/15.0)*C*init[0]**(2.5) + ((1-B)*eta**3 *inf1))
	for i in range(len(phi)):
		#print phi[i]
		if phi[i]>0:
    
			density[i] = ((np.exp(phi[i])*g(1.5)*gl(1.5,phi[i])-(2.0/3.0)*phi[i]**(1.5)*B)- (4.0/15.0)*C*phi[i]**(2.5) + ((1-B)*eta**3 *inf[i]))/((np.exp(init[0])*g(1.5)*gl(1.5,init[0])-(2.0/3.0)*init[0]**(1.5)*B) + ((1-B)*eta**3 *inf1))

			dispersion[i] = np.sqrt( 2*(((np.exp(phi[i])*g(2.5)*gl(2.5,(phi[i]))-0.4*phi[i]**(2.5)*B) - (4.0/35.0)*C*phi[i]**(3.5)+ ((1-B)*eta**5 *inf2[i])))/((np.exp(phi[i])*g(1.5)*gl(1.5,(phi[i]))-(2.0/3)*phi[i]**(1.5)*B) - (4.0/15.0)*C*phi[i]**(2.5)+ ((1-B)*eta**3 *inf[i])))
			density_bound[i] = ((np.exp(phi[i])*g(1.5)*gl(1.5,phi[i])-(2.0/3.0)*phi[i]**(1.5)*B) - (4.0/15.0)*C*phi[i]**(2.5) )/((np.exp(init[0])*g(1.5)*gl(1.5,init[0])-(2.0/3.0)*init[0]**(1.5)*B) - (4.0/15.0)*C*init[0]**(2.5)+ ((1-B)*eta**3 *inf1))
			density_pe[i]= (((1-B)*eta**3 *inf[i]))/((np.exp(init[0])*g(1.5)*gl(1.5,init[0])-(2.0/3.0)*init[0]**(1.5)*B) - (4.0/15.0)*C*init[0]**(2.5)+ ((1-B)*eta**3 *inf1))
			#print "phi", phi[i], i
			fe[i] = A*(np.exp(phi[i])-B-(C*phi[i]))
			fe_1[i] = A*(np.exp(phi[i])-C)
			fe_2[i] = A*(np.exp(phi[i]))
		if phi[i] <0:
			inf_neg = np.exp(phi[i]/eta**2)*g(1.5)
			inf2_neg = np.exp(phi[i]/(eta**2))*g(2.5)
			density[i] = (((1-B)*eta**3 *inf_neg))/((np.exp(init[0])*g(1.5)*gl(1.5,init[0])-(2.0/3.0)*init[0]**(1.5)*B) + ((1-B)*eta**3 *inf1))
			dispersion[i] = np.sqrt( 2*(( ((1-B)*eta**5 *inf2_neg)))/(+ ((1-B)*eta**3 *inf_neg)))


	rho0 = ((np.exp(init[0])*g(1.5)*gl(1.5,init[0])-(2.0/3.0)*init[0]**(1.5)*B) - (4.0/15.0)*C*init[0]**(2.5)+ ((1-B)*eta**3 *inf1))

	R = r
	surface_density = np.zeros(len(density))
	surface_dispersion = np.zeros(len(density))
	for i in range(len(r)):
		c = (r >= R[i])
		r_2 = r[c]
		z = sqrt(abs(r_2**2 - R[i]**2)) # avoid small neg. values
		surface_density[i] = 2.0*abs(simps(density[c], x=z))
		surface_dispersion[i] = np.sqrt((abs(2.0*(simps(density[c]*(1.0/3.0)*dispersion[c]**2,x=z))))/surface_density[i])

	N = [0]*len(phi[phi>0])
	phii = np.linspace(0,-init[0],len(phi))#solve[:,0][solve[:,0]<0]
	N2=[0]*len(phii)
	for i in range(len(phi[phi>0])):
		N[i] = ((4*np.pi)*(np.exp(phi[i])-B-(C*phi[i]))*simps((r[0:(i+1)]**2 * np.sqrt((-phi[i] + phi[0:(i+1)]))),r[0:(i+1)])/(np.exp(init[0])*gl(1.5,init[0])*g(1.5)-((B*2.0/3)*init[0]**1.5)- (4.0/15.0)*C*init[0]**(2.5)+ ((1-B)*eta**3 *inf1)))
		#N[i] = ((4*np.pi)*(np.exp(phi[i])-B)*simps((r[0:(i+1)]**2 * np.sqrt((-phi[i] + phi[0:(i+1)]))),r[0:(i+1)])/(np.exp(init[0])*gl(1.5,init[0])*g(1.5)-((B*2.0/3)*init[0]**1.5)+ ((1-B)*eta**3 *inf1)))
	for i in range(len(phii)):
		N2[i] =((1-B) * (4*np.pi/np.sqrt(2))*np.exp(phii[i]/eta**2)*simps(r[phi>0]**2 * np.sqrt(2*(-phii[i] + phi[phi>0])),r[phi>0])/(np.exp(init[0])*gl(1.5,init[0])*g(1.5)- (4.0/15.0)*C*init[0]**(2.5)+ ((1-B)*eta**3 *inf1)))

	fe2=[0]*len(phii)
	fe2_1=[0]*len(phii)
	fe2_2=[0]*len(phii)
	for i in range(len(phii)):
		fe2[i] = A*(1-B)*(np.exp(phii[i]/eta**2))
		fe2_1[i] = A*(1-B)*(np.exp(phii[i]/eta**2))/eta**2
		fe2_2[i] = A*(1-B)*(np.exp(phii[i]/eta**2))/eta**4
	
	count = 0
	Mss = []
	if mr==0:
		Mf = (4*np.pi*simps((r[0:len(phi[phi>0])]**2)*density[0:len(phi[phi>0])],r[0:len(phi[phi>0])]))
	if mr==1:
		Mf = (4*np.pi*simps((r[0:len(phi)]**2)*density[0:len(phi)],r[0:len(phi)]))
		Mf2 = (4*np.pi*simps((r[0:len(phi[phi>0])]**2)*density[0:len(phi[phi>0])],r[0:len(phi[phi>0])]))
		
	for i in range(len(phi)-1):
		Mc =(4*np.pi*simps((r[0:i+1]**2)*density[0:i+1],r[0:i+1]))
		if (Mc > 0.5*Mf) & (count==0):
			count = 1
			rs = rhm/r[i]
			break
	#print rs
	Ms = Mt/(Mf)
	#print (Mf-Mf2)*Ms
	vs = np.sqrt( (Ms/rs)*(np.pi*4.0/9.0))

	#print rs,Ms,vs
	density_all = np.array(density)*(Ms)/(rs**3)
	density_pe = np.array(density_pe)*(Ms)/(rs**3)
	density_bound = np.array(density_bound)*(Ms)/(rs**3)
	disp_all =  np.array(dispersion)*(vs)


	
	N = np.array(N)*(Ms/(vs**2))
	N2 = np.array(N2)*(Ms/(vs**2))
	phi_bp = (0.5*(4*np.pi/9.0)*Mf/rt)*vs**2
	phi = phi*vs**2
	phii = phii*vs**2
	rt = rt*rs
	r = r*rs

	#print r2.max(), rs, r2.max()*rs
	r2 = r2*rs
	
	return density_all,disp_all,r[phi>0],N,N2,phi,phii,density,dispersion,density_bound,density_pe,r2,phi_bp,fe_2,fe2_2,rt

# MAIN
for imf in range(1):
	imf = 1
	aa = [80,60,40,20]#[80,60,40,20]
	output =[]
	output2=[]
	
	ff4, ((ax13)) = plt.subplots(1,1)
	
	f3=open('table_%i%i%i%i.dat'%(model,mr,lambd,lr),'w')
	rjs = []
	rjs_m =[]
	r_lb = []
	Ms = []
	Mbs = []
	Mpes = []
	Mbs_adj = []
	Mpes_adj = []
	Mbs_sim = []
	Mpes_sim = []
	Ms_sim = []
	Bs = []
	Cs=[]
	etas=[]
	phi0s=[]
	Mcs_array= [0]*4
	rhs_array= [0]*4
	rjs_array = [0]*4
	fpe_array = [0]*4

	for j in range(1):
		chain = read_final_chain(ss)
		limepy_chain = read_final_chain_limepy(ss)
		Nmod =50
		M_cs = 0
		r_hs=0
		M_pe = 0
		M_b = 0
		M_pe_adj = 0
		M_b_adj = 0
		r_j = 0
		r_j_l=0
		r_lim=0
		notin=0
		print j
 
		r_disps = [0]*Nmod
		r1_disps = [0]*Nmod
		disps =[0]*Nmod
		sbs = [0]*Nmod
		sbs_bound = [0]*Nmod
		sbs_pe = [0]*Nmod
		sb_min = 1

		disps_l = [[]]*Nmod
		sbs_l = [[]]*Nmod
		disps_interp_l = [[]]*Nmod
		sbs_interp_l = [[]]*Nmod
		r_disps_l = [[]]*Nmod
		for i in range(Nmod):
			print i
			values = [[11.5, 0.25, 0.4 , 0.7867, 0.964],[10.18,0.298,0.394,0.811,0.883]]
			values = [8.0, 0.2024  ,0.8 , 0.951064,  0.9]

			W0_l, M_l, rh_l, g_l = limepy_chain[i,:]
			m = limepy(W0_l, g_l, G=1, M=M_l, rh=rh_l,verbose=False)

			disps_l[i] = numpy.sqrt(m.v2)
			sbs_l[i] = m.rho
			r_disps_l[i] = m.r
			r_j_l += m.r.max()
			W0, eta, Mtot, rh, B  =  chain[i,:]
			
			density_all,disp_all,r1,N,N2,phi,phii,density,dispersion,density_bound,density_pe,r,phi_bp,fe2,fe2_2,rt = poissonsolve(W0,eta,Mtot,rh,B)

			
			r_j += rt
			if (i>0): 
				if (np.interp(0,-phi,r)>1.5*r_j/i):
					r_j -= np.interp(0,-phi,r)
					notin += 1
			r_lim += r.max()

			
			M = (4*np.pi*simps((r1**2)*density_all[phi>0],r1))
			Menc = (4*np.pi*simps((r**2)*density_all,r))
			s2 = ((M/r1.max())-(Menc/r)[phi>0])/phi[phi>0]
			M_bp = 4*np.pi*simps(r[(phi>0)&(phi<phi_bp)]**2*density_all[(phi>0)&(phi<phi_bp)],r[(phi>0)&(phi<phi_bp)])
			phi_in = phi[phi>0]
			phi_out = phi[phi<0]
			phi_bp = M/(2*r1.max())
			M_bp = simps(N[phi_in<(phi_bp)],-phi_in[phi_in<(phi_bp)])


			M_cs += M
			r_hs += rh
			M_b += (4*np.pi*simps((r1**2)*density_bound[phi>0],r1))#-M_bp# (simps(N,-phi))
			M_pe += 4*np.pi*simps((r1**2)*density_pe[phi>0],r1)#+M_bp#
			M_b_adj += (4*np.pi*simps((r1**2)*density_bound[phi>0],r1))-M_bp# (simps(N,-phi))
			M_pe_adj += 4*np.pi*simps((r1**2)*density_pe[phi>0],r1)+M_bp#

			r_disps[i] = r
			disps[i] = disp_all
			sbs[i] = density_all
			r1_disps[i] = r1
			sbs_bound[i] = density_bound
			sbs_pe[i] = density_pe
			if sb_min>density_all[phi>0].min():
				sb_min = density_all[phi>0].min()

			
				
		
		rho_pe_dat = numpy.loadtxt('data/rho_pe_%i_equal.dat'%(ss*100)).T
		Mpe = M_pe/Nmod
		Mb = M_b/Nmod
		Mpe_adj = M_pe_adj/Nmod
		Mb_adj = M_b_adj/Nmod
		r_j = r_j/(Nmod-notin)
	

		r_lim = rho_pe_dat[0].max()
		
		disps_interp =[0]*Nmod
		sbs_interp = [0]*Nmod
		sb_bound_interp = [0]*Nmod
		sb_pe_interp = [0]*Nmod
		if mr==0:
			r_interp = np.linspace(0,(rt),101)
			print "rt", rt
			r_log_interp = np.logspace(-2,np.log10(rt),101)
		if mr==1:
			r_interp = np.linspace(0,(1.5*rt),101)
			print "rt", rt
			r_log_interp = np.logspace(-2,np.log10(1.5*rt),101)
		for i in range(len(disps)):

			sb_interp = np.logspace(-1,-5,101)
			disps_interp[i] = np.interp(r_log_interp,r_disps[i],disps[i])
			sbs_interp[i] = np.interp(r_log_interp,r_disps[i],sbs[i])
			sb_bound_interp[i] = np.interp(-sb_interp,-sbs_bound[i],r_disps[i])
			sb_pe_interp[i] = np.interp(-sb_interp,-sbs_pe[i],r_disps[i])
			
		disp_percs = np.percentile(disps_interp,[16,50,84],axis=0)
		sb_percs = np.percentile(sbs_interp,[16,50,84],axis=0)
		sb_bound_percs = np.percentile(sb_bound_interp,[16,50,84],axis=0)
		sb_pe_percs = np.percentile(sb_pe_interp,[16,50,84],axis=0)

		r_j_l = r_j_l/Nmod
		for i in range(len(disps)):
			r_interp_l = np.logspace(-2,np.log10(r_j_l),101)
			r_log_interp_l = np.logspace(-2,np.log10(r_j_l),101)
			disps_interp_l[i] = np.interp(r_interp_l,r_disps_l[i],disps_l[i])
			sbs_interp_l[i] = np.interp(r_log_interp_l,r_disps_l[i],sbs_l[i])
		disp_percs_l = np.percentile(disps_interp_l,[16,50,84],axis=0)
		sb_percs_l = np.percentile(sbs_interp_l,[16,50,84],axis=0)



		
		aa = [ss*100]
		rho_dat = numpy.loadtxt('data/rho_%i_equal.dat'%(aa[j])).T
		sigma_dat = numpy.loadtxt('data/sigma_%i_equal.dat'%(aa[j])).T
		rho_b_dat = numpy.loadtxt('data/rho_b_%i_equal.dat'%(aa[j])).T
		sigma_b_dat = numpy.loadtxt('data/sigma_b_%i_equal.dat'%(aa[j])).T
		rho_pe_dat = numpy.loadtxt('data/rho_pe_%i_equal.dat'%(aa[j])).T
		sigma_pe_dat = numpy.loadtxt('data/sigma_pe_%i_equal.dat'%(aa[j])).T
		Ej_dat = numpy.loadtxt('data/E_%i_equal.dat'%(aa[j]))
		Ej_pe_dat = numpy.loadtxt('data/E_pe_%i_equal.dat'%(aa[j]))
		values = np.loadtxt('data/equalmass_32')
		#f = open('BC_pm/%.1f/quantiles.txt'%(aa[j]*0.01),'r')
		f = open('SPES_mcmc_output/quantiles.txt','r')

		
                #j = 3
		M_sim, Mb_sim, Mpe_sim, rh_sim, rj_sim = values[3] 
		Mc = values[3][0]
		rh = values[3][3]
		rj = values[3][4]
		#Mc = values[7-(j)][0]
		#rj = values[7-(j)][4]
		yEh = np.linspace(-5,1,241)
		Mcs_array[j]=M_cs/(Nmod*Mc)
		rhs_array[j]=r_hs/(Nmod*rh)
		rjs_array[j]=r_j/(rj)
		fpe_array[j] =(Mpe*Nmod/M_cs)/(Mpe_sim/M_sim)
		weights = [(1/(16384*0.025))]*len(Ej_dat)
		weights2 = [(1/(16384*0.025))]*len(Ej_pe_dat)
		lbl = '${M_{\\rm c}=%.4f, r_{\\rm hm}=%.4f}$'%(values[3][0],values[3][3])
		#lbl = '$\mathrm{M_{c}=%.4f, r_{hm}=%.4f}$'%(values[7-(j)][0],values[7-(j)][3])
		
	
		a = []
		for line in f:
			line = line.split()
			a.append(np.array(line))

		phi_t = float(a[1][1])
		eta_t = float(a[2][1])
		M_t = float(a[3][1])
		rh_t = float(a[4][1])
		B_t = float(a[6][1])


		etas.append(eta_t)
		Bs.append(B_t)
		phi0s.append(phi_t)
		if model==1:
			Cs.append(1- ((1-B_t)/eta_t**2))
		if model==0:
			Cs.append(0)
	
		#ax14.plot(phi_in,fe2nd)
		#ax14.plot(phii,fe2nd_pe)
		#print "Simulation;",values[7-(2*j)][0:4]
		#print "Quantiles;",M_t,Mb,Mpe,rh_t
		output.append(values[7-(2*j)][0:4])
		output2.append(np.array([M_t,Mb,Mpe,rh_t]))
		lbl = '$\mathrm{ss3:} M_{\\rm c}=%.3f, r_{\\rm J}=%.2f}$'%(values[3][0],values[3][4])
		#lbl= '$\mathrm{M_{c}=%.3f, r_{hm}=%.3f}$'%(values[7-(2*j)][0])
		for i in range(len(rho_dat[2])):
			if rho_dat[1][i] - rho_dat[2][i] < 0:
				rho_dat[2][i] = rho_dat[1][i]-0.0000000001
		
		
		if j ==0:
			ax13.errorbar(sigma_dat[0]/rj, sigma_dat[1],yerr= sigma_dat[2],fmt='mo',label=lbl)
			ax13.plot(r_interp_l/rj,disp_percs_l[1],'k-',lw=0.5,label='$\mathrm{LIMEPY}$: $M_{\\rm c}=0.399, r_{\\rm t}=%.2f$'%r_j_l)
			ax13.fill_between(r_interp_l/rj,disp_percs_l[0],disp_percs_l[2],facecolor='grey',alpha=0.5)#,label='LIMEPY model: $M_{\\rm c}=0.399, r_{\\rm hm}=0.790$')			
	
			
			ax13.set_ylabel('$\sigma$ $\\rm [N-body]$',fontsize=15)
			ax13.set_xlabel('$r/r_{\\rm J}$',fontsize=20)
			
			
			ax13.plot(r_log_interp/rj,disp_percs[1],'g-',lw=1,label='$\mathrm{SPES}$: $M_{\\rm c}=0.399, r_{\\rm crit}=%.2f$'%r_j)		
			ax13.fill_between(r_log_interp/rj,disp_percs[0],disp_percs[2],facecolor='green',alpha=0.5)	
			ax13.plot([r_j/rj]*101,np.linspace(0,0.7,101),'g--')	
			ax13.plot([r_j_l/rj]*101,np.linspace(0,0.7,101),'k--')	
			

		M_sim, Mb_sim, Mpe_sim, rh_sim, rj_sim = values[7-(2*j)] 

		f3.write("Sim ss%i & - & - & - & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f\n"%(j+1,M_sim,rh_sim, rj_sim, Mb_sim, Mpe_sim, Mpe_sim/M_sim,rho_dat[0].max()))
		f3.write("Model ss%i & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & - \n"%(j+1, W0, eta, B, M_t, rh_t, r_j, Mb,Mpe, Mpe/M_t))
		rjs.append(rj_sim)	
		rjs_m.append(r_j)
		r_lb.append(rho_dat[0].max())
		Ms.append(M_sim)
		Mbs.append(Mb)
		Mpes.append(Mpe)
		Mbs_adj.append(Mb_adj)
		Mpes_adj.append(Mpe_adj)
		Mbs_sim.append(Mb_sim)
		Mpes_sim.append(Mpe_sim)
		Ms_sim.append(M_t)
		#plt.show()
	f3.close()
	
	ax13.legend(numpoints=1,prop={'size':12},frameon=False)
	
	ax13.set_xlim([0,2])

	savestring4="BC_limepy.pdf"

	ff4.set_size_inches(5, 4)
	ff4.savefig(savestring4,bbox_inches='tight')



