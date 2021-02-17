# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 11:16:54 2020

@author: heerburk

OLG_simple_VI.py 

simple value function iteration for the optimization paper 
with Alfred Maussner

you may need to adjust the load path for the excel file below (line 20)

"""

# Part 1: import libraries
import pandas as pd
import numpy as np
import quantecon as qe
from scipy.stats import norm
import time
import math

# abbreviations
exp = np.e
log = math.log

# Part 2: Import data
data = pd.read_excel (r'C:\Users\heerburk\Documents\papers\optimization\python\survival_probs.xlsx') 
df = pd.DataFrame(data, columns= ['sp1','ef'])
#print(df)
arr = np.array(df)
#print(arr)
sp1 = arr[:,0]
ef = arr[0:45,1]
#print(ef)

# Part 3: Parameterization of the model
# demographics
nage=70                # maximum age                  
nw=45                  # number of working years        
Rage=46                # first period of retirement 
nr=nage-Rage+1         # number of retirement years
popgrowth0 = 0.0075400000 
# preferences
b=1.011             # discount factor 
sigma=2.0           # coefficient of relative risk aversion 
hbar=0.3            # fixed shift length 
mu0=123             # disutility of labor
mu1=0.5             # Frisch elasticity of labor supply
averagehours=0.3    # average number of working hours to calibrate mu0

# production
r=0.04              # initial value of the interest rate (in order to compute kbar)
alp=0.36            # production elasticity of capital 
delta=0.06          # rate of depreciation 

# productivity of workers
lamb0=.96           # autoregressive parameter 
sigmay1=0.38        # variance for 20-year old, log earnings */
sigmae=0.045        # earnings disturbance term variance */
ny=5                # number of productivity types
m=2                 # width of the productivity grid

# parameters of the algorighm
psi1=0.0001         # parameter of utility function for zero or negative consumption
update_agg=0.8      # parameter for the update of initial values
nq=50               # maximum number of iterations to find a solution

tol=0.001           # percentage deviation of final solution 

# grids
kmin=0              # inidividual wealth
kmax=20             # upper limit of capital grid 
na=100              # number of grid points over assets a in [kmin,kmax]
a =  np.linspace(kmin, kmax, na)   # asset grid 
emin=0              # individual cumulated earnings
emax=3
nce=20              # number of grid points on cumulated earnings
e =  np.linspace(emin, emax, nce)   # grid over cumulated earnings
labormin=0
labormax=0.7        # maximum labor supply
nlabor=100
lgrid= np.linspace(labormin, labormax, nlabor)    # grid over labor supply
ymin=0
ymax=4
nin=100
incomegrid = np.linspace(ymin, ymax, nin)        # grid for gross income

# measure of living persons
mass = np.ones(nage)

for i in range(nage-1):
    print(i)
    mass[i+1]=mass[i]*sp1[i]/(1+popgrowth0)

mass = mass / mass.sum()

# compute productivity grid
sy = np.sqrt(sigmae/(1-lamb0**2))    # earnings variance 
x0=1   
nsim=10
ye =  np.linspace(-m*sy, m*sy, ny)
ye1 = exp**ye

# transition matrix using Tauchen's approximation
# return is a class 'Markov Chain'
mc = qe.markov.approximation.tauchen(lamb0, sigmae, 0.0,  m, ny)
# transition matrix is stored in object 'PÄ
py = mc.P

# mass of the workers
muy = np.zeros((nw,ny))
w = ye[1]-ye[0]

# first year mass distribution
muy[0,0] = norm.cdf( (ye[0]+w/2)/np.sqrt(sigmay1) ) * mass[0]
muy[0,ny-1] = 1-norm.cdf( (ye[ny-1]-w/2)/np.sqrt(sigmay1))
muy[0,ny-1] = muy[0,ny-1]*mass[0]

for i in range(1,ny-1):
    print(i)
    muy[0,i] = ( norm.cdf( (ye[i]+w/2)/np.sqrt(sigmay1) ) - 
        norm.cdf( (ye[i]-w/2)/np.sqrt(sigmay1) ) )
    muy[0,i] = muy[0,i]*mass[0]


for i in range (1,nw):
    print(i)
    muy[i,:] = muy[i-1,:] @ py * sp1[i] / (1+popgrowth0)

# initial guess for aggregate effective labor supply
temp = np.transpose(muy*ye1) * ef
nbar = temp.sum() * hbar / sum(mass[0:nw])

xmat = np.zeros((4,nq))     # stores initial values at each iteration q

# initial guesses
kbar = (alp/(r+delta))**(1/(1-alp))*nbar
wage = (1-alp)*kbar**alp * nbar**(-alp)
tau = 0.068/(1-alp)      # pensions in the US are 6.8 of GDP in 2012, since wage income is (1-alpha)*GDP, 6.8%/(1-alp) provides an initial guess
tau = 0.124              # social security tax in the US 2020 
trbar = np.mean(1-sp1[0:nage-1]) * kbar        # proxy of accidental bequests

    
# part 4: definition of the functions
#
# wage function
def wage_rate(k,l):
    return (1-alp) * k**alp * l**(-alp)

# interest rate function
def interest_rate(k,l):
    return alp * k**(alp - 1) * l**(1-alp) - delta


# utility function 
# I changed this: only if c.le.0, a small amout is added @
def u(x,y): 
    x1 = np.where(x<=0, psi1, x)   # replace negative values by psi1  
    if sigma==1:
        return  log(x1) - mu0*y**(1+1/mu1) / (1+1/mu1)
    else:
        return x1**(1-sigma) / (1-sigma) -mu0*y**(1+1/mu1) / (1+1/mu1)

# pension schedule
 # see Huggett and Parra (2010, JPE)
def pension(x):   
    bendpoint1 = 0.20*meanincome
    bendpoint2 = 1.24*meanincome
    bmin = penmin
    
    
    if x<bendpoint1:
        y = bmin+0.9*x
    elif x>=bendpoint1 and x<=bendpoint2:
        y = bmin + 0.9*bendpoint1 + 0.32*(x-bendpoint1)
    else:
        y = bmin+ 0.9*bendpoint1 + 0.32*(bendpoint2-bendpoint1) + 0.15*(x-bendpoint2)
    
    return y

# computes the gini for distribution where x has measure g(x) 
def ginid(x,gy,ng):
    x =  np.where(x<=0, 0, x)
    xmean = np.dot(x,gy)
#    y = np.c_[x,gy]
    # sorting according to first column still has to be implemented
#    y0 = np.sort(y,axis=0)
#    x = y0[:,0]
#    g = y0[:,1]
    g = gy
    f = np.zeros(ng)  # accumulated frequency */
    f[0] = g[0]*x[0]/xmean
    gini = 1- f[0] * g[0]
    for i in range(1,ng):
        f[i] = f[i-1] + g[i]*x[i]/xmean
        gini = gini - (f[i]+f[i-1])*g[i]
	
    return gini

start_time = time.time()

# part 5: main program
# outer loop
q=0
crit = 1+tol
while q<nq and crit>tol:
    print("q: " + str(q))
    xmat[0,q] = kbar
    xmat[1,q] = nbar
    xmat[2,q] = tau
    xmat[3,q] = trbar


    # compute variables derived from (kbar,nbar,tau,trbar) 
    # required for the computation of decision rules    
    wage = wage_rate(kbar, nbar)
    rate = interest_rate(kbar, nbar)
    ybar=kbar**(alp) * nbar**(1-alp)    # produciton
    meanincome=wage*nbar / np.sum(mass[0:nw])  # average earnings of the workers
    penmin=0.1242*ybar                  # minimum pension
    
    start_time1 = time.time()   # time to compute decision rules
    
    # initialization of agents' value function: retired households  

    vr = np.zeros((na*nce,nr))      # value function 
    aropt = np.zeros((na*nce,nr))   # optimal asset 
    cropt = np.zeros((na*nce,nr))   # optimal consumption 

    k0 = 0                   # initialization of workers
    k1 = 0                   # current assets and cumulated earnings,
    e0 = 0                   # next-period assets and 
    e1 = 0                   # cumulated earnings


    # Initialization in the last period of the life: household consumes
    #  all his wealth and pension: no bequests
    #    
    for l in range(na):
        for j in range(nce): 
            # print(l,j)
            pen = pension(e[j])
            vr[j*na+l,nr-1] = u( a[l] * (1+rate) + pen + trbar, 0 )
            cropt[j*na+l,nr-1] = a[l]*(1+rate) + pen + trbar
    


    # workers' value function 
    vw = np.zeros((na*ny*nce,nw))
    awopt = np.ones((na*ny*nce,nw))
    cwopt = np.zeros((na*ny*nce,nw))
    laboropt = np.zeros((na*ny*nce,nw))


    # compute retiree's policy function 

    for i in  range(nr-1,0,-1):
        print(i)
        # the na x nce variables of the next-period value function are stored in y 
        y = vr[:,i]  
        y.shape = (nce,na)  
        y = np.transpose(y)          
        for l in range(na):
            asset0 = a[l]
            for j in range(nce):
                c0 = (1+rate)*a[l] + pension(e[j]) + trbar - a
                # zero and nonnegative values of c are handled in the procedure u
                numrows = c0.shape
                y0 = u(c0,np.zeros(numrows)) + sp1[nw+i-1]*b*y[:,j] 
                indexopt = np.argmax(y0)
                k1 = a[indexopt]
                aropt[j*na+l,i-1] = k1
                vr[j*na+l,i-1] = y0[indexopt]
                cropt[j*na+l,i-1] = (1+rate)*a[l] + pension(e[j]) + trbar - k1
 



    #  compuate decsion rules of workers

    for i in range (nw,0,-1):
        print("Compute the policy function of the worker of age= " + str(i))
        if i==nw:           # next-period value function is first year of retirement
            y=vr[:,0]       # value function of retired at retirement period i+1
            y.shape = (nce,na)  
            y = np.transpose(y) 
        else:
            for prod in range(ny):
                print(prod)
                # the na x nce variables of the productivity type prod are stored in y 
                y=vw[prod*na*nce:(prod+1)*na*nce,i]    
                y.shape = (nce,na)  
                y = np.transpose(y) 
                if prod==0:
                    zd1 = y
                elif prod==1:
                    zd2 = y
                elif prod==2:
                    zd3 = y
                elif prod==3:
                    zd4 = y
                elif prod==4:
                    zd5 = y
                else:
                    print("not implemented yet")
            
        for j in range(ny):  # productivity at age i
            ymax = wage*ef[i-1]*ye1[j]*labormax     # maximum labor income
            for l in range(na):     # asset leval at age i and productivity j
                m0=0                # index
                k0=a[l]
                for j0 in range(nce):   #  cumulated labor earnings at age i
                    e0=e[j0] 
                    # next-period cumulated earnings  
                    e1 = e0*(i-1)/i + wage*ef[i-1]*ye1[j]*lgrid/i
                    labor1 = lgrid*np.ones((na,nlabor))
                    zz = a*np.ones((nlabor,na)) 
                    zz = np.transpose(zz)
                    c0 = (1+rate)*k0 +(1-tau)*wage*ef[i-1]*ye1[j]*labor1
                    c0 = c0 + trbar - zz 
                    if i==nw:
                        # contruction of the next period value function 
                        for ilabor in range(nlabor):
                            e10=e1[ilabor]
                            # interpolation of value function at e10
                            # values of e below e10
                            i1 = np.count_nonzero([e<e10])
                            if i1<1:
                                i1 = 1
                            elif i1 > nce-1:
                                i1 = nce-1
                            
                            z1 = (e10-e[i1-1])/(e[i1]-e[i1-1])
                            
                            # stores the linear interpolated next-period value function for e1[ilabor]
                            v0 = (1-z1)*y[:,i1-1] + z1*y[:,i1]
                            
                            # np.c_[a,b] column stack, np.r_[a,b] row stack
                            if ilabor==0:
                                vtemp = v0
                            else:
                                vtemp = np.c_[vtemp,v0]

                        bellman = u(c0,labor1)
                        bellman = bellman+sp1[i-1]*b*vtemp
                        # index where bellman eq attains maximum
                        zmax = np.where(bellman == np.max(bellman))
                        iamax = zmax[0]
                        ilabormax = zmax[1]
                        k1 = a[iamax]
                        e1 = e1[ilabormax]
                        labor = lgrid[ilabormax]
                        c0 = (1+rate)*k0+(1-tau)*wage*ef[i-1]*ye1[j]*labor+trbar-k1                    
                        vw[j*nce*na+j0*na+l,i-1] = bellman[iamax,ilabormax]
                        awopt[j*nce*na+j0*na+l,i-1] = k1
                        cwopt[j*nce*na+j0*na+l,i-1] = c0
                        laboropt[j*nce*na+j0*na+l,i-1] = labor 
                        
                    else:   # i<nw
                        vtemp = np.zeros((na,nlabor))
                        # contruction of the next period value function ated next-period value function for e1[ilabor]
                        for iprod in range(ny):
                            # the na x nce variables of the productivity 
                            # type prod are stored in y     
                            y = vw[iprod*na*nce:(iprod+1)*na*nce,i]  
                            y.shape = (nce,na)  
                            y = np.transpose(y)
                            for ilabor in range(nlabor):
                                e10=e1[ilabor]
                                # interpolation of value function at e10
                                # values of e below e10
                                i1 = np.count_nonzero([e<e10])
                                if i1<1:
                                    i1 = 1
                                elif i1 > nce-1:
                                    i1 = nce-1
                            
                                z1 = (e10-e[i1-1])/(e[i1]-e[i1-1])
                            
                                # stores the linear interpolated next-period value function for e1[ilabor]
                                v0 = (1-z1)*y[:,i1-1] + z1*y[:,i1]
                            
                                # np.c_[a,b] column stack, np.r_[a,b] row stack
                                if ilabor==0:
                                    vtemp0 = v0
                                else:
                                    vtemp0 = np.c_[vtemp0,v0]
                            
                            vtemp = vtemp + py[j,iprod]*vtemp0;
                            
                       
                        bellman = u(c0,labor1)
                        bellman = bellman+sp1[i-1]*b*vtemp
                        # index where bellman eq attains maximum
                        zmax = np.where(bellman == np.max(bellman))
                        iamax = zmax[0]
                        ilabormax = zmax[1]
                        k1 = a[iamax]
                        e1 = e1[ilabormax]
                        labor = lgrid[ilabormax]
                        c0 = (1+rate)*k0+(1-tau)*wage*ef[i-1]*ye1[j]*labor+trbar-k1                    
                        vw[j*nce*na+j0*na+l,i-1] = bellman[iamax,ilabormax]
                        awopt[j*nce*na+j0*na+l,i-1] = k1
                        cwopt[j*nce*na+j0*na+l,i-1] = c0
                        laboropt[j*nce*na+j0*na+l,i-1] = labor 
                        

                        

        
    print("Compute aggregate variables")
    trs1=0 # this variable stores aggregate bequests


    ga = np.zeros((nage,ny,na,nce))

    # age-profiles
    bigl = 0				# aggregate labor 
    biga = 0				# aggregate assets
    bigcontrib = 0			# aggregate contributions to pension system
    bigbequests = 0			# aggregate accidental bequests
    bigpensions = 0			# total pensions
    agen = np.zeros(nage)     # assets of generation i
    fa = np.zeros(na)		    # distribution of wealth
    fy = np.zeros(nin)        # distribution of gross income 

    # initialization of distribution function at age 1
    # all households start with zero wealth
    for j in range(ny):     # productivity at age i 
        ga[0,j,0,0] = muy[0,j]
  
    
    # iteration over periods it=1,..,nage-1
    for i in range(nage-1):
        for ia in range(na):   # asset holding at age i
            asset0 = a[ia]
            for ice in range(nce):
                e0 = e[ice]
                for j in range(ny):
                    measure = ga[i,j,ia,ice]    # measure of household i,j,ia,ice
                    #if measure>0:
                     #   print("measure" +str(measure))
                      #  print(i,j,ia,ice)
                        
                    agen[i] = agen[i] + measure*asset0	
                    fa[ia] = fa[ia] + measure			# wealth distribution
                    if i<=nw-1:
                        #  working hours at age i
                        labor = laboropt[j*nce*na+ice*na+ia,i]	
                        # next-period assets
                        a1 = awopt[j*nce*na+ice*na+ia,i]	
                    else:
                        a1 = aropt[ice*na+ia,i-nw]
                        
                    if i<=nw-1:
                        e1 = e0*i/(i+1) + wage*ef[i]*ye1[j]*labor/(i+1)          # next-period cumulated earnings
                        bigl = bigl + labor*ef[i]*ye1[j]*measure
                        bigcontrib = bigcontrib + tau*labor*ef[i]*ye1[j]*wage*measure		# pension contributions                        
                        gy = wage*ef[i]*ye1[j]*labor+rate*asset0 # gross income of worker
                    else:
                        e1 = e0
                        bigpensions = bigpensions + pension(e0)*measure
                        gy = pension(e0) + rate*asset0 # gross income of retiree    
                              
                    if gy<=0:
                        fy[0] = fy[0] + measure
                    elif gy>=ymax:
                        fy[nin-1] = fy[nin-1] + measure
                    else:
                        in1 = np.count_nonzero([incomegrid<gy])
                        if in1<1:
                            in1 = 1
                        elif in1 > nin-1:
                            in1 = nin-1
                        
                        lambda0 = (incomegrid[in1]-gy) / (incomegrid[in1]-incomegrid[in1-1] )
                        fy[in1-1] = fy[in1-1] + lambda0*measure
                        fy[in1] = fy[in1] + (1-lambda0)*measure


                    trs1 = trs1 + a1*measure*(1-sp1[i])        
                    biga = biga + asset0*measure

                    #
                    # Step 4.2:
					# computation of next-periods distribution: 
                    #      method: bilinear interpolation
                    #      method: cubic spline interpolation 
                    #        
					#  for a[ia1]<= a1 <= a[ia1+1] 
					#  and e[j1]<= e1 <= e[j1+1]
                    #
					# check if a1<=0



                    if a1<=0:
                        if e1<=0:
                            for j1 in range(ny):
                                ga[i+1,j1,0,0] = (ga[i+1,j1,0,0]+
                                                  py[j,j1]*sp1[i]/(1+popgrowth0)*measure)
                                                      
                        elif e1>=emax:
                            for j1 in range(ny):
                                ga[i+1,j1,0,nce-1] = (ga[i+1,j1,0,nce-1]+
                                                  py[j,j1]*sp1[i]/(1+popgrowth0)*measure)
                                                    
                        else:  # linear interpolation between the two adjacent grid points of e1 on grid e
                            ice1 = np.count_nonzero([e<e1])
                            if ice1<1:
                                ice1 = 1
                            elif ice1 > nce-1:
                                ice1 = nce-1
							
                            lambda0 = (e[ice1]-e1) / (e[ice1]-e[ice1-1] )
                            for j1 in range(ny):
                                ga[i+1,j1,0,ice1-1]=ga[i+1,j1,0,ice1-1]+lambda0*py[j,j1]*sp1[i]/(1+popgrowth0)*measure  
                                ga[i+1,j1,0,ice1]=ga[i+1,j1,0,ice1]+(1-lambda0)*py[j,j1]*sp1[i]/(1+popgrowth0)*measure
                           
                        
                    elif a1>=kmax:
                        if e1<=0:
                            for j1 in range(ny):
                                ga[i+1,j1,na-1,0] = (ga[i+1,j1,na-1,0]+
                                                  py[j,j1]*sp1[i]/(1+popgrowth0)*measure)
                                                      
                        elif e1>=emax:
                            for j1 in range(ny):
                                ga[i+1,j1,na-1,nce-1] = (ga[i+1,j1,na-1,nce-1]+
                                                  py[j,j1]*sp1[i]/(1+popgrowth0)*measure)
                                                    
                        else:  # linear interpolation between the two adjacent grid points of e1 on grid e
                            ice1 = np.count_nonzero([e<e1])
                            if ice1<1:
                                ice1 = 1
                            elif ice1 > nce-1:
                                ice1 = nce-1
							
                            lambda0 = (e[ice1]-e1) / (e[ice1]-e[ice1-1] )
                            for j1 in range(ny):
                                ga[i+1,j1,na-1,ice1-1]=ga[i+1,j1,na-1,ice1-1]+lambda0*py[j,j1]*sp1[i]/(1+popgrowth0)*measure  
                                ga[i+1,j1,na-1,ice1]=ga[i+1,j1,na-1,ice1]+(1-lambda0)*py[j,j1]*sp1[i]/(1+popgrowth0)*measure
                        
                    else:
                        ia1 = np.count_nonzero([a<a1])
                        lambda1 =  (a[ia1]-a1) / (a[ia1]-a[ia1-1] )
						
                        if e1 <= 0: # linear interpolation between the two adjacent grid points on a1 on agrid
                            for j1 in range(ny):
                                ga[i+1,j1,ia1-1,0]=ga[i+1,j1,ia1-1,0] + lambda1*py[j,j1]*sp1[i]/(1+popgrowth0)*measure
                                ga[i+1,j1,ia1,0]=ga[i+1,j1,ia1,0] + (1-lambda1)*py[j,j1]*sp1[i]/(1+popgrowth0)*measure
                                                                
                        elif e1>=emax:
                             for j1 in range(ny):
                                ga[i+1,j1,ia1-1,nce-1]=ga[i+1,j1,ia1-1,nce-1] + lambda1*py[j,j1]*sp1[i]/(1+popgrowth0)*measure
                                ga[i+1,j1,ia1,nce-1]=ga[i+1,j1,ia1,nce-1] + (1-lambda1)*py[j,j1]*sp1[i]/(1+popgrowth0)*measure
                           
                        else: # linear interpolation       
                            ice1 = np.count_nonzero([e<e1])
                            if ice1<1:
                                ice1 = 1
                            elif ice1 > nce-1:
                                ice1 = nce-1
                                
                            lambda2 = (e[ice1]-e1) / (e[ice1]-e[ice1-1] )
                            
                            for j1 in range(ny):
                          
                                ga[i+1,j1,ia1-1,ice1-1] = ga[i+1,j1,ia1-1,ice1-1] +lambda1*lambda2*py[j,j1]*sp1[i]/(1+popgrowth0)*measure  
                                ga[i+1,j1,ia1,ice1-1] = ga[i+1,j1,ia1,ice1-1]+(1-lambda1)*lambda2*py[j,j1]*sp1[i]/(1+popgrowth0)*measure       
                                ga[i+1,j1,ia1-1,ice1] = ga[i+1,j1,ia1-1,ice1]+lambda1*(1-lambda2)*py[j,j1]*sp1[i]/(1+popgrowth0)*measure  
                                ga[i+1,j1,ia1,ice1] = ga[i+1,j1,ia1,ice1]+(1-lambda1)*(1-lambda2)*py[j,j1]*sp1[i]/(1+popgrowth0)*measure                                   
                           


    # last period of life
    i = nage -1 
    for ia in range(na):   # asset holding at age i
        asset0 = a[ia]
        for ice in range(nce):
            e0 = e[ice]
            for j in range(ny):
                measure = ga[i,j,ia,ice]    # measure of household i,j,ia,ice
                bigpensions = bigpensions + pension(e0)*measure
                biga =biga + asset0 * measure
                agen[i] = agen[i] + measure*asset0	
                fa[ia] = fa[ia] + measure			# wealth distribution
                gy = pension(e0) + rate*asset0 # gross income of retiree  
                         
                if gy<=0:
                    fy[0] = fy[0] + measure
                elif gy>=ymax:
                    fy[nin-1] = fy[nin-1] + measure
                else:
                    in1 = np.count_nonzero([incomegrid<gy])
                    if in1<1:
                        in1 = 1
                    elif in1 > nin-1:
                        in1 = nin-1
                        
                    lambda0 = (incomegrid[in1]-gy) / (incomegrid[in1]-incomegrid[in1-1] )
                    fy[in1-1] = fy[in1-1] + lambda0*measure
                    fy[in1] = fy[in1] + (1-lambda0)*measure
    
    
    
    # pension contributions: pensions relative to wage sum
    taunew = bigpensions / ((1-alp)*(biga/bigl)**alp)    
    ybar = biga**(alp)*bigl**(1-alp)

    print("")
    print("Aggregate values after interation:")
    print("kbar1 = " + str(biga))
    print("lbar1 = " + str(bigl))
    print("tau   = " + str(taunew))
    print("tbar  = " + str(trs1))
    print("Gini_w= " + str(ginid(a,fa,na)))
    print("Gini_y= " + str(ginid(incomegrid,fy,nin)))

    # save results
    np.save('vw',vw)
    np.save('vr',vr)
    np.save('cwopt',cwopt)
    np.save('cropt',cropt)
    np.save('awopt',awopt)
    np.save('aropt',aropt)

    # update
    kbar = update_agg*kbar+(1-update_agg)*biga
    nbar = update_agg*nbar+(1-update_agg)*bigl
    tau = update_agg*tau+(1-update_agg)*taunew
    trbar = update_agg*trbar+(1-update_agg)*trs1  
    
    
    crit = np.abs((kbar-biga)/biga)   
    
    q = q+1     # outer iteration over aggregates                   
     

print("runtime: --- %s seconds ---" % (time.time() - start_time))    
sec = (time.time() - start_time)
ty_res = time.gmtime(sec)
res = time.strftime("%H : %M : %S", ty_res)
print(res)
  
                        
