#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 15:18:25 2019

@author: gionuno
"""

import numpy        as np;
import numpy.random as rd;

import matplotlib.colors as csp;
import matplotlib.pyplot as plt;
import matplotlib.image  as img;

def proj_splx(y):
    sy = np.sort(y)[::-1];
    cy = np.cumsum(sy);
    t = None;
    for i in range(y.shape[0]-1):
        t_i = (cy[i]-1.0)/(i+1);
        if t_i >= sy[i+1]: 
            t = t_i; 
            break;
    if t == None:
        t = (np.sum(y)-1.0)/y.shape[0];
    x = (y > t)*(y-t);
    return x;

def ot_map(alpha,beta,C):
    T = np.outer(alpha,np.ones(beta.shape))+np.outer(np.ones(alpha.shape),beta)-C;
    T = T*(T>0.0);
    return T;

def ot(a,b,igam,T,alpha,beta):
    return np.dot(a,alpha)+np.dot(b,beta)-0.5*igam*np.sum(T**2);
    
def grad_ot_alpha(a,igam,T):
    return a - igam*np.sum(T,axis=1);

def grad_ot_beta(b,igam,T):
    return b - igam*np.sum(T,axis=0);

def opt_ot(a,b,C,igam=1e-1,L=100):
    
    alpha = 1e-14*rd.rand(a.shape[0]);
    beta  = 1e-14*rd.rand(b.shape[0]);
    
    galphal = np.zeros(alpha.shape);
    gbetal  = np.zeros(beta.shape);
    
    salpha  = np.zeros(alpha.shape);
    sbeta   = np.zeros(beta.shape);
    
    T = ot_map(alpha,beta,C);

    otab = ot(a,b,igam,T,alpha,beta);
    
    c1 = 1e-4; 
    
    dt = 1.0;
    
    for l in range(L+1):
        dt = min(1.25*dt,1.0);
        
        galpha = grad_ot_alpha(a,igam,T);
        
        balpha = 0.0 if l == 0 else np.dot(galpha,galpha-galphal)/(1e-14+np.dot(galphal,galphal));
        
        salpha = galpha + balpha*salpha; 
        
        next_T      = ot_map(alpha+dt*salpha,beta,C);
        next_otab   = ot(a,b,igam,next_T,alpha+dt*salpha,beta);
        
        dtg_alpha = np.dot(galpha,salpha);
        
        for k in range(10):
            if next_otab >= otab + c1*dt*dtg_alpha:
                break;
            dt *= 0.75;
            next_T      = ot_map(alpha+dt*salpha,beta,C);
            next_otab   = ot(a,b,igam,next_T,alpha+dt*salpha,beta);
                
        alpha = alpha+dt*salpha;
        
        galphal = galpha;
        
        T    = next_T;
        otab = next_otab;

        dt = min(1.25*dt,1.0);
        
        gbeta  = grad_ot_beta(b,igam,T);

        bbeta = 0 if l == 0 else np.dot(gbeta,gbeta-gbetal)/(1e-14+np.dot(gbetal,gbetal));
        
        sbeta = gbeta + bbeta*sbeta;
        
        next_T     = ot_map(alpha,beta+dt*sbeta,C);
        next_otab  = ot(a,b,igam,next_T,alpha,beta+dt*sbeta); 

        dtg_beta = np.dot(gbeta,sbeta);
        
        for k in range(10):
            if next_otab >= otab + c1*dt*dtg_beta:
                break;
            dt *= 0.75;
            next_T     = ot_map(alpha,beta+dt*sbeta,C);
            next_otab  = ot(a,b,igam,next_T,alpha,beta+dt*sbeta);
        
        beta = beta+dt*sbeta;
        
        gbetal = gbeta;
        
        T    = next_T;
        otab = next_otab;
        if l%10 == 0:
            print(str(l)+" "+str(otab)+" "+str(np.sum(C*T)/(np.sum(T)+1e-14))+" "+str(dt));
    return T;
