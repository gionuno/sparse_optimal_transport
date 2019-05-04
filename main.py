#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 18:38:19 2019

@author: gionuno
"""

from sot import *;

def rgb2luv(I):
    J = np.zeros(I.shape);
    J[:,:,0] = np.sum(I,axis=2);
    J[:,:,1] = I[:,:,0]/(1e-14+J[:,:,0]);
    J[:,:,2] = I[:,:,1]/(1e-14+J[:,:,0]);
    return J;

def luv2rgb(J):
    I = np.zeros(J.shape);
    I[:,:,0] = (J[:,:,0]+1e-14)*J[:,:,1];
    I[:,:,1] = (J[:,:,0]+1e-14)*J[:,:,2];
    I[:,:,2] = (J[:,:,0]+1e-14)*(1.0-J[:,:,1]-J[:,:,2]);
    return I;

def make_2d_histogram(X,B):
    H = np.zeros((B,B));
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
           r = X[i,j,1];
           g = X[i,j,2];
           
           k = int(np.clip(B*r,0,B-1));
           l = int(np.clip(B*g,0,B-1));

           H[k,l] += 1.0;
    
    H /= X.shape[0]*X.shape[1];
    return H;

def make_3d_histogram(X,B):
    H = np.zeros((B,B,B));
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            r = int(np.clip(B*X[i,j,0],0,B-1));
            g = int(np.clip(B*X[i,j,1],0,B-1));
            b = int(np.clip(B*X[i,j,2],0,B-1));
            H[r,g,b] += 1.0;
    H /= X.shape[0]*X.shape[1];
    return H;

X = img.imread("blossom.jpg")/255.0;
Y = img.imread("forest.jpg")/255.0;

luvX = rgb2luv(X);
luvY = rgb2luv(Y);

chrX = X/(1e-14+np.repeat(luvX[:,:,0].reshape((X.shape[0],X.shape[1],1)),3,axis=2));
chrY = Y/(1e-14+np.repeat(luvY[:,:,0].reshape((Y.shape[0],Y.shape[1],1)),3,axis=2));

B = 64;

L = np.linspace(0.0,1.0,B);
C = np.zeros((B,B,3));

for k in range(B):
    for l in range(B-k):
        r = L[k];
        g = L[l];
        b = 1.0-r-g;
        C[k,l,0] = r;
        C[k,l,1] = g;
        C[k,l,2] = b;
            
mu = make_2d_histogram(luvX,B);
nu = make_2d_histogram(luvY,B);

smu = mu/np.max(mu);
snu = nu/np.max(nu);

Cmu = C*(0.125+0.875*np.repeat(smu.reshape((B,B,1)),3,axis=2));
Cnu = C*(0.125+0.875*np.repeat(snu.reshape((B,B,1)),3,axis=2));


f, ax = plt.subplots(2,4);
ax[0,0].imshow(X);
ax[0,1].imshow(luvX[:,:,0],cmap='gray');
ax[0,2].imshow(chrX);
ax[0,3].imshow(Cmu,cmap='gray');

ax[1,0].imshow(Y);
ax[1,1].imshow(luvY[:,:,0],cmap='gray');
ax[1,2].imshow(chrY);
ax[1,3].imshow(Cnu,cmap='gray');
plt.show();

B = 32;

D = 4*(B-1.0)**2*np.ones((B*B,B*B));
for i in range(B):
    for j in range(B-i):        
        for k in range(B):
            for l in range(B-k):
                D[B*i+j,B*k+l] = (i-k)**2+(j-l)**2;
D /= (B-1.0)**2;
plt.imshow(D,cmap='hot');
plt.show();

mu = make_2d_histogram(luvX,B);
nu = make_2d_histogram(luvY,B);

T = opt_ot(nu.reshape((-1,)),mu.reshape((-1,)),D,L=1000,igam=1e-1);
#T += 1e-14;
#T /= np.sum(T);

plt.imshow(T,cmap='gray');
plt.show();

om = np.zeros((B,B,3));
for i in range(B):
    for j in range(B-i):
        som = 1e-14;
        for k in range(B):
            for l in range(B-k):
                om[i,j,0] += k*T[B*i+j,B*k+l];
                om[i,j,1] += l*T[B*i+j,B*k+l];
                om[i,j,2] += (B-1-k-l)*T[B*i+j,B*k+l];
                som += T[B*i+j,B*k+l];
        om[i,j,0] /= som;
        om[i,j,1] /= som;
        om[i,j,2] /= som;

cY = np.zeros(Y.shape);
nY = np.zeros(Y.shape);
for i in range(Y.shape[0]):
    for j in range(Y.shape[1]):
        r = luvY[i,j,1];
        g = luvY[i,j,2];
        b = 1-r-g;
        
        k = int(np.clip(B*r,0.0,B-1.0));
        l = int(np.clip(B*g,0.0,B-1.0));
        
        cY[i,j,:] = (luvY[i,j,0]+1e-14)*np.array([r,g,b]);
        nY[i,j,:] = (luvY[i,j,0]+1e-14)*om[k,l,:]/(B-1.0);

f, ax = plt.subplots(1,2);
ax[0].imshow(cY);
ax[1].imshow(nY);
plt.show();