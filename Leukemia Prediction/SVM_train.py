#! python3

"""
Created on Thur Aug 10 9:34:59 2017

@author: Rex
"""

'''
     Below is the pseudo code of SMO algorithm implemented in fitting funciton (the one named as 'fit(X,Y)' )
       
    ◦ Initialize alpha[i] = 0, ∀i, b = 0.
    ◦ Initialize passes = 0
    ◦ while (passes < max passes)
        ◦ num changed alphas = 0.
            ◦ for i = 1, . . .m,
                ◦ Calculate Ei = f(x(i)) − y(i) using (2).
                ◦ if ((y(i)Ei < −tol && alpha[i] < C) || (y(i)Ei > tol && alpha[i] > 0))
                    ◦ Select j (not equal to i) randomly.
                    ◦ Calculate Ej = f(x(j)) − y(j) using (2).
                    ◦ Save old alpha’s
                    ◦ Compute L and H by (10) or (11).
                    ◦ if (L == H)
                        continue to next i.
                    ◦ Compute eta by (14).
                    ◦ if (eta >= 0)
                        continue to next i.
                   ◦ Compute and clip new value for alpha[j] using (12) and (15).
                   ◦ if ( | alpha[j]_new - alpha[j]_old | < 10^(−5) )
                       continue to next i.
                   ◦ Determine new value for alpha[i] using (16).
                   ◦ Compute b1 and b2 using (17) and (18) respectively.
                   ◦ Compute b by (19).
                   ◦ num changed alphas := num changed alphas + 1.
               ◦ end if
        ◦ end for
        ◦ if (num changed alphas == 0)
            passes := passes + 1
        ◦ else
            passes := 0
    ◦ end while
    
    Equations :
        
        • f(x) = summation i = 1 to m (alpah[i]*y[i] * <x[i], x[j]>) + b     (2)
          where <x, y> = dot product of (transpose(x) and y)
        
        • If y(i) =/= y(j), L = max(0, alpha[j] − alpha[i]),      H = min(C,C + alpha[j] − alpha[i]) (10)
        • If y(i) == y(j),  L = max(0, alpha[i] + alpha[j] − C),  H = min(C, alpha[i] + alpha[j])    (11)
        
        • neta = 2<x(i), x(j)> − <x(i), x(i)> − <x(j), x(j)>      (14)
        
        
        • alpha[j] = alpha[j]_old - y[i]*(E[i] - E[j]) / neta     (12)
        
                     |  H ,         if alpha[j] > H
        • alpha[j] = |  alpha[j],   if L < alpha[j] < H            (15)
                     |  L ,         if alpha[j] < L
                       
        • alpha[i] = alpha[i]_old + y[i]*y[j](alpha[j]_old - alpha[j])     (16)
        
        • b1 = b − Ei − y(i)(alpha[i] − alpha[i]_old)<x(i), x(i)> − y(j)(alpha[j] − alpha[j]_old)<x(i), x(j)>  (17)
        
        • b2 = b − Ej − y(i)(alpha[i] − alpha[i]_old)<x(i), x(j)> − y(j)(alpha[j] − alpha[j]_old)<x(j), x(j)>  (18)
        
              |  b1         , if 0 < alpha[i] < C
        • b = |  b2         , if 0 < alpha[j] < C           (19)
              | (b1 + b2)/2 , otherwise
'''

import numpy as np
import random as rnd

'''
    List of methods in class SMO_train along with their return types:
        
        train(X, Y)    tuple of arrays of dimension n {first array is of alpha and second is of weights}
        
'''

class SMO_train:
    
    def __init__(self, max_iter=10000, C=100.0, tol=0.001, epsilon=0.00001):
        self.max_iter = max_iter
        self.C = C
        self.tol = tol
        self.epsilon = epsilon
        self.alpha

    # defining the funciton for training the SVM using the SMO algorithm for optimization
    # will return the parameters array alpha
    def train(self, X, Y):
        n = X.shape[0]
        alpha = np.zeros((n))
        b = 0
        count = 0
        while count < self.max_iter:
            
            # alpha_prev = np.copy(alpha)
            num_changed_alpha = 0
            
            for i in range(1, n+1):
                
                # defining x_i and y_i
                x_i, y_i = X[i, :], Y[i]
                
                # computing E_i
                E_i = self.E(x_i, y_i, self.w, self.b)
                
                if(y_i * E_i < -self.tol and alpha[i] < self.C) or (y_i * E_i > self.tol and alpha[i] > 0):
                    
                    # getting random j
                    j = self.get_random_int(1, n-1, i)
                    
                    # defining x_j and y_j and saving alpha[i] and alpha[j] old
                    alpha_pi, alpha_pj = alpha[i], alpha[j]
                    x_j, y_j = X[j, :], Y[j]
                    
                    # computing weights and parameter b
                    self.w = self.get_weights(alpha, X, Y)
                    self.b = self.calculate_b(X, Y, self.w)
                    
                    #computing E_j
                    E_j = self.E(x_j, y_j, self.w, self.b)
                    
                    # computing L and H
                    (L, H) = self.get_L_H(self.C, y_i, y_j, alpha[i], alpha[j])
                    
                    if L == H :
                        continue
                    
                    # computing neta
                    neta = 2 * self.kernel(x_i, x_j) - self.kernel(x_i, x_i) - self.kernel(x_j, x_j)
                    
                    if neta >= 0 :
                        continue
                    
                    # computing and clipping new alpha[j]
                    alpha[j] = alpha_pj - y_i * (E_i - E_j) / neta
                    alpha[j] = min(alpha[j], H)
                    alpha[j] = max(alpha[j], L)
                    
                    if alpha[j] - alpha_pj < self.epsilon :
                        continue
                    
                    # computing new alpha[i] , b1, b2 and b
                    alpha[i] = alpha_pi + y_i*y_j*(alpha_pj - alpha[j])
                    b1 = b - E_i - y_i*(alpha[i] - alpha_pi)*self.kernel(x_i, x_i) - y_j * (alpha[j] - alpha_pj) * self.kernel(x_i, x_j)
                    b2 = b - E_j - y_i*(alpha[i] - alpha_pi)*self.kernel(x_i, x_j) - y_j * (alpha[j] - alpha_pj) * self.kernel(x_j, x_j)
                    
                    if alpha[i] < self.C and alpha[i] > 0:
                        b = b1
                    elif alpha[j] < self.C and alpha[j] > 0 :
                        b = b2
                    else :
                        b = (b1 + b2)/2
                    
                    num_changed_alpha += 1
                    
                # end if
            # end for
            if num_changed_alpha == 0:
                count += 1
            else :
                count = 0
        
        # end while
        
        # computing the final value of parameters b and alpha
        self.w = self.get_weights(alpha, X, Y)
        self.b = self.calculate_b(X, Y, self.w)
        
        return (alpha, self.w)
    
    ####################################################################################################    
    # These all are helper functions
    
    # defining and computing kernel(x_i, x_j)
    def kernel(x1, x2):
        return np.dot(x1, x2.T)
                    
    # computing L and H
    def get_L_H(C, y_i, y_j, alpha_i, alpha_j):
        if y_i == y_j :
            return ( max(0, alpha_i + alpha_j - C), min(C, alpha_i + alpha_j) )
        else :
            return ( max(0, alpha_j - alpha_i), min(C, C + alpha_j - alpha_i) )
    
    # computing random i~=z
    def get_rand_int(a, b, z):
        i = z
        while i == z:
            i = rnd.randint(a, b)
        return i
    
    # computing hypothesis fuction value
    def h(X, w, b):
        return np.sign(np.dot(w.T, X.T) + b).astype(int)
    
    # computing weights w_i
    def get_weights(alpha, X, Y):
        return np.dot(alpha * Y, X)
    
    # computing constant b
    def calculate_b(X, Y, w):
        b_temp = Y - np.dot(w.T, X.T)
        return np.mean(b_temp)
    
    # computing error in prediction of hypothesis function (E_x)
    def E(self, x_k, y_k, w, b):
        return self.h(x_k, w, b) - y_k
    
    #####################################################################################################