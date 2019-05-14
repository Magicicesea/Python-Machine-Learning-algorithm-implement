from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probabilities. pi[i] = P(X_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probabilities. A[i, j] = P(X_t = s_j|X_t-1 = s_i))
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, o] = P(Z_t = z_o| X_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    # TODO
    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(X_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(X_t = s_j|X_t-1 = s_i))
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, o] = P(Z_t = z_o| X_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array delta[i, t] = P(X_t = s_i, Z_1:Z_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        ###################################################
        # Edit here
        ###################################################
        obs_index = self.obs_dict[Osequence[0]]
        alpha[:,0] = self.pi * self.B[:,obs_index]
        i = 1
        while i < L:
            obs_index = self.obs_dict[Osequence[i]]
            #sum_col = np.sum((self.A.T*alpha[:,0]).T,axis=0)
            alpha[:,i] = self.B[:,obs_index] * np.sum((self.A.T*alpha[:,i-1]).T,axis=0)
            i+=1
        return alpha

    # TODO:
    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(X_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(X_t = s_j|X_t-1 = s_i))
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, o] = P(Z_t = z_o| X_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array gamma[i, t] = P(Z_t+1:Z_T | X_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        ###################################################
        # Edit here
        ###################################################
        beta[:,L-1] = np.ones(S)

        i = L - 2
        while i >= 0:
            obs_index = self.obs_dict[Osequence[i+1]]
            #beta[:,i] = np.sum(self.A * self.B[:,obs_index] * beta[:,i+1],axis=0).transpose()
            beta[:,i] = np.sum((self.A * self.B[:,obs_index] * beta[:,i+1]).T,axis=0)
            i-=1

        return beta

    # TODO:
    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(Z_1:Z_T | λ)
        """
        prob = 0
        ###################################################
        # Edit here
        ###################################################
        alpha = self.forward(Osequence)
        prob = sum(alpha[:,len(Osequence) -1])
        return prob

    # TODO:
    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(X_t = i | O, λ)
        """
        ###################################################
        # Edit here
        ###################################################

        prob = self.forward(Osequence) * self.backward(Osequence) / self.sequence_prob(Osequence)
        return prob


    def transition_condition(self,Osequence):

        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - cond_prob: (L-1*num_state*num_state) A numpy list of Probability Array(num_state*num_state)
        """

        cond_prob = []
        L = len(Osequence)
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        prob = self.sequence_prob(Osequence)

        t = 0
        while t < L-1:
            obs_ind = self.obs_dict[Osequence[t+1]]
            #a = alpha[:,t]
            a = alpha[:,t][np.newaxis].T
            b = self.A[:,:]
            c = self.B[:,obs_ind]
            d = beta[:,t+1]

            cond_prob.append(np.multiply(np.multiply(a,b),np.multiply(c,d)) /prob)

            t+=1

        return cond_prob

    # TODO:
    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        
        S = len(self.pi)
        L = len(Osequence)
        delta = np.zeros([S,L])
        Delta = np.zeros(L)
        path = []
        ###################################################
        # Edit here
        ###################################################
        

        obs_index = self.obs_dict[Osequence[0]]
        delta[:,0] = self.pi * self.B[:,obs_index]

        i = 1
        while i < L:
            obs_index = self.obs_dict[Osequence[i]]
            delta[:,i] = self.B[:,obs_index]* np.nanmax(self.A.T*delta[:,i-1],axis=1)
            i+=1
        
        Delta[L-1] = int(np.argmax(delta[:,L-1]))

        i = L-2
        while i >=0:

            last_index = int(Delta[i+1])
            res = self.A[:,last_index].T*delta[:,i]
            Delta[i] = np.argmax(res)
            i-=1
        
        inv_state_dict = {v: k for k, v in self.state_dict.items()}
        for inx in Delta:
            path.append(inv_state_dict[inx])

        return path
