import numpy as np
import torch
from torch.linalg import lstsq
from submodules.function import *


class GS:
    def __init__(self, iw:np.ndarray, ew_int_record:np.ndarray, propagator, metric:list, obj=None):

        #torch.set_default_device('cuda')

        if len(iw.shape)==2:
            iw = iw[None,:,:]
            ew = ew[None,:,:]

        self.iw_raw = torch.tensor(iw)
        self.ew_raw = torch.tensor((ew_int_record/ew_int_record.sum())**0.5)
        self.prop = propagator
        self.metric = metric
        if obj is not None:
            self.obj_0 = torch.tensor(obj)
        else:
            self.obj_0 = torch.ones(self.obj.shape)
        self.alg_lst = [self.alg_0, self.alg_1, self.alg_2, self.alg_3]

    def _prepare(self, dose):
        self.iw, self.ew = torch.clone(self.iw_raw).type(torch.complex64), torch.clone(self.ew_raw).type(torch.complex64)
        self._copy_iw = torch.clone(self.iw)
        for i,e in zip(self.iw, self.ew):
            i /= (torch.abs(i)**2).sum() ** 0.5
            e /= (torch.abs(e)**2).sum() ** 0.5
        if dose != 0:
            np.random.seed(0)
            self.ew = (torch.poisson(torch.abs(self.ew)**2*dose).type(torch.complex64)/dose)**0.5

        self.obj = torch.tensor(np.exp(np.random.rand(self.iw.shape[1],self.iw.shape[2])*1j*2*np.pi -np.pi)) #Initialize Object as Random number between -pi and pi
        #self.obj = torch.ones(self.iw.shape[-2:],dtype=torch.complex64)*torch.exp(torch.rand(self.iw.shape[-2:])*np.pi*1j)
        #self.obj = torch.exp(torch.ones(self.iw.shape[-2:],dtype=torch.complex64)*1j*np.pi)
        self.idx = torch.abs(self.iw_raw[0])>(torch.abs(self.iw_raw[0]).max()*0.1)
        
        _idx = np.array(self.idx.to('cpu'))

    def _metric(self):  
        if len(self.metric) != 0:
            for i in range(len(self.metric)):
                self.score[i, self.ite] = self.metric[i](self)

    def _update_obj(self, update, step_size, b_ph):
        self.obj[self.idx] += (update[self.idx] - self.obj[self.idx]) * step_size 
        if b_ph:
            self.obj = torch.exp(1j*torch.angle(self.obj))
            #self.obj /= torch.abs(self.obj)

    def _update_obj_ortho_gd(self, update, step_size, b_ph):
        '''
        R = conj(in) * error * ew
        gradient = 2A^T (Ax-B)
        '''
        _ew    = torch.zeros(self.ew.shape,dtype=torch.complex64)
        _obj   = torch.zeros(update.shape,dtype=torch.complex64)
        update = torch.clone(self.obj)
        self.iw_inner = self.iw.conj()[:,None,self.idx] * self.prop(self.ew, -1)[:,self.idx]
        self.R = (self.iw_inner / self.obj[self.idx]).sum(-1) - self.I

        _iw_inner = torch.zeros(self.iw_inner.shape,dtype=torch.complex64)
        _R = torch.zeros(self.R.shape,dtype=torch.complex64)

        self.inv_e_grad = 2 * \
            torch.reshape(self.iw_inner, (self.iw_inner.shape[0]*self.iw_inner.shape[1],-1)).T @ \
            (-1 * self.R.flatten())
        step_grad = min(0.1/abs(self.inv_e_grad).max()*(1-abs(torch.diagonal(self.R)).min()),0.1)

        if abs(self.R).mean()<1e-4:
            return False
        for _ in range(4):
            update[self.idx] += ((self.obj[self.idx]**-1 - step_grad*self.inv_e_grad)**-1 - self.obj[self.idx])*step_size
            if b_ph:
                update /= torch.abs(update)
            _ew = self.prop(self.iw * update,1)
            _ew = torch.exp(1j*torch.angle(_ew)) * torch.abs(self.ew)
            _iw_inner = self.iw.conj()[:,None,self.idx] *  self.prop(_ew, -1)[:,self.idx]
            _R = (_iw_inner / update[self.idx]).sum(-1) - self.I

            residual = [abs(self.R).sum(),  abs(_R).sum()]
            if residual[1] < residual[0]:
                self.obj[self.idx] = update[self.idx]
                if b_ph:
                    self.obj = torch.exp(1j*torch.angle(self.obj))
                return True
            else:
                step_grad *= 0.5
                update[self.idx] = self.obj[self.idx]
        return False

    def _update_obj_ortho(self, update, step_size, b_ph):
        '''
        R = conj(in) * error * ew
        gradient = 2A^T (Ax-B)
        '''
        update = torch.clone(self.obj)
        self.iw_inner = self.iw.conj()[:,None,self.idx] * self.prop(self.ew, -1)[:,self.idx]
        self.R = (self.iw_inner / self.obj[self.idx]).sum(-1) - self.I

        self.inv_e_grad = 2 * \
            torch.reshape(self.iw_inner, (self.iw_inner.shape[0]*self.iw_inner.shape[1],-1)).T @ \
            (-1 * self.R.flatten())
        step_grad = 0.05/abs(self.inv_e_grad).max()*(1-abs(torch.diagonal(self.R)).min())
        self.obj[self.idx] += ((self.obj[self.idx]**-1 - step_grad*self.inv_e_grad)**-1 - self.obj[self.idx])*step_size

    def alg_0(self, _obj, _e, step_size, b_phobj):
        '''
        gs, update object on each iw/ew pair
        '''
        for i,e in zip(self.iw, self.ew):
            _e = self.prop(i * self.obj,1)
            e = torch.exp(1j*torch.angle(_e)) * torch.abs(e)
            _obj = self.prop(e,-1) / i 
            self._update_obj(_obj, step_size, b_phobj)
                
    def alg_1(self, _obj, _ew, step_size, b_phobj): ##Batch update for the object, since we take the array with all configs at once pass _iw as an argument to this too
        '''
        gs, update object after iw/ew 
        '''
        _ew = self.prop(self.iw * self.obj,1) # Propagate current object multiplied by current iw to the detector plane
        dc_filter = torch.angle(_ew).mean()
        self.ew = torch.exp(1j*(torch.angle(_ew))-dc_filter) * torch.abs(self.ew) # Replace amplitude of the measured ew multiplied by the phase of the guessed ew
        #self.ew = torch.exp(1j*(torch.angle(_ew))) * torch.abs(self.ew) # Replace amplitude of the measured ew multiplied by the phase of the guessed ew
        _obj = (self.prop(self.ew,-1) / self.iw).mean(0) # Backpropagate update and divide by iw to remove (multi-dimensional matrix, batch update will average out all the _obj from each nRPI)
        ## What if you DC filter before the mean
        self._update_obj(_obj, step_size, b_phobj) # Update current object estimate using the object from previous line   

    def alg_2(self, _obj, _ew, step_size, b_phobj):
        '''
        update object by optimizing the orthogonality of the waves
        requres incident waves to be strictly orthogonal
        '''
        _ew = self.prop(self.iw * self.obj,1)
        self.ew = torch.exp(1j*torch.angle(_ew)) * torch.abs(self.ew) 
        _obj = (self.prop(self.ew,-1) / self.iw).mean(0)

        self._update_obj(_obj, step_size, b_phobj)
        #self._update_obj_ortho(_obj, step_size, b_phobj)
        self._update_obj_ortho_gd(_obj, step_size, b_phobj)

    def alg_3(self, _obj, _ew, step_size, b_phobj):
        '''
        update object by optimizing the orthogonality of the waves
        requres incident waves to be strictly orthogonal
        '''

        _ew = self.prop(self.iw * self.obj,1)
        self.ew = torch.exp(1j*torch.angle(_ew)) * torch.abs(self.ew) 
        _obj = (self.prop(self.ew,-1) / self.iw).mean(0)

        self._update_obj(_obj, step_size, b_phobj)

        if self.ite > 0:
            self.iw = self._copy_iw * torch.exp(-1j*torch.angle(_obj))
            self.ew = torch.abs(self.prop(self.iw*self.obj_0,1))**2
            self.ew = (self.ew/self.ew.sum())**0.5
            
            for i,e in zip(self.iw, self.ew):
                i /= (torch.abs(i)**2).sum() ** 0.5
                e /= (torch.abs(e)**2).sum() ** 0.5

            self.ew = (torch.poisson(torch.abs(self.ew)**2*self.dose).type(torch.complex64)/self.dose)**0.5

    def run(self, type, ite_max, step_size, dose, b_phobj, phase_matrix=None):
        ## allocate
        self.dose = dose
        self._prepare(dose)
        self.ite_max = ite_max
        if len(self.metric) != 0:
            self.score = np.zeros((len(self.metric), ite_max+1))
        if type == 0:
            _ew   = torch.zeros(self.iw.shape[-2:],dtype=torch.complex64).to('cuda') #Instantiated exit wave guess
        else:
            _ew   = torch.zeros(self.iw.shape,dtype=torch.complex64).to('cuda')  #Instantiated exit wave guess
        _obj = torch.zeros(self.iw.shape[-2:],dtype=torch.complex64)
        self.I = (self.iw.conj()[:,None,self.idx] * self.iw[:,self.idx]).sum(-1)
       
        if (type == 2) or (type == 3):
            self.R = torch.zeros((self.iw.shape[0],self.iw.shape[0]),dtype=torch.complex64)
            self.iw_inner = torch.zeros((self.iw.shape[0],self.iw.shape[0]),dtype=torch.complex64)
            self.inv_e_grad = torch.zeros(self.idx.sum(),dtype=torch.complex64)
            self.zero = torch.zeros(self.idx.sum(),dtype=torch.complex64)
        if type == 3:
            self._exit0  = torch.zeros(self.ew.shape,dtype=torch.complex64)
            self._exit1  = torch.zeros(self.ew.shape,dtype=torch.complex64)
        if type == 4:
            self.pm  = torch.from_numpy(phase_matrix,dtype=torch.complex64)
            self.pmi = torch.conj(self.pm).T
            self.idx_set = torch.zeros(self.iw.shape,dtype=torch.bool)
            self.iw_dia = torch.clone(self.iw)
            self.ew_dia = torch.clone(self.ew)
            self.ew_adjust = torch.clone(self.ew)
            for i in range(len(self.iw)):
                self.iw_dia[i] = (self.iw*self.pmi[i][:,None,None]).sum(0)
                self.idx_set[i] = abs(self.iw_dia[i]) > (abs(self.iw_dia[i]).max()*0.5)

        self.ite = -1
        ## run search
        for self.ite in range(ite_max):
            self._metric()
            self.alg_lst[type](_obj, _ew, step_size, b_phobj) 
        self.ite += 1
        self._metric()

