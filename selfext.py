import torch
import torch.nn as nn


class DSERope(nn.Module):
    def __init__(self,cos,isin,split=0.5):
        self.cos = cos 
        self.isin = isin
        
        self.original_len = self.cos[0]
        self.max_len = self.cos[0]
        self.split = split
        self.e = int((1-split)*self.original_len)
        self.r = self.original_len - self.e
        
        self.block_size = 2**6
        
        self._cos = cos 
        self._isin = isin
        
    def extend(self,freq:torch.Tensor,new_len):
        e = self.e
        r = self.r 
        e,r = freq[e:,:],freq[:e,:]
        rept = 1
        for i in range(2,100):
            if new_len<(i*e+r):
                rept = i
                break
        e = e.repeat_interleave(rept,dim=0)
        
        freq = self.stack([e,r],dim=0)
        
        return freq
    
    def reset(self):
        self.max_len = self._cos.shape[0]
        
        
    def forward(self,x):
        B,T,C = x.shape
        
        if T<self.max_len:
            self.cos = self.extend(self._cos,T)
            self.isin = self.extend(self._isin,T)
        
        return self.cos[-T:,:], self.isin[-T:,:]
        
        
        
        