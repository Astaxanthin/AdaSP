import torch
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F

def distMC(Mat_A, Mat_B, norm=1, cpu=False, sq=True):#N by F
    N_A = Mat_A.size(0)
    N_B = Mat_B.size(0)
    
    DC = Mat_A.mm(torch.t(Mat_B))
    if cpu:
        if sq:
            DC[torch.eye(N_A).bool()] = -norm
    else:
        if sq:
            DC[torch.eye(N_A).bool().cuda()] = -norm
            
    return DC

def Mat(Lvec):
    N = Lvec.size(0)
    Mask = Lvec.repeat(N,1)
    Same = (Mask==Mask.t())
    return Same.clone().fill_diagonal_(0), ~Same#same diff
    
class EPHNLoss(Module):
    def __init__(self,s=0.1):
        super(EPHNLoss, self).__init__()
        self.semi = False
        self.sigma = s
        
    def forward(self, fvec, Lvec):
        N = Lvec.size(0)
        fvec_norm = F.normalize(fvec, p = 2, dim = 1)
        # matting
        Same, Diff = Mat(Lvec.view(-1))
        
        # Similarity Matrix
        Dist = distMC(fvec_norm,fvec_norm)
        
        ############################################
        # finding max similarity on same label pairs
        D_detach_P = Dist.clone().detach()
        D_detach_P[Diff]=-1
        D_detach_P[D_detach_P>0.9999]=-1
        V_pos, I_pos = D_detach_P.max(1)
 
        # prevent duplicated pairs
        Mask_not_drop_pos = (V_pos>0)

        # extracting pos score
        Pos = Dist[torch.arange(0,N), I_pos]
        Pos_log = Pos.clone().detach().cpu()
        
        ############################################
        # finding max similarity on diff label pairs
        D_detach_N = Dist.clone().detach()
        D_detach_N[Same]=-1
        if self.semi:
            D_detach_N[(D_detach_N>(V_pos.repeat(N,1).t()))&Diff]=-1#extracting SHN
        V_neg, I_neg = D_detach_N.max(1)
            
        # prevent duplicated pairs
        Mask_not_drop_neg = (V_neg>0)

        # extracting neg score
        Neg = Dist[torch.arange(0,N), I_neg]
        Neg_log = Neg.clone().detach().cpu()
        
        # triplets
        T = torch.stack([Pos,Neg],1)
        Mask_not_drop = Mask_not_drop_pos&Mask_not_drop_neg

        # loss
        Prob = -F.log_softmax(T/self.sigma,dim=1)[:,0]
        loss = Prob[Mask_not_drop].mean()

        #print('loss:{:.3f} rt:{:.3f}'.format(loss.item(), Mask_not_drop.float().mean().item()), end='\r')

        # return loss, torch.cat([Pos_log[(V_pos>0)], torch.Tensor([0,1])],0), torch.cat([Neg_log[(V_neg>0)], torch.Tensor([0,1])],0), Pos_log.mean()-Neg_log.mean()
        
        return loss
    
