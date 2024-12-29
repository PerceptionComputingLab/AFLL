from torch import nn
import torch
def js_div(p_output, q_output):
    """
    Function that measures JS divergence between target and output logits:
    """
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    log_mean_output = torch.log((p_output + q_output )/2)
    return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output))/2

def cjs(p,q):
    cdf_p = torch.cumsum(p,dim=1)
    # print("cdf_p:\n",cdf_p)
    cdf_q = torch.cumsum(q, dim=1)
    # print("cdf_q:\n", cdf_q)
    cjs_t = js_div(cdf_p, cdf_q)
    # for i in range(cls):
    #     cjs_t = cjs_t + js_div(cdf_p[:,i],cdf_q[:,i])
    return cjs_t

def bi_directional_kl(p, q):
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    return 0.5 * KLDivLoss(torch.log(p), q) + 0.5 * KLDivLoss(torch.log(q), p)

def hellinger(p,q):
    return torch.sqrt(0.5 * torch.sum((torch.sqrt(p) - torch.sqrt(q)) **2))
