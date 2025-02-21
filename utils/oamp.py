import torch

def batch_multinormal_zero_pdf(mean, var):
    # mean: P * N * M
    # var: P * M
    # var = torch.abs(var)
    temp_var = torch.diag_embed(1/var)
    denom = -torch.abs((mean[:,:,None,:].conj() @ temp_var[:,None,:,:] @ mean[:,:,:,None])).squeeze()
    nom = torch.abs(torch.prod(var,dim=1,keepdim=True))
    pdf = torch.exp(denom)/(nom)
    return pdf

def OAMP(Y, A, B, device=None):

    if device is None:
        device = Y.device
    
    P,Q,M = Y.shape
    _,_,N = A.shape


    h_mean = torch.zeros([P, N, M], dtype=torch.complex64, device=device)
    h_var = torch.ones([P, M], dtype=torch.float32, device=device)
    h_mean_pri_le = torch.zeros([P, N, M], dtype=torch.complex64, device=device)
    h_var_pri_le = torch.ones([P, M], dtype=torch.float32, device=device)
    c_mean = torch.zeros([P, N, M], dtype=torch.complex64, device=device)
    c_var = torch.ones([P, M], dtype=torch.float32, device=device)
    c_mean_pri_le = torch.zeros([P, N, M], dtype=torch.complex64, device=device)
    c_var_pri_le = torch.ones([P, M], dtype=torch.float32, device=device)

    noise_var = 0
    sigma = torch.zeros([M,Q,Q], dtype=torch.float32, device=device)
    h_mean_post = torch.zeros([P, N, M], dtype=torch.complex64, device=device)
    h_var_post = torch.ones([P, M], dtype=torch.float32, device=device)
    c_mean_ext_nle = torch.zeros([P, N, M], dtype=torch.complex64, device=device)
    c_var_ext_nle = torch.ones([P, M], dtype=torch.complex64, device=device)

    BBH = torch.matmul(B,B.conj().transpose(1,2))
    lambda_pri = 0.1 * torch.ones((P,N),dtype=torch.float32, device=device)

    for idx_iter in range(30):
        # LE of H
        sigma = h_var_pri_le[:,:,None,None] * torch.eye(Q,device=device) + c_var_pri_le[:,:,None,None] * BBH[:,None,:,:] + noise_var * torch.eye(Q,device=device)
        sigma_inv = torch.linalg.inv(sigma)
        res = Y - torch.matmul(A, h_mean_pri_le) - torch.matmul(B, c_mean_pri_le)
        lmmse_tmp = (sigma_inv @ res.transpose(1,2)[:,:,:,None]).transpose(1,2).squeeze(-1)
        lmmse_tmp = h_var_pri_le[:,None,:] * (A.conj().transpose(1,2) @ lmmse_tmp)
        h_mean_post_le = h_mean_pri_le + lmmse_tmp
        var_tmp = h_var_pri_le[:,:,None,None] * (A.conj().transpose(1,2)[:,None,:,:] @ (sigma_inv[:,:,:] @ A[:,None,:,:]))
        h_var_post_le = h_var_pri_le - (1/N) * var_tmp.diagonal(offset=0,dim1=-2,dim2=-1).sum(dim=-1)
        h_var_ext_le = 1/(1/h_var_post_le - 1/h_var_pri_le)
        h_mean_ext_le = h_var_ext_le[:,None,:] * (h_mean_post_le/h_var_post[:,None,:] - h_mean_pri_le/h_var_pri_le[:,None,:])

        # alpha C->H
        tmp_v = torch.abs((c_mean_ext_nle[:,:,None,:] @ torch.inv(torch.diag_embed(c_var_ext_nle)[:,None,:,:]) @ c_mean_ext_nle[:,:,:,None]).squeeze())
        tmp_vi = torch.abs((c_mean_ext_nle[:,:,None,:] @ torch.inv(torch.diag_embed(c_var_ext_nle + c_var)[:,None,:,:]) @ c_mean_ext_nle[:,:,:,None])).squeeze()
        tmp_v = torch.clamp(torch.exp(tmp_v),min=1e-20)
        tmp_vi = torch.clamp(torch.exp(tmp_vi),min=1e-20)
        pi_c = 1/(1 + tmp_v/tmp_vi)
        lambda_ch = pi_c * lambda_pri / (pi_c * lambda_pri + (1-pi_c)*(1-lambda_pri))

        # NLE of H
        tmp_v = torch.abs((h_mean_ext_le[:,:,None,:] @ torch.diag_embed(h_var_ext_le)[:,None,:,:] @ h_mean_ext_le[:,:,:,None])).squeeze()
        tmp_vi = torch.abs((h_mean_ext_le[:,:,None,:] @ torch.diag_embed(h_var_ext_le + h_var)[:,None,:,:] @ h_mean_ext_le[:,:,:,None])).squeeze()
        tmp_v = torch.clamp(torch.exp(tmp_v),min=1e-20)
        tmp_vi = torch.clamp(torch.exp(tmp_vi),min=1e-20)
        lambda_post_h = 1/(1 + ((1-lambda_ch) * tmp_v)/(lambda_ch * tmp_vi))
        gau_var = 1/(1/h_var + 1/h_var_ext_le)
        gau_mean = (h_var[:,None,:] * h_mean_ext_le) / (h_var + h_var_ext_le)[:,None,:]
        h_mean_post_nle = gau_mean * lambda_post_h[:,:,None]
        h_var_post_nle = (lambda_ch[:,:,None] * (gau_mean + gau_var[:,None,:]) - torch.abs(h_mean_post_nle) ** 2).mean(dim=1)
        h_var_ext_nle = 1/(1/h_var_post_nle - 1/h_var_ext_le)
        h_mean_ext_nle =  h_var_ext_nle[:,None,:] * (h_mean_post_nle/h_var_post_nle[:,None,:] - h_mean_ext_le/h_var_ext_le[:,None,:])

        # LE of C
        sigma = c_var_pri_le[:,:,None,None] * torch.eye(Q,device=device) + c_var_pri_le[:,:,None,None] * BBH[:,None,:,:] + noise_var * torch.eye(Q,device=device)
        sigma_inv = torch.linalg.inv(sigma)
        res = Y - torch.matmul(A, h_mean_ext_le) - torch.matmul(B, c_mean_pri_le)
        lmmse_tmp = (sigma_inv @ res.transpose(1,2)[:,:,:,None]).transpose(1,2).squeeze(-1)
        lmmse_tmp = c_var_pri_le[:,None,:] * (A.conj().transpose(1,2) @ lmmse_tmp)
        c_mean_post_le = c_mean_pri_le + lmmse_tmp
        var_tmp = c_var_pri_le[:,:,None,None] * (A.conj().transpose(1,2)[:,None,:,:] @ (sigma_inv[:,:,:] @ A[:,None,:,:]))
        c_var_post_le = c_var_pri_le - (1/N) * var_tmp.diagonal(offset=0,dim1=-2,dim2=-1).sum(dim=-1)
        c_var_ext_le = 1/(1/c_var_post_le - 1/c_var_pri_le)
        c_mean_ext_le = c_var_ext_le[:,None,:] * (c_mean_post_le/c_var_post_le[:,None,:] - c_mean_pri_le/c_var_pri_le[:,None,:])

        # alpha H->C
        tmp_v = torch.abs((h_mean_ext_nle[:,:,None,:] @ torch.diag_embed(h_var_ext_nle)[:,None,:,:] @ h_mean_ext_nle[:,:,:,None])).squeeze()
        tmp_vi = torch.abs((h_mean_ext_nle[:,:,None,:] @ torch.diag_embed(h_var_ext_nle + h_var)[:,None,:,:] @ h_mean_ext_nle[:,:,:,None])).squeeze()
        tmp_v = torch.clamp(torch.exp(tmp_v),min=1e-20)
        tmp_vi = torch.clamp(torch.exp(tmp_vi),min=1e-20)
        pi_h = 1/(1 + tmp_v/tmp_vi)
        lambda_hc = pi_h * lambda_pri / (pi_h * lambda_pri + (1-pi_h)*(1-lambda_pri))

        # alpha H->C
        tmp_v = torch.abs((c_mean_ext_le[:,:,None,:] @ torch.diag_embed(c_var_ext_le)[:,None,:,:] @ c_mean_ext_le[:,:,:,None])).squeeze()
        tmp_vi = torch.abs((c_mean_ext_le[:,:,None,:] @ torch.diag_embed(c_var_ext_le + h_var)[:,None,:,:] @ c_mean_ext_le[:,:,:,None])).squeeze()
        tmp_v = torch.clamp(torch.exp(tmp_v),min=1e-20)
        tmp_vi = torch.clamp(torch.exp(tmp_vi),min=1e-20)
        lambda_post_c = 1/(1 + ((1-lambda_hc) * tmp_v)/(lambda_hc * tmp_vi))
        gau_var = 1/(1/c_var + 1/c_var_ext_le)
        gau_mean = (c_var[:,None,:] * c_mean_ext_le) / (c_var + c_var_ext_le)[:,None,:]
        c_mean_post_nle = gau_mean * lambda_post_c[:,:,None]
        c_var_post_nle = (lambda_pri[:,:,None] * (gau_mean + gau_var[:,None,:]) - torch.abs(c_mean_post_nle) ** 2).mean(dim=1)
        c_var_ext_nle = 1/(1/c_var_post_nle - 1/c_var_ext_le)
        c_mean_ext_nle = c_var_ext_nle[:,None,:] * (c_mean_post_nle/c_var_post_nle[:,None,:] - c_mean_ext_le/c_var_ext_le[:,None,:])

        lambda_post = lambda_pri * pi_h * pi_c / (lambda_pri * pi_h * pi_c + (1 - lambda_pri) * (1 - pi_h) * (1 - pi_c))
        lambda_pri = lambda_post.mean(dim=1,keepdim=True).expand_as(lambda_pri)

        h_var_pri_le = h_var_ext_nle
        c_var_pri_le = c_var_ext_nle

    return lambda_post


if __name__ == "__main__":
    print('a')
    mean = 0.001*(torch.ones((32,128,256)) + 1j * torch.ones((32,128,256)))
    var = 0.9*torch.ones((32,256),dtype=torch.complex64)
    pdf = batch_multinormal_zero_pdf(mean,var)
    print(f'{pdf}')


