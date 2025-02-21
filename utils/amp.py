from torch import nn
import torch.nn.functional as F
import torch
from torch.nn.parameter import Parameter
import numpy as np
import scipy.stats as st
import math

torch.manual_seed(777)

def AMP_DA(y, X, H, args):
    H = H.cuda()
    y = y.cuda()
    X = X.cuda()

    N_RAs = H.shape[0]
    N_UEs = H.shape[1]
    N_dim = y.shape[1]
    N_M = y.shape[2]
    tol = 1e-5
    exx = 1e-10
    damp = 0.3
    alphabet = torch.arange(0.0, args.num_users * args.frac + 1, 1)
    M = len(alphabet) - 1

    lam = N_RAs / N_UEs
    c = torch.arange(0.01, 10, 10 / 1024)
    rho = (1 - 2 * N_UEs * ((1 + c ** 2) * st.norm.cdf(-c) - c * st.norm.pdf(c)) / N_RAs) / (
            1 + c ** 2 - 2 * ((1 + c ** 2) * st.norm.cdf(-c) - c * st.norm.pdf(c)))
    alpha = lam * torch.max(rho) * torch.ones((N_UEs, N_dim))
    x_hat0 = (alpha * torch.sum(alphabet) / M * torch.ones((N_UEs, N_dim)))[:, :, None].repeat(1, 1, N_M)
    x_hat = (x_hat0 + 1j * x_hat0).cuda()
    var_hat = torch.ones((N_UEs, N_dim, N_M)).cuda()

    V = torch.ones((N_RAs, N_dim, N_M)).cuda()
    V_new = torch.ones((N_RAs, N_dim, N_M)).cuda()
    Z_new = y.clone()
    sigma2 = 100
    t = 1
    Z = y.clone()
    maxIte = 50
    MSE = torch.zeros(maxIte)
    MSE[0] = 100
    hvar = (torch.norm(y) ** 2 - N_RAs * sigma2) / (N_dim * lam * torch.max(rho) * torch.norm(H) ** 2)
    hmean = 0
    alpha_new = torch.ones((N_UEs, N_dim, N_M))
    x_hat_new = torch.ones((N_UEs, N_dim, N_M)) + 1j * torch.ones((N_UEs, N_dim, N_M))
    var_hat_new = torch.ones((N_UEs, N_dim, N_M))

    hvarnew = torch.zeros(N_M)
    hmeannew = torch.zeros(N_M) + 1j * torch.zeros(N_M)
    sigma2new = torch.zeros(N_M)

    alphabet = alphabet.cuda()
    alpha = alpha.cuda()
    while t < maxIte:
        x_hat_pre = x_hat.clone()
        for i in range(N_M):
            V_new[:, :, i] = torch.abs(H) ** 2 @ var_hat[:, :, i]
            Z_new[:, :, i] = H @ x_hat[:, :, i] - ((y[:, :, i] - Z[:, :, i]) / (sigma2 + V[:, :, i])) * V_new[:, :, i]  # + 1e-8

            Z_new[:, :, i] = damp * Z[:, :, i] + (1 - damp) * Z_new[:, :, i]
            V_new[:, :, i] = damp * V[:, :, i] + (1 - damp) * V_new[:, :, i]

            var1 = (torch.abs(H) ** 2).T @ (1 / (sigma2 + V_new[:, :, i]))
            var2 = H.conj().T @ ((y[:, :, i] - Z_new[:, :, i]) / (sigma2 + V_new[:, :, i]))

            Ri = var2 / (var1) + x_hat[:, :, i]
            Vi = 1 / (var1)

            sigma2new[i] = ((torch.abs(y[:, :, i] - Z_new[:, :, i]) ** 2) / (
                        torch.abs(1 + V_new[:, :, i] / sigma2) ** 2) + sigma2 * V_new[:, :, i] / (
                                        V_new[:, :, i] + sigma2)).mean()

            if i == 0:
                r_s = Ri[None, :, :].repeat(M + 1, 1, 1) - alphabet[:, None, None].repeat(1, N_UEs, N_dim)
                pf8 = torch.exp(-(torch.abs(r_s) ** 2 / Vi)) / Vi / np.pi
                pf7 = torch.zeros((M + 1, N_UEs, N_dim)).cuda()
                pf7[0, :, :] = pf8[0, :, :] * (torch.ones((N_UEs, N_dim)).cuda() - alpha)
                pf7[1:, :, :] = pf8[1:, :, :] * (alpha / M)
                del pf8
                PF7 = torch.sum(pf7, axis=0)
                pf6 = pf7 / PF7
                del pf7, PF7
                AAA = alphabet[None, :, None].repeat(N_dim, 1, 1)
                BBB = torch.permute(pf6,(2,1,0))
                x_hat_new[:, :, i] = (torch.einsum("ijk,ikn->ijn", BBB, AAA).squeeze(-1)).T
                del AAA
                alphabet2 = alphabet ** 2
                AAA2 = alphabet2[None, :, None].repeat(N_dim, 1, 1)
                var_hat_new[:, :, i] = (torch.einsum("ijk,ikn->ijn", BBB, AAA2).squeeze(-1)).T.cpu() - torch.abs(
                    x_hat_new[:, :, i]) ** 2
                del AAA2
                alpha_new[:, :, i] = torch.clamp(torch.sum(pf6[1:, :, :], axis=0), exx, 1 - exx)
                del pf6
            else:
                A = (hvar * Vi) / (Vi + hvar)
                B = (hvar * Ri + Vi * hmean) / (Vi + hvar)
                lll = torch.log(Vi / (Vi + hvar)) / 2 + torch.abs(Ri) ** 2 / 2 / Vi - torch.abs(Ri - hmean) ** 2 / 2 / (
                            Vi + hvar)
                pai = torch.clamp(alpha / (alpha + (1 - alpha) * torch.exp(-lll)), exx, 1 - exx, out=None)
                x_hat_new[:, :, i] = pai * B
                var_hat_new[:, :, i] = (pai * (torch.abs(B) ** 2 + A)).cpu() - torch.abs(x_hat_new[:, :, i]) ** 2
                # mean update
                hmeannew[i] = (torch.sum(pai * B, axis=0) / torch.sum(pai, axis=0)).mean()
                # variance update
                hvarnew[i] = (torch.sum(pai * (torch.abs(hmean - B) ** 2 + Vi), axis=0) / torch.sum(pai, axis=0)).mean()
                # activity indicator update
                alpha_new[:, :, i] = torch.clamp(pai, exx, 1 - exx)
        if N_M > 1:
            hvar = hvarnew[1:].mean()
            hmean = hmeannew[1:].mean()
        sigma2 = sigma2new.mean()
        alpha = (torch.sum(alpha_new, axis=2) / N_M).cuda()
        # alpha = alpha_new
        III = x_hat_pre.cpu() - x_hat_new
        NMSE_iter = torch.sum(torch.abs(III) ** 2) / torch.sum(torch.abs(x_hat_new) ** 2)
        # del III
        MSE[t] = torch.sum(torch.abs(y - torch.permute(
            torch.einsum("ijk,ikn->ijn", torch.permute(x_hat, (2, 1, 0)), H.T[None, :, :].repeat(N_M, 1, 1)),
            (2, 1, 0))) ** 2) / N_RAs / N_dim / N_M

        x_hat = x_hat_new.cuda().clone()
        if t > 15 and MSE[t] >= MSE[t - 1]:
            x_hat = x_hat_pre.clone()
            break

        NMSE = 10 * torch.log10(torch.sum(torch.abs(x_hat[:, :, :] - X[:, :]) ** 2) / torch.sum(torch.abs(X[:, :]) ** 2))

        var_hat = var_hat_new.cuda().clone()
        # alpha = alpha_new
        V = V_new.clone()
        Z = Z_new.clone()
        t = t + 1
    return x_hat, var_hat, alpha, t, NMSE, sigma2

# def GMMV_AMP(Y, Phi, damp, niter, tol, device=None):
#     """
#     GMMV-AMP algorithm for GMMV CS problem (estimating 3D matrix)

#     Args:
#         Y: received signal, shape (P, M, Q)
#         Phi: measurement matrix, shape (P, M, N)
#         Pn: noise variance
#         damp: damping factor
#         niter: number of iterations
#         tol: termination threshold
#         device: torch.device, if None will use same device as input tensors

#     Returns:
#         Xhat: estimated matrix
#         temppp: belief indicators after thresholding
#         iter: number of iterations performed
#         NMSE_re: Normalized Mean Square Error
#     """
#     if device is None:
#         device = Y.device

#     Y = Y.to(device)
#     Phi = Phi.to(device)
#     # Get dimensions
#     P, M, Q = Y.shape
#     _,_, N = Phi.shape

#     # Calculate alpha and initial lambda
#     alpha = M / N
#     alpha_grid = torch.linspace(0, 10, 1024, device=device)
    
#     # Normal CDF and PDF
#     normal_cdf = 0.5 * (1 + torch.erf(-alpha_grid / math.sqrt(2)))
#     normal_pdf = 1 / math.sqrt(2 * math.pi) * torch.exp(-alpha_grid ** 2 / 2)
    
#     rho_SE = (1 - 2 / alpha * ((1 + alpha_grid ** 2) * normal_cdf - alpha_grid * normal_pdf)) / \
#              (1 + alpha_grid ** 2 - 2 * ((1 + alpha_grid ** 2) * normal_cdf - alpha_grid * normal_pdf))
    
#     # Initialize parameters
#     lambda_init = alpha * torch.max(rho_SE)
#     lambda_tensor = torch.full((P, N, Q), torch.tensor(lambda_init), device=device)
#     SNR0 = 100
#     nvar = torch.mean((torch.abs(Y)**2)/(SNR0+1),dim=(0,1,2))
#     xmean = torch.zeros((P, N, Q),dtype=torch.complex64) # eps equivalent
#     xvar = (torch.sum((torch.abs(Y)**2),dim=1) - N * nvar)[:,None,:].expand((P,N,Q))/torch.mean(torch.abs(Phi)**2, dim=(0,1,2))
#     xmean = xmean.cuda()
#     xvar = xvar.cuda()

#     # Initialize tensors
#     Xhat = xmean
#     v = xvar
#     V = torch.ones(P, M, Q, device=device)
#     Z = Y.clone()

#     # Allocate memory for intermediate variables
#     D = torch.zeros(P, N, Q, device=device)
#     C = torch.zeros(P, N, Q, device=device)
#     L_cal = torch.zeros(P, N, Q, device=device)
#     pai = torch.zeros(P, N, Q, device=device)
#     A = torch.zeros(P, N, Q, device=device)
#     B = torch.zeros(P, N, Q, device=device)
#     Yhat = torch.zeros(P, M, Q, device=device)

#     # AMP iteration
#     for iter in range(niter):
#         Xhat_pre = Xhat.clone()
#         V_pre = V.clone()

#         # Factor node update
#         V = damp * V_pre + (1 - damp) * torch.matmul(torch.abs(Phi) ** 2, v)
#         Z = damp * Z + (1 - damp) * (torch.matmul(Phi, Xhat) - V*(Y - Z) / (nvar + V_pre))

#         # Variable node update
#         D = 1 / torch.matmul((torch.abs(Phi) ** 2).transpose(1,2), 1/ (nvar + V))
#         C = Xhat + D * torch.matmul(Phi.transpose(1,2).conj(), (Y - Z) / (nvar + V))

#         # Compute posterior mean and variance
#         L_cal = 0.5 * (torch.log(D / (D + xvar)) + torch.abs(C) ** 2 / D -
#                        torch.abs(C - xmean) ** 2 / (D + xvar))
#         pai = lambda_tensor / (lambda_tensor + (1 - lambda_tensor) * torch.exp(-L_cal))
#         A = (xvar * C + xmean * D) / (D + xvar)
#         B = (xvar * D) / (xvar + D)
#         Xhat = pai * A
#         v = pai * (torch.abs(A) ** 2 + B) - torch.abs(Xhat) ** 2

#         # EM update
#         nvar = ((torch.abs(Y- Z) ** 2) / (torch.abs(1 + V/ nvar) ** 2) + nvar * V / (V + nvar)).mean()
#         xmean = torch.sum(pai * A, axis=0, keepdim=True).expand_as(pai) / torch.sum(pai, axis=0, keepdim=True).expand_as(pai)
#         xvar = torch.sum(pai * (torch.abs(xmean - A) ** 2 + B), axis=0, keepdim=True).expand_as(pai) / torch.sum(pai, axis=0, keepdim=True).expand_as(pai)
#         # Refine sparsity ratio
#         # pai_tmp = torch.mean(pai, dim=2)
#         pai_tmp = (torch.sum(pai, dim=2)[:,:,None].expand_as(pai) - pai)/(Q-1)
#         lambda_tensor = pai_tmp

#         # Reconstruct received signal
#         Yhat = torch.matmul(Phi, Xhat)
#         # Check stopping criteria
#         NMSE_iter = torch.norm(Xhat - Xhat_pre) ** 2 / torch.norm(Xhat_pre) ** 2
#         NMSE_re = torch.norm(Y - Yhat) ** 2 / torch.norm(Y) ** 2

#         if NMSE_iter < tol or NMSE_re < tol:
#             break

#     # Final processing
#     temppp = torch.mean(pai_tmp, dim=(0,2))
#     temppp[temppp > 0.5] = 1
#     temppp[temppp <= 0.5] = 0

#     return Xhat, temppp, iter + 1, NMSE_re

def GMMV_AMP(Y, Phi, damp, niter, tol, device=None):
    """
    GMMV-AMP algorithm for GMMV CS problem (estimating 3D matrix)

    Args:
        Y: received signal, shape (P, M, Q)
        Phi: measurement matrix, shape (P, M, N)
        Pn: noise variance
        damp: damping factor
        niter: number of iterations
        tol: termination threshold
        device: torch.device, if None will use same device as input tensors

    Returns:
        Xhat: estimated matrix
        temppp: belief indicators after thresholding
        iter: number of iterations performed
        NMSE_re: Normalized Mean Square Error
    """
    if device is None:
        device = Y.device

    Y = Y.to(device)
    Phi = Phi.to(device)
    # Get dimensions
    P, M, Q = Y.shape
    _,_, N = Phi.shape

    # Calculate alpha and initial lambda
    alpha = M / N
    alpha_grid = torch.linspace(0, 10, 1024, device=device)
    
    # Normal CDF and PDF
    normal_cdf = 0.5 * (1 + torch.erf(-alpha_grid / math.sqrt(2)))
    normal_pdf = 1 / math.sqrt(2 * math.pi) * torch.exp(-alpha_grid ** 2 / 2)
    
    rho_SE = (1 - 2 / alpha * ((1 + alpha_grid ** 2) * normal_cdf - alpha_grid * normal_pdf)) / \
             (1 + alpha_grid ** 2 - 2 * ((1 + alpha_grid ** 2) * normal_cdf - alpha_grid * normal_pdf))
    
    # Initialize parameters
    lambda_init = alpha * torch.max(rho_SE)
    lambda_tensor = torch.full((P, N, Q), torch.tensor(lambda_init), device=device)
    SNR0 = 100
    nvar = torch.mean((torch.abs(Y)**2)/(SNR0+1),dim=(0,1,2))
    xmean = torch.zeros((P, N, Q),dtype=torch.complex64) # eps equivalent
    xvar = (torch.sum((torch.abs(Y)**2),dim=1) - N * nvar)[:,None,:].expand((P,N,Q))/torch.mean(torch.abs(Phi)**2, dim=(0,1,2))
    xmean = xmean.cuda()
    xvar = xvar.cuda()

    # Initialize tensors
    Xhat = xmean
    v = xvar
    V = torch.ones(P, M, Q, device=device)
    Z = Y.clone()

    # Allocate memory for intermediate variables
    D = torch.zeros(P, N, Q, device=device)
    C = torch.zeros(P, N, Q, device=device)
    L_cal = torch.zeros(P, N, Q, device=device)
    pai = torch.zeros(P, N, Q, device=device)
    A = torch.zeros(P, N, Q, device=device)
    B = torch.zeros(P, N, Q, device=device)
    Yhat = torch.zeros(P, M, Q, device=device)

    # AMP iteration
    for iter in range(niter):
        Xhat_pre = Xhat.clone()
        V_pre = V.clone()

        # Factor node update
        V = damp * V_pre + (1 - damp) * torch.matmul(torch.abs(Phi) ** 2, v)
        Z = damp * Z + (1 - damp) * (torch.matmul(Phi, Xhat) - V*(Y - Z) / (nvar + V_pre))

        # Variable node update
        D = 1 / torch.matmul((torch.abs(Phi) ** 2).transpose(1,2), 1/ (nvar + V))
        C = Xhat + D * torch.matmul(Phi.transpose(1,2).conj(), (Y - Z) / (nvar + V))

        # Compute posterior mean and variance
        L_cal = 0.5 * (torch.log(D / (D + xvar)) + torch.abs(C) ** 2 / D -
                       torch.abs(C - xmean) ** 2 / (D + xvar))
        pai = lambda_tensor / (lambda_tensor + (1 - lambda_tensor) * torch.exp(-L_cal))
        A = (xvar * C + xmean * D) / (D + xvar)
        B = (xvar * D) / (xvar + D)
        Xhat = pai * A
        v = pai * (torch.abs(A) ** 2 + B) - torch.abs(Xhat) ** 2

        # EM update
        nvar = ((torch.abs(Y- Z) ** 2) / (torch.abs(1 + V/ nvar) ** 2) + nvar * V / (V + nvar)).mean()
        xmean = torch.sum(pai * A, axis=0, keepdim=True).expand_as(pai) / torch.sum(pai, axis=0, keepdim=True).expand_as(pai)
        xvar = torch.sum(pai * (torch.abs(xmean - A) ** 2 + B), axis=0, keepdim=True).expand_as(pai) / torch.sum(pai, axis=0, keepdim=True).expand_as(pai)
        # Refine sparsity ratio
        # pai_tmp = torch.mean(pai, dim=2, keepdim=True).expand_as(pai)
        # pai_tmp = (torch.sum(pai, dim=(0,2),keepdim=True).expand_as(pai) #- pai)/(N*Q-1)
        lambda_tensor = pai

        # Reconstruct received signal
        Yhat = torch.matmul(Phi, Xhat)
        # Check stopping criteria
        NMSE_iter = torch.norm(Xhat - Xhat_pre) ** 2 / torch.norm(Xhat_pre) ** 2
        NMSE_re = torch.norm(Y - Yhat) ** 2 / torch.norm(Y) ** 2

        if NMSE_iter < tol or NMSE_re < tol:
            break

    # Final processing
    # temppp = torch.mean(pai_tmp, dim=(0,2))
    # temppp[temppp > 0.5] = 1
    # temppp[temppp <= 0.5] = 0

    return Xhat, lambda_tensor, iter + 1, NMSE_re

def GMMV_AMP_blk(Y, Phi, damp, niter, tol, device=None):
    """
    GMMV-AMP algorithm for GMMV CS problem (estimating 3D matrix)

    Args:
        Y: received signal, shape (P, M, Q)
        Phi: measurement matrix, shape (P, M, N)
        Pn: noise variance
        damp: damping factor
        niter: number of iterations
        tol: termination threshold
        device: torch.device, if None will use same device as input tensors

    Returns:
        Xhat: estimated matrix
        temppp: belief indicators after thresholding
        iter: number of iterations performed
        NMSE_re: Normalized Mean Square Error
    """
    if device is None:
        device = Y.device

    Y = Y.to(device)
    Phi = Phi.to(device)
    # Get dimensions
    P, M, Q = Y.shape
    _,_, N = Phi.shape

    # Calculate alpha and initial lambda
    alpha = M / N
    alpha_grid = torch.linspace(0, 10, 1024, device=device)
    
    # Normal CDF and PDF
    normal_cdf = 0.5 * (1 + torch.erf(-alpha_grid / math.sqrt(2)))
    normal_pdf = 1 / math.sqrt(2 * math.pi) * torch.exp(-alpha_grid ** 2 / 2)
    
    rho_SE = (1 - 2 / alpha * ((1 + alpha_grid ** 2) * normal_cdf - alpha_grid * normal_pdf)) / \
             (1 + alpha_grid ** 2 - 2 * ((1 + alpha_grid ** 2) * normal_cdf - alpha_grid * normal_pdf))
    
    # Initialize parameters
    lambda_init = alpha * torch.max(rho_SE)
    lambda_tensor = torch.full((P, N, Q), torch.tensor(lambda_init), device=device)
    SNR0 = 100
    nvar = torch.mean((torch.abs(Y)**2)/(SNR0+1),dim=(0,1,2))
    xmean = torch.zeros((P, N, Q),dtype=torch.complex64) # eps equivalent
    xvar = (torch.sum((torch.abs(Y)**2),dim=1) - N * nvar)[:,None,:].expand((P,N,Q))/torch.mean(torch.abs(Phi)**2, dim=(0,1,2))
    xmean = xmean.cuda()
    xvar = xvar.cuda()

    # Initialize tensors
    Xhat = xmean
    v = xvar
    V = torch.ones(P, M, Q, device=device)
    Z = Y.clone()

    # Allocate memory for intermediate variables
    D = torch.zeros(P, N, Q, device=device)
    C = torch.zeros(P, N, Q, device=device)
    L_cal = torch.zeros(P, N, Q, device=device)
    pai = torch.zeros(P, N, Q, device=device)
    A = torch.zeros(P, N, Q, device=device)
    B = torch.zeros(P, N, Q, device=device)
    Yhat = torch.zeros(P, M, Q, device=device)

    # AMP iteration
    for iter in range(niter):
        Xhat_pre = Xhat.clone()
        V_pre = V.clone()

        # Factor node update
        V = damp * V_pre + (1 - damp) * torch.matmul(torch.abs(Phi) ** 2, v)
        Z = damp * Z + (1 - damp) * (torch.matmul(Phi, Xhat) - V*(Y - Z) / (nvar + V_pre))

        # Variable node update
        D = 1 / torch.matmul((torch.abs(Phi) ** 2).transpose(1,2), 1/ (nvar + V))
        C = Xhat + D * torch.matmul(Phi.transpose(1,2).conj(), (Y - Z) / (nvar + V))

        # Compute posterior mean and variance
        L_cal = 0.5 * (torch.log(D / (D + xvar)) + torch.abs(C) ** 2 / D -
                       torch.abs(C - xmean) ** 2 / (D + xvar))
        pai = lambda_tensor / (lambda_tensor + (1 - lambda_tensor) * torch.exp(-L_cal))
        A = (xvar * C + xmean * D) / (D + xvar)
        B = (xvar * D) / (xvar + D)
        Xhat = pai * A
        v = pai * (torch.abs(A) ** 2 + B) - torch.abs(Xhat) ** 2

        # EM update
        nvar = ((torch.abs(Y- Z) ** 2) / (torch.abs(1 + V/ nvar) ** 2) + nvar * V / (V + nvar)).mean()
        xmean = torch.sum(pai * A, axis=0, keepdim=True).expand_as(pai) / torch.sum(pai, axis=0, keepdim=True).expand_as(pai)
        xvar = torch.sum(pai * (torch.abs(xmean - A) ** 2 + B), axis=0, keepdim=True).expand_as(pai) / torch.sum(pai, axis=0, keepdim=True).expand_as(pai)
        # Refine sparsity ratio
        # pai_tmp = torch.mean(pai, dim=2)
        # pai_tmp = (torch.sum(pai, dim=2)[:,:,None].expand_as(pai) - pai)/(Q-1)
        pai_reshape = pai.view(P,int(N/2),2,Q)
        pai_tmp = torch.mean(pai_reshape, dim=(2,3))
        pai_tmp = pai_tmp[:,:,None,None].expand_as(pai_reshape)
        pai_tmp = pai_tmp.reshape_as(pai)
        lambda_tensor = pai_tmp

        # Reconstruct received signal
        Yhat = torch.matmul(Phi, Xhat)
        # Check stopping criteria
        NMSE_iter = torch.norm(Xhat - Xhat_pre) ** 2 / torch.norm(Xhat_pre) ** 2
        NMSE_re = torch.norm(Y - Yhat) ** 2 / torch.norm(Y) ** 2

        if NMSE_iter < tol or NMSE_re < tol:
            break

    # Final processing
    temppp = torch.mean(pai_tmp, dim=(0,2))
    temppp[temppp > 0.5] = 1
    temppp[temppp <= 0.5] = 0

    return Xhat, temppp, iter + 1, NMSE_re

def GMMV_AMP_cluster(Y, Phi, damp, niter, tol, device=None):
    """
    GMMV-AMP algorithm for GMMV CS problem (estimating 3D matrix)

    Args:
        Y: received signal, shape (P, M, Q)
        Phi: measurement matrix, shape (P, M, N)
        Pn: noise variance
        damp: damping factor
        niter: number of iterations
        tol: termination threshold
        device: torch.device, if None will use same device as input tensors

    Returns:
        Xhat: estimated matrix
        temppp: belief indicators after thresholding
        iter: number of iterations performed
        NMSE_re: Normalized Mean Square Error
    """
    if device is None:
        device = Y.device

    Y = Y.to(device)
    Phi = Phi.to(device)
    # Get dimensions
    P, M, Q = Y.shape
    _,_, N = Phi.shape

    # Calculate alpha and initial lambda
    alpha = M / N
    alpha_grid = torch.linspace(0, 10, 1024, device=device)
    
    # Normal CDF and PDF
    normal_cdf = 0.5 * (1 + torch.erf(-alpha_grid / math.sqrt(2)))
    normal_pdf = 1 / math.sqrt(2 * math.pi) * torch.exp(-alpha_grid ** 2 / 2)
    
    rho_SE = (1 - 2 / alpha * ((1 + alpha_grid ** 2) * normal_cdf - alpha_grid * normal_pdf)) / \
             (1 + alpha_grid ** 2 - 2 * ((1 + alpha_grid ** 2) * normal_cdf - alpha_grid * normal_pdf))
    
    # Initialize parameters
    lambda_init = 0.1 * alpha * torch.max(rho_SE)
    # lambda_init = 1e-3
    lambda_tensor = lambda_init * torch.ones((P, N, Q), device=device)
    SNR0 = 100
    nvar = torch.mean((torch.abs(Y)**2)/(SNR0+1),dim=(0,1,2))
    xmean = torch.zeros((P, N, Q),dtype=torch.complex64) # eps equivalent
    xvar_init = (torch.sum((torch.abs(Y)**2),dim=(0,1,2)) - M * nvar)/torch.mean(torch.abs(Phi)**2, dim=(0,1))
    # xvar_init = 1
    # xvar = xvar_init*torch.ones(P,N,Q,device=device)
    xvar = xvar_init[None,:,None].expand_as(xmean)
    xmean = xmean.cuda()
    xvar = xvar.cuda()

    # Initialize tensors
    Xhat = xmean
    v = xvar
    V = torch.ones(P, M, Q, device=device)
    Z = Y.clone()

    # Allocate memory for intermediate variables
    D = torch.zeros(P, N, Q, device=device)
    C = torch.zeros(P, N, Q, device=device)
    L_cal = torch.zeros(P, N, Q, device=device)
    pai = torch.zeros(P, N, Q, device=device)
    A = torch.zeros(P, N, Q, device=device)
    B = torch.zeros(P, N, Q, device=device)
    Yhat = torch.zeros(P, M, Q, device=device)
    supp = torch.zeros(P, N, device=device)

    # AMP iteration
    for iter in range(niter):
        Xhat_pre = Xhat.clone()
        V_pre = V.clone()

        # Factor node update
        V = damp * V_pre + (1 - damp) * torch.matmul(torch.abs(Phi) ** 2, v)
        Z = damp * Z + (1 - damp) * (torch.matmul(Phi, Xhat) - V*(Y - Z) / (nvar + V_pre))

        # Variable node update
        D = 1 / torch.matmul((torch.abs(Phi) ** 2).transpose(1,2), 1/ (nvar + V))
        C = Xhat + D * torch.matmul(Phi.transpose(1,2).conj(), (Y - Z) / (nvar + V))

        # Compute posterior mean and variance
        L_cal = 0.5 * (torch.log(D / (D + xvar)) + torch.abs(C) ** 2 / D -
                       torch.abs(C - xmean) ** 2 / (D + xvar))
        pai = lambda_tensor / (lambda_tensor + (1 - lambda_tensor) * torch.exp(-L_cal))
        pai = torch.clamp(pai, min=1e-20,max=1-1e-20)
        A = (xvar * C + xmean * D) / (D + xvar)
        B = (xvar * D) / (xvar + D)
        Xhat = pai * A
        v = pai * (torch.abs(A) ** 2 + B) - torch.abs(Xhat) ** 2

        # EM update
        nvar = ((torch.abs(Y- Z) ** 2) / (torch.abs(1 + V/ nvar) ** 2) + nvar * V / (V + nvar)).mean()
        xmean = torch.sum(pai * A, axis=0, keepdim=True).expand_as(pai) / torch.sum(pai, axis=0, keepdim=True).expand_as(pai)
        xvar = torch.sum(pai * (torch.abs(xmean - A) ** 2 + B), axis=0, keepdim=True).expand_as(pai) / torch.sum(pai, axis=0, keepdim=True).expand_as(pai)
        # xmean = (torch.sum(pai * A, axis=0) / torch.sum(pai, axis=0)).mean()
        # xvar = (torch.sum(pai * (torch.abs(xmean - A) ** 2 + B), axis=0) / torch.sum(pai, axis=0)).mean()
        # Refine sparsity ratio
        # pai = torch.mean(pai, dim=2)
        pai_nn_left = torch.zeros(pai.shape).cuda()
        pai_nn_right = torch.zeros(pai.shape).cuda()
        pai_nn_front = torch.zeros(pai.shape).cuda()
        pai_nn_back = torch.zeros(pai.shape).cuda()
        pai_nn_front = torch.cat((pai[-1,:,:].unsqueeze(0), pai[:-1, :,:]), dim=0)
        pai_nn_back = torch.cat((pai[1:,:,:], pai[0, :,:].unsqueeze(0)), dim=0)
        
        pai_nn_left = torch.cat((pai[:,:,-1].unsqueeze(-1), pai[:, :,:-1]), dim=2)
        pai_nn_right = torch.cat((pai[:, :, 1:],pai[:,:,0].unsqueeze(-1)), dim=2)

        pai = (pai_nn_left + pai_nn_right + pai_nn_front + pai_nn_back)/4
        # pai_tmp = pai
        lambda_tensor = pai

        # Reconstruct received signal
        Yhat = torch.matmul(Phi, Xhat)
        # Check stopping criteria
        NMSE_iter = torch.norm(Xhat - Xhat_pre) ** 2 / torch.norm(Xhat_pre) ** 2
        NMSE_re = torch.norm(Y - Yhat) ** 2 / torch.norm(Y) ** 2

        if NMSE_iter < tol or NMSE_re < tol:
            break

    # Final processing
    # temppp = torch.mean(pai_tmp, dim=(0,2))
    # temppp[temppp > 0.5] = 1
    # temppp[temppp <= 0.5] = 0

    
    # pai_mean = torch.mean(pai, dim=2)
    # supp[pai_mean > 0.5] = True
    # supp[pai_mean < 0.5] = False

    return Xhat, lambda_tensor, iter + 1, NMSE_re

def GMMV_AMP_FG(Y, Phi, damp, niter, tol, num_ue, num_idx, device=None):
    """
    GMMV-AMP algorithm for GMMV CS problem (estimating 3D matrix)

    Args:
        Y: received signal, shape (P, M, Q)
        Phi: measurement matrix, shape (P, M, N)
        Pn: noise variance
        damp: damping factor
        niter: number of iterations
        tol: termination threshold
        device: torch.device, if None will use same device as input tensors

    Returns:
        Xhat: estimated matrix
        temppp: belief indicators after thresholding
        iter: number of iterations performed
        NMSE_re: Normalized Mean Square Error
    """
    if device is None:
        device = Y.device

    Y = Y.to(device)
    Phi = Phi.to(device)
    # Get dimensions
    P, M, Q = Y.shape
    _, N = Phi.shape

    # Calculate alpha and initial lambda
    alpha = M / N
    alpha_grid = torch.linspace(0, 10, 1024, device=device)
    
    # Normal CDF and PDF
    normal_cdf = 0.5 * (1 + torch.erf(-alpha_grid / math.sqrt(2)))
    normal_pdf = 1 / math.sqrt(2 * math.pi) * torch.exp(-alpha_grid ** 2 / 2)
    
    rho_SE = (1 - 2 / alpha * ((1 + alpha_grid ** 2) * normal_cdf - alpha_grid * normal_pdf)) / \
             (1 + alpha_grid ** 2 - 2 * ((1 + alpha_grid ** 2) * normal_cdf - alpha_grid * normal_pdf))
    
    # Initialize parameters
    lambda_init = alpha * torch.max(rho_SE)
    lambda_tensor = torch.full((P, num_ue, Q), torch.tensor(lambda_init), device=device)
    SNR0 = 100
    nvar = torch.mean((torch.abs(Y)**2)/(SNR0+1),dim=(0,1,2))
    xmean = torch.zeros((P, N, Q),dtype=torch.complex64) # eps equivalent
    xvar_init = (torch.sum((torch.abs(Y)**2),dim=1) - M * nvar)[:,None,:].expand((P,N,Q))/torch.mean(torch.abs(Phi)**2, dim=(0,1))
    xvar = xvar_init * torch.ones((P,N,Q), device=device)
    xmean = xmean.cuda()

    # Initialize tensors
    Xhat = xmean
    v = xvar
    V = torch.ones(P, M, Q, device=device)
    Z = Y.clone()

    # Allocate memory for intermediate variables
    D = torch.zeros(P, N, Q, device=device)
    C = torch.zeros(P, N, Q, device=device)
    L_cal = torch.zeros(P, N, Q, device=device)
    pai = torch.zeros(P, N, Q, device=device)
    A = torch.zeros(P, N, Q, device=device)
    B = torch.zeros(P, N, Q, device=device)
    Yhat = torch.zeros(P, M, Q, device=device)

    # AMP iteration
    for iter in range(niter):
        Xhat_pre = Xhat.clone()
        V_pre = V.clone()

        # Factor node update
        V = damp * V_pre + (1 - damp) * torch.matmul(torch.abs(Phi) ** 2, v)
        Z = damp * Z + (1 - damp) * (torch.matmul(Phi, Xhat) - V*(Y - Z) / (nvar + V_pre))

        # Variable node update
        D = 1 / torch.matmul((torch.abs(Phi) ** 2).T, 1/ (nvar + V))
        C = Xhat + D * torch.matmul(Phi.T.conj(), (Y - Z) / (nvar + V))

        # Compute posterior mean and variance
        # theta = (1/(torch.pi*D)) * torch.exp(-torch.abs(C)**2/D)
        # phi = (1/(torch.pi*(D+xvar))) * torch.exp(-torch.abs(C - xmean)**2/(D + xvar))

        # theta_temp = theta.reshape((P, num_ue, num_idx, Q))
        # phi_tmp = phi.reshape((P, num_ue, num_idx, Q))
        # phi_prod = torch.prod(phi_tmp, dim=2, keepdim=True)/phi_tmp
        # a_factor = torch.sum(theta_temp * phi_prod, dim=2)
        # a_factor = torch.clamp(a_factor,min=1e-20,max=1-1e-20)

        L_cal = 0.5 * (torch.log(D / (D + xvar)) + torch.abs(C) ** 2 / D -
                       torch.abs(C - xmean) ** 2 / (D + xvar))
        L_tmp = L_cal.reshape((P, num_ue, num_idx, Q))
        exp_L_tmp =  torch.exp(L_tmp)
        exp_L_tmp = torch.clamp(exp_L_tmp, min=1e-20, max=1e10)
        exp_L_sum = torch.sum(exp_L_tmp, dim=2)
        rec_exp_L_sum = 1/exp_L_sum

        pai = lambda_tensor[:,:,None,:] * exp_L_tmp / ((exp_L_sum*(lambda_tensor + num_idx* (1 - lambda_tensor) * rec_exp_L_sum))[:,:,None,:])
        pai = pai.reshape(D.shape)
        pai = torch.clamp(pai, min=1e-20, max=1-1e-20)
        A = (xvar * C + xmean * D) / (D + xvar)
        B = (xvar * D) / (xvar + D)
        Xhat = pai * A
        v = pai * (torch.abs(A) ** 2 + B) - torch.abs(Xhat) ** 2

        # EM update
        pai_tmp = pai.reshape(P, num_ue, num_idx, Q)
        lambda_tensor = torch.mean(1/(1 + 1/torch.sum(pai_tmp / (1 - pai_tmp),dim=2)),dim=-1,keepdim=True).expand(P, num_ue, Q)
        nvar = ((torch.abs(Y- Z) ** 2) / (torch.abs(1 + V/ nvar) ** 2) + nvar * V / (V + nvar)).mean()
        # xmean = (torch.sum(pai * A, axis=0) / torch.sum(pai, axis=0)).mean()
        # xvar = (torch.sum(pai * (torch.abs(xmean - A) ** 2 + B), axis=0) / torch.sum(pai, axis=0)).mean()
        xmean = torch.sum(pai * A, axis=0, keepdim=True).expand_as(pai) / torch.sum(pai, axis=0, keepdim=True).expand_as(pai)
        xvar = torch.sum(pai * (torch.abs(xmean - A) ** 2 + B), axis=0, keepdim=True).expand_as(pai) / torch.sum(pai, axis=0, keepdim=True).expand_as(pai)
        # Refine sparsity ratio
        # pai_tmp = torch.mean(pai, dim=2)
        # pai_tmp = (torch.sum(pai, dim=2)[:,:,None].expand_as(pai) - pai)/(Q-1)
        # lambda_tensor = pai_tmp


        # Reconstruct received signal
        Yhat = torch.matmul(Phi, Xhat)
        # Check stopping criteria
        NMSE_iter = torch.norm(Xhat - Xhat_pre) ** 2 / torch.norm(Xhat_pre) ** 2
        NMSE_re = torch.norm(Y - Yhat) ** 2 / torch.norm(Y) ** 2

        if NMSE_iter < tol or NMSE_re < tol:
            break

    # Final processing
    supp = torch.mean(pai, dim=(0,2))
    supp[supp > 0.5] = 1
    supp[supp <= 0.5] = 0

    return Xhat, supp, iter + 1, NMSE_re

def GMMV_AMP_Mtx(Y, Phi, Pn, damp, niter, tol, num_ind, num_ue, device=None):
    """
    GMMV-AMP algorithm for GMMV CS problem (estimating 3D matrix)
    Denoiser is degined based on factor graph

    Args:
        Y: received signal, shape (P, M, Q)
        Phi: measurement matrix, shape (P, M, N)
        Pn: noise variance
        damp: damping factor
        niter: number of iterations
        tol: termination threshold
        device: torch.device, if None will use same device as input tensors
        num_ind: number of indices
        num_ue:

    Returns:
        Xhat: estimated matrix
        temppp: belief indicators after thresholding
        iter: number of iterations performed
        NMSE_re: Normalized Mean Square Error
    """
    if device is None:
        device = Y.device

    Y = Y.to(device)
    Phi = Phi.to(device)
    # Get dimensions
    P, M, Q = Y.shape
    _, N = Phi.shape

    a_factor = 0.5 * torch.ones((num_ue), device=device)

    # Initialize parameters
    nvar = Pn
    mean_pri = torch.tensor(1e-16, device=device) + 1j* torch.tensor(1e-16, device=device) # eps equivalent
    var_pri = torch.tensor(1.0, device=device)

    # Initialize tensors
    x_hat = torch.zeros((P, N, Q), device=device,dtype=torch.complex64)
    v = torch.full((P, N, Q), var_pri, device=device)
    V = torch.ones(P, M, Q, device=device)
    Z = Y.clone()

    # Allocate memory for intermediate variables
    D = torch.zeros(P, N, Q, device=device)
    C = torch.zeros(P, N, Q, device=device)
    L_cal = torch.zeros(P, N, Q, device=device)
    pai = torch.zeros(P, N, Q, device=device)
    A = torch.zeros(P, N, Q, device=device)
    B = torch.zeros(P, N, Q, device=device)
    Yhat = torch.zeros(P, M, Q, device=device)

    # AMP iteration
    for iter in range(niter):
        Xhat_pre = x_hat.clone()
        V_pre = V.clone()

        # Factor node update
        V = damp * V_pre + (1 - damp) * torch.matmul(torch.abs(Phi) ** 2, v)
        Z = damp * Z + (1 - damp) * (torch.matmul(Phi, x_hat) - V*(Y - Z) / (nvar + V_pre))

        # Variable node update
        D = 1 / torch.matmul((torch.abs(Phi) ** 2).T, 1/ (nvar + V))
        C = x_hat + D * torch.matmul(Phi.T.conj(), (Y - Z) / (nvar + V))

        A = (var_pri * C) / (D + var_pri)
        B = (var_pri * D) / (var_pri + D)

        D_tmp = torch.reshape(D, (P, num_ue, num_ind, Q))
        gn_var = torch.mean(D_tmp, dim=(2,3))
        theta = var_pri / (var_pri + gn_var)

        psi = torch.log(1 + var_pri/gn_var)
        inner_prod = torch.norm(C, dim=2) ** 2
        pi = (1/Q) *(1/gn_var - 1/(gn_var + var_pri)).unsqueeze(-1) * inner_prod.reshape(P,num_ue, num_ind)
        omega = torch.clamp(torch.exp(Q * (pi - psi[:,:,None])), min=1e-20, max=1-1e-20)
        omega_sum = torch.sum(omega, dim=2)
        cst =  num_ind * (1 - a_factor) / a_factor
        af_post = omega / (omega_sum + cst[None,:]).unsqueeze(-1)
        
        # Compute posterior mean and variance
        x_hat = af_post[:,:,:,None] * theta[:,:,None,None] * C.reshape(P, num_ue, num_ind, Q)
        x_hat = x_hat.reshape(P, num_ue*num_ind, Q)
        v = af_post.reshape(P, num_ue*num_ind, 1) * (torch.abs(A) ** 2 + B) - torch.abs(x_hat) ** 2

        # Reconstruct received signal
        Yhat = torch.matmul(Phi, x_hat)

        # Refine sparsity ratio
        #pai_tmp = torch.mean(pai, dim=2)
        #lambda_tensor = pai_tmp.unsqueeze(2).expand_as(lambda_tensor)

        # pai_tmp = pai / (1 - pai)
        # pai_tmp = torch.transpose(torch.reshape(pai_tmp, [P, num_ue, num_ind, Q]),0,1)
        # pai_tmp = 1/(1 + (1 / torch.sum(pai_tmp, dim=2)))
        # a_factor = torch.mean(pai_tmp, dim=(1,2))

        # Check stopping criteria
        NMSE_iter = torch.norm(x_hat - Xhat_pre) ** 2 / torch.norm(Xhat_pre) ** 2
        NMSE_re = torch.norm(Y - Yhat) ** 2 / torch.norm(Y) ** 2

        if NMSE_iter < tol or NMSE_re < tol:
            break

    # Final processing
    # temppp = pai_tmp.clone()
    # temppp[temppp > 0.5] = 1
    # temppp[temppp <= 0.5] = 0

    return x_hat, iter + 1, NMSE_re, a_factor