import numpy as np
from tsl_channel_args import *
from options import *
import matplotlib.pyplot as plt
from utils.dftmtx import dftmtx
from utils.amp import *
import torch
from utils.visualize import *
from utils.oamp import *
from utils.hungary import *
from scipy.optimize import linear_sum_assignment
from stitch import *

torch.manual_seed(77)
np.random.seed(1)

args = args_parser()
args.nc = 1
args.nt = int(args.len_code/args.nc)

num_sim = 10000
pe_sum = 0
pec_sum = 0
for idx_sim in range(num_sim):
    # generate channel
    '''
    channel: num_ue * (num_blk * nc) * num_antenna
    '''
    carrier_idx_list = get_carrier_list(args)
    channel = sat_ma_channel(args, carrier_idx_list)
    channel = channel.transpose((0,2,1)) 
    num_code = 2 ** args.num_bit

    # generate codebook
    '''
    codebook: num_blk * len_code * num_code
    '''
    # dftmtx_tmp = dftmtx(num_code)
    # codebook = np.zeros((args.num_blk, args.len_code, num_code),dtype=np.complex64)
    # for idx_blk in range(args.num_blk):
    #     idx_tmp = random.sample(range(num_code), args.len_code)
    #     idx_tmp = np.sort(idx_tmp)
    #     codebook[idx_blk,:,:] = dftmtx_tmp[idx_tmp,:]

    codebook = np.random.randn(args.num_blk, args.len_code, num_code) + 1j * np.random.randn(args.num_blk, args.len_code, num_code)

    # generate active user indices and codewords
    '''
    message: num_seg * num_aue

    '''
    id_aue = np.sort(np.random.choice(args.num_ue, args.num_aue, replace=False))
    message = np.random.randint(0,num_code, (args.num_seg, args.num_aue))

    # derive channel matrix according to codewords

    channel_mtx = np.zeros((args.num_seg, args.nc*args.num_blk, num_code, args.Nx*args.Ny),dtype=np.complex64)

    for idx_seg in range(args.num_seg):
        for idx_aue in range(args.num_aue):
            id_code = message[idx_seg,idx_aue]
            id_ue = id_aue[idx_aue]
            channel_mtx[idx_seg, :, id_code,:] += channel[id_ue]

    # # 计算幅值
    # magnitude = np.real(channel[0,:,0])

    # # 绘制幅值的折线图
    # plt.figure(figsize=(8, 5))
    # plt.plot(magnitude, marker='o', linestyle='-', color='b', label='Real Part')
    # plt.title("Real part of Channel", fontsize=14)
    # plt.xlabel("Index", fontsize=12)
    # plt.ylabel("Real part", fontsize=12)
    # plt.grid(True, linestyle='--', alpha=0.7)
    # plt.legend(fontsize=12)
    # plt.show()

    codebook_res = codebook.reshape((args.num_blk * args.nc, args.nt, num_code))

    R = codebook_res @ channel_mtx
    snr = 10 ** (args.snr_dB/10)
    sig_pow = np.square(np.abs(R)).sum((0,1,2,3)) / R.size
    noise_pow = sig_pow / snr
    noise_real = np.random.randn(*R.shape)  # 解包 R.shape
    noise_imag = np.random.randn(*R.shape)  # 解包 R.shape
    noise = np.sqrt(noise_pow / 2) * (noise_real + 1j * noise_imag)
    R = R.reshape((args.num_seg, args.num_blk, args.nc*args.nt, args.Nx*args.Ny))
    noise = noise.reshape(*R.shape)
    # R = R + noise


    R = torch.tensor(R, dtype=torch.complex64).cuda()
    codebook = torch.tensor(codebook, dtype=torch.complex64).cuda()


    fx = torch.tensor(dftmtx(args.Nx),dtype=torch.complex64).cuda()
    fy = torch.tensor(dftmtx(args.Ny),dtype=torch.complex64).cuda()
    f_all = torch.kron(fy, fx)
    R_ang = R @ f_all.T

    # codewords detectopm
    error = 0
    act_code_hat = torch.zeros(args.num_seg,args.num_aue,dtype=torch.long)
    act_code_num = torch.zeros(args.num_seg,dtype=torch.int)
    channel_ang_hat = torch.zeros((args.num_seg,args.num_blk, num_code, args.Nx*args.Ny),dtype=torch.complex64).cuda()

    for idx_seg in range(args.num_seg):
        channel_hat_tmp, belief, iter, NMSE_re = GMMV_AMP_cluster(R_ang[idx_seg], codebook, args.damp, args.niter, args.tol, device='cuda')
        channel_ang_hat[idx_seg] = channel_hat_tmp

        belief = belief.mean(0)
        act_code_be = torch.where(belief > 0.5)
        act_code_be = sorted(set(act_code_be[0].cpu().numpy()))

        error += len(set(message[idx_seg].flatten()) - set(act_code_be))/args.num_aue/args.num_seg
        act_code_num[idx_seg] = int(len(act_code_be))
        act_code_hat[idx_seg,:int(len(act_code_be))] = torch.tensor(list(act_code_be))

    pec_sum += error
    channel_spa_hat = channel_ang_hat @ f_all.conj()
    ce_nmse = (channel_spa_hat.cpu() - torch.tensor(channel_mtx)).abs().square().sum((0,1,2,3))/torch.tensor(channel_mtx).abs().square().sum((0,1,2,3))
    ce_nmse = 10 * torch.log10(ce_nmse)
    print('sim = %4d, code error prob = %6.6f, average_pc = %6.6f, nmse = %6.6f' % (idx_sim, error, pec_sum/(idx_sim+1), ce_nmse))

    '''
    channel_hat: num_seg * num_blk * num_code * NxNy (S * G * K * N)
    act_code_num: num_seg
    act_code_hat: num_seg * num_aue
    '''
    channel_ang_hat = channel_ang_hat.cpu()
    # transform the estimated channel to delay-angle domain
    for idx_seg in range(args.num_seg):
        for idx_code in range(act_code_num[idx_seg]):
            ch_hat_tmp = channel_ang_hat[idx_seg, :,act_code_hat[idx_seg,idx_code]]
            ch_hat_tmp = ch_hat_tmp.transpose(0,1)
            ch_hat_tmp = ch_hat_tmp @ dftmtx(args.num_blk)
            channel_ang_hat[idx_seg, :,act_code_hat[idx_seg,idx_code]] = ch_hat_tmp.transpose(0,1)
    
    # # saving channels
    # ch_tmp = np.abs(channel_hat[0,:,act_code_hat[0,5],:]).reshape(-1,args.Nx*args.Ny).numpy()
    # np.savetxt('channel.csv', ch_tmp, delimiter=',')

    # if idx_sim == 1:
    #     # visualize the estimated channel
    #     for idx_seg in range(args.num_seg):
    #         for idx_code in range(act_code_num[idx_seg]):
    #             code = act_code_hat[idx_seg,idx_code]
    #             ch_tmp = channel_ang_hat[idx_seg,:,code,:].numpy()
    #             plt.imsave('./channel_img/channel_seg%d_%d.png' % (idx_seg, code), np.abs(ch_tmp), cmap='viridis')

    # codewords stitching
    message_hat = codeword_stitch(args, act_code_hat, act_code_num, channel_ang_hat)

    # calculate error rate
    error_md = 0
    for io in range(args.num_aue):
        flag = 0
        for ih in range(message_hat.shape[0]):
            if (message[:,io] == message_hat[ih,:]).all():
                flag = 1
                break
        if flag == 0:
            error_md += 1
    
    error_fa = 0
    for ih in range(message_hat.shape[0]):
        flag = 0
        for io in range(args.num_aue):
            if (message[:,io] == message_hat[ih,:]).all():
                flag = 1
                break
        if flag == 0:
            error_fa += 1
    
    pmd = error_md/args.num_aue
    pfa = error_fa/message_hat.shape[0]
    pe = pmd + pfa
    pe_sum += pe

    print('sim = %4d, code error prob= %4.4f, stitch error prob = %4.4f, average pe = %4.4f' % (idx_sim, error, pe, pe_sum/(idx_sim+1)))
assert False





