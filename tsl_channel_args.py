import numpy as np
import numpy.matlib
from options import *

#def sat_mimo_planar(N_x = 16, N_y = 16, num_path = 3, k_rician = 1, f_c = 14.5 * 1e9, BW = 252.9 * 1e6):

def gen_pos(num_ue, dis_lim = 500, sat_height = 550):
    
    dis = np.random.uniform(0, dis_lim, num_ue)

    ang = np.random.uniform(0, 2*np.pi, num_ue)
    
    phi = np.arctan(dis/sat_height)

    theta = ang

    sine_val_x = np.sin(phi) * np.cos(theta)

    sine_val_y = np.sin(phi) * np.sin(theta)

    return sine_val_x, sine_val_y


def array_vector(N_x, N_y, sine_val_x, sine_val_y, f_c):
    
    c_speed = 3e8
    wave_num = 2 * np.pi * f_c / c_speed
    
    v_x = np.arange(0, N_x, 1) 
    v_x = np.sqrt(1 / N_x) *np.exp(- 1j * wave_num * v_x * sine_val_x)
    v_y = np.arange(0, N_y, 1)
    v_y = np.sqrt(1 / N_y) *np.exp(- 1j * wave_num * v_y * sine_val_y)
    v = np.kron(v_x, v_y)

    return v


# def tsl_mimo_planar(args, sine_val_x, sine_val_y):

#     N_r = args.N_x * args.N_y
#     chmtx_f = np.zeros((N_r, num_carrier), dtype=np.complex64)

#     for idx_carrier in range(num_carrier):

#         carrier_freq = idx_carrier * sys_car_space + sys_f_c
#         v = array_vector(N_x, N_y, sine_val_x, sine_val_y, carrier_freq)
        
#         # generate gain
#         gain = np.random.randn(ch_num_path) + 1j * np.random.randn(ch_num_path)
#         los_gain = np.sqrt(ch_rician/(2*(ch_rician + 1)))
#         nlos_gain = np.sqrt(1/(2*(ch_rician + 1)))
#         gain[0] = los_gain * gain[0]
#         gain[1:] = nlos_gain * gain[1:]

#         delay = np.random.choice(np.arange(ch_tap_max), size=ch_num_path, replace=False)

#         gain_delay_stack = gain * np.exp(-1j * 2 * np.pi * delay * sys_ts)
#         av_stack = numpy.matlib.repmat(v, ch_num_path, 1).T
#         chmtx_f[:,idx_carrier] = av_stack @ gain_delay_stack.T

#     return chmtx_f

def tsl_uplink_mimo_planar(args, sine_val_x, sine_val_y, carrier_idx_list):

    N_r = args.Nx * args.Ny
    chmtx_f = np.zeros((N_r, len(carrier_idx_list)), dtype=np.complex64)
    idx = 0
    delay = np.sort(np.random.choice(np.arange(args.tau_max), size=args.num_path, replace=False))
    offset = np.random.randint(args.offset_max)
    delay += offset
    ts = 1/args.bw
        
    # generate gain
    gain = np.random.randn(args.num_path) + 1j * np.random.randn(args.num_path)
    los_gain = np.sqrt(args.rician/(2*(args.rician + 1)))
    nlos_gain = np.sqrt(1/(2*(args.rician + 1)))
    gain[0] = los_gain * gain[0]
    gain[1:] = nlos_gain * gain[1:]

    for idx_carrier in carrier_idx_list:
        freq = idx_carrier * args.delta_f - args.bw/2
        v = array_vector(args.Nx, args.Ny, sine_val_x, sine_val_y, args.fc)
        gain_delay_stack = gain * np.exp(-1j * 2 * np.pi * freq * delay * ts)
        av_stack = numpy.matlib.repmat(v, args.num_path, 1).T
        chmtx_f[:,idx] = av_stack @ gain_delay_stack.T
        idx +=1

    return chmtx_f

def sat_ma_channel(args, carrier_idx_list):

    N_r = args.Nx * args.Ny

    sine_val_x, sine_val_y = gen_pos(args.num_ue)

    ma_ch_mtx = np.zeros((args.num_ue, N_r, len(carrier_idx_list)), dtype=np.complex64)
    
    for idx_user in range(args.num_ue):
        
        # ma_ch_mtx[idx_user, :, :] = tsl_mimo_planar(N_x, N_y, sine_val_x[idx_user], sine_val_y[idx_user], sys_f_c,sys_car_space, carrier_idx_list)
        ma_ch_mtx[idx_user, :, :] = tsl_uplink_mimo_planar(args, sine_val_x[idx_user], sine_val_y[idx_user], carrier_idx_list)
    return ma_ch_mtx

def get_carrier_list(args):
    len_nc = args.nc
    blk_space = args.blk_space
    num_carrier = args.num_carrier

    carrier_idx_list = np.zeros((len_nc, args.num_blk))
    carrier_list_base = np.arange(0,num_carrier,blk_space)[:args.num_blk]
    for idx_nc in range(len_nc):
        carrier_idx_list[idx_nc,:] = carrier_list_base + idx_nc
    carrier_idx_list = numpy.reshape(carrier_idx_list, (-1),order='F')
    return carrier_idx_list

def get_carrier_list_random(args):
    len_nc = args.nc
    blk_space = args.blk_space
    num_carrier = args.num_carrier

    carrier_idx_list = np.zeros((len_nc, args.num_blk))
    carrier_list_base = np.sort(np.random.choice(np.arange(0,int(num_carrier/len_nc)), args.num_blk, replace=False))
    for idx_nc in range(len_nc):
        carrier_idx_list[idx_nc,:] = carrier_list_base + idx_nc
    carrier_idx_list = numpy.reshape(carrier_idx_list, (-1),order='F')
    return carrier_idx_list, carrier_list_base

if __name__ ==  "__main__":
    args = args_parser()
    args.nc = 2
    args.nt = 12
    carrier_idx_list = get_carrier_list(args)
    channel = sat_ma_channel(args, carrier_idx_list)

    assert False