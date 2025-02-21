#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse
import numpy as np

def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--snr_dB', type=int, default=10,
                        help="snr_dB")
    
    parser.add_argument('--num_ue', type=int, default=100,
                        help="number of users: K")
    parser.add_argument('--num_aue', type=int, default=10,
                        help="number of users: Ka")
    parser.add_argument('--fc', type=int, default=14*1e9,
                        help="number of users: K")
    parser.add_argument('--bw', type=int, default=10 * 1e6,
                        help="bandwidth: K")
    parser.add_argument('--delta_f', type=int, default=4.88*1e3,
                        help="subcarrier interval: \delta_f")
    
    parser.add_argument('--Nx', type=float, default=16,
                        help='number of antenna on axis x: Nx')
    parser.add_argument('--Ny', type=float, default=16,
                        help='number of antenna on axis y: Ny')
    
    parser.add_argument('--plos', type=int, default=0.8,
                        help="the number of local epochs: E")
    parser.add_argument('--rician', type=int, default=10,
                        help="rician factor: kappa")
    parser.add_argument('--num_path', type=int, default=3,
                        help="number of path")

    parser.add_argument('--tau_max', type=int, default=16,
                        help="maximum taps of delay")
    parser.add_argument('--offset_max', type=int, default=128,
                        help="maximum taps of offset")
    parser.add_argument('--len_code', type=int, default=24,
                        help="length of code: L")
    parser.add_argument('--nc', type=int, default=4, help="length of code: Nc")
    parser.add_argument('--nt', type=int, default=6, help="length of code: Nt")
    parser.add_argument('--num_carrier', type=int, default=2048, help="number of subcarrier: Nc")
    parser.add_argument('--num_blk', type=int, default=32, help="number of block: G")
    parser.add_argument('--blk_space', type=int, default=60,help="number of subcarrier: deltaF")
    
    parser.add_argument('--num_seg', type=int, default=7,
                        help="number of segments")
    parser.add_argument('--num_bit', type=int, default=7,
                        help="number of bits per segment")
    
    parser.add_argument('--damp', type=int, default=0.3,
                        help="damping factor")
    parser.add_argument('--niter', type=int, default=200,
                        help="number of iterations of AMP")
    parser.add_argument('--tol', type=int, default=1e-4,
                        help="number of iterations of AMP")
    args = parser.parse_args()
    return args
