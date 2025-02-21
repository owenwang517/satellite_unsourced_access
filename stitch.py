from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import torch

def codeword_stitch(args, act_code_hat, act_code_num, channel_hat):
    # init mediods
    '''
    med:         num_aue_hat * num_blk * NxNy
    message_hat: num_aue_hat * num_seg
    '''
    ka_hat = int(max(act_code_num))
    message_hat = torch.zeros((ka_hat, args.num_seg),dtype=torch.long)
    for idx_seg in range(args.num_seg):
        if act_code_num[idx_seg] == ka_hat:
            med = channel_hat[idx_seg, :,act_code_hat[idx_seg],:].transpose(0,1)
            break
    
    # codeword stitching
    for idx_seg in range(args.num_seg):
        kd = act_code_num[idx_seg]
        cost = torch.zeros((ka_hat,ka_hat)).cuda()
        if kd == ka_hat:
            for i in range(ka_hat):
                for j in range(ka_hat):
                    ix1 = act_code_hat[idx_seg,i]
                    ch = channel_hat[idx_seg,:,ix1]
                    dis = (ch - med[j]).abs().sum((0,1))
                    # dis = torch.sum(ch * med[j]).abs()
                    cost[i,j] = dis
            #ans_pos = hungarian_algorithm(cost).cpu()
            ans_pos = linear_sum_assignment(cost.cpu())
            message_hat[ans_pos[1],idx_seg] = act_code_hat[idx_seg][ans_pos[0]]
            
            #update mediods
            # for idx_aue in range(ka_hat):
            #     ch = channel_hat[range(idx_seg+1),:,message_hat[idx_aue,:(idx_seg+1)]]
            #     dist_sum = torch.zeros((idx_seg+1,))
            #     for idx_seg_med in range(idx_seg+1):
            #         ch_au = channel_hat[idx_seg_med,:,message_hat[idx_aue,idx_seg_med]].unsqueeze(0)
            #         dist_sum[idx_seg_med] = (ch - ch_au).abs().square().sum((0,1,2))
            #     med_seg_id = dist_sum.argmin()
            #     med[idx_aue] = channel_hat[med_seg_id,:,message_hat[idx_aue,med_seg_id]]
                
        else:
            for i in range(act_code_num[idx_seg]):
                for j in range(ka_hat):
                    ix1 = act_code_hat[idx_seg,i]
                    ch = channel_hat[idx_seg,:,ix1]
                    dis = (ch - med[j]).abs().sum((0,1))
                    cost[i,j] = dis
            row_min = cost.min(dim=1).values
            _,indices = torch.sort(row_min, descending=True)
            # _, top_indices = torch.topk(row_sum[:kd], k= ka_hat-kd)
            top_indices = indices[:(ka_hat-kd)]
            cost[kd:] = cost[top_indices]
            new_code = act_code_hat[idx_seg,top_indices.cpu()]
            act_code_hat[idx_seg,kd:] = new_code
            # cost[i:] = cost[max_ix].expand_as(cost[i:])
            ans_pos = linear_sum_assignment(cost.cpu())

            message_hat[ans_pos[1],idx_seg] = act_code_hat[idx_seg][ans_pos[0]]
    return message_hat.cpu().numpy()

def codeword_stitch_prod(args, act_code_hat, act_code_num, channel_hat):
    # init mediods
    '''
    med:         num_aue_hat * num_blk * NxNy
    message_hat: num_aue_hat * num_seg
    '''
    ka_hat = int(max(act_code_num))
    message_hat = torch.zeros((ka_hat, args.num_seg),dtype=torch.long)
    for idx_seg in range(args.num_seg):
        if act_code_num[idx_seg] == ka_hat:
            med = channel_hat[idx_seg, :,act_code_hat[idx_seg],:].transpose(0,1)
            break
    
    # codeword stitching
    for idx_seg in range(args.num_seg):
        kd = act_code_num[idx_seg]
        prod = torch.zeros((ka_hat,ka_hat)).cuda()
        if kd == ka_hat:
            for i in range(ka_hat):
                for j in range(ka_hat):
                    ix1 = act_code_hat[idx_seg,i]
                    ch = channel_hat[idx_seg,:,ix1]
                    # dis = (ch - med[j]).abs().sum((0,1))
                    dis = torch.sum(ch * med[j]).abs()
                    prod[i,j] = dis
            #ans_pos = hungarian_algorithm(cost).cpu()
            ans_pos = linear_sum_assignment(prod.cpu(), True)
            message_hat[ans_pos[1],idx_seg] = act_code_hat[idx_seg][ans_pos[0]]
            
            #update mediods
            # for idx_aue in range(ka_hat):
            #     ch = channel_hat[range(idx_seg+1),:,message_hat[idx_aue,:(idx_seg+1)]]
            #     dist_sum = torch.zeros((idx_seg+1,))
            #     for idx_seg_med in range(idx_seg+1):
            #         ch_au = channel_hat[idx_seg_med,:,message_hat[idx_aue,idx_seg_med]].unsqueeze(0)
            #         dist_sum[idx_seg_med] = (ch - ch_au).abs().square().sum((0,1,2))
            #     med_seg_id = dist_sum.argmin()
            #     med[idx_aue] = channel_hat[med_seg_id,:,message_hat[idx_aue,med_seg_id]]
                
        else:
            for i in range(act_code_num[idx_seg]):
                for j in range(ka_hat):
                    ix1 = act_code_hat[idx_seg,i]
                    ch = channel_hat[idx_seg,:,ix1]
                    dis = torch.sum(ch * med[j].conj()).abs()
                    prod[i,j] = dis
            # row_sum = cost.sum(dim=1)
            # _, top_indices = torch.topk(row_sum[:kd], k= ka_hat-kd)
            prod_mean = prod.mean(dim=1,keepdim=True)
            code_detect = torch.sum((prod - prod_mean) > 0, dim=1)
            cost[kd:] = cost[top_indices]
            new_code = act_code_hat[idx_seg,top_indices.cpu()]
            act_code_hat[idx_seg,kd:] = new_code
            # cost[i:] = cost[max_ix].expand_as(cost[i:])
            ans_pos = linear_sum_assignment(cost.cpu(), True)
            
            message_hat[ans_pos[1],idx_seg] = act_code_hat[idx_seg][ans_pos[0]]
    return message_hat.cpu().numpy()

