# Pytorch implementation of Hungarian Algorithm
# Inspired from here : https://python.plainenglish.io/hungarian-algorithm-introduction-python-implementation-93e7c0890e15

# Despite my effort to parallelize the code, there is still some sequential workflows in this code

from typing import Tuple
import torch
from torch import Tensor

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def min_zero_row(zero_mat: Tensor) -> Tuple[Tensor, Tensor]:
    sum_zero_mat = zero_mat.sum(1)
    sum_zero_mat[sum_zero_mat == 0] = 9999

    zero_row = sum_zero_mat.min(0)[1]
    zero_column = zero_mat[zero_row].nonzero()[0]

    zero_mat[zero_row, :] = False
    zero_mat[:, zero_column] = False

    mark_zero = torch.tensor([[zero_row, zero_column]], device = device)
    return zero_mat, mark_zero

def mark_matrix(mat: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    zero_bool_mat = (mat == 0)
    zero_bool_mat_copy = zero_bool_mat.clone()

    marked_zero = torch.tensor([], device = device,dtype=torch.int)
    while (True in zero_bool_mat_copy):
        zero_bool_mat_copy, mark_zero = min_zero_row(zero_bool_mat_copy)
        marked_zero = torch.concat([marked_zero, mark_zero], dim = 0)

    marked_zero_row = marked_zero[:, 0]
    marked_zero_col = marked_zero[:, 1]

    arange_index_row = torch.arange(mat.shape[0], dtype=torch.float, device = device).unsqueeze(1)
    
    repeated_marked_row = marked_zero_row.repeat(mat.shape[0], 1)
    bool_non_marked_row = torch.all(arange_index_row != repeated_marked_row, dim = 1)
    non_marked_row = arange_index_row[bool_non_marked_row].squeeze()

    non_marked_mat = zero_bool_mat[non_marked_row.long(), :]
    marked_cols = non_marked_mat.nonzero().unique()

    is_need_add_row = True
    while is_need_add_row:
        repeated_non_marked_row = non_marked_row.repeat(marked_zero_row.shape[0], 1)
        repeated_marked_cols = marked_cols.repeat(marked_zero_col.shape[0], 1)

        first_bool = torch.all(marked_zero_row.unsqueeze(1) != repeated_non_marked_row, dim = 1)
        second_bool = torch.any(marked_zero_col.unsqueeze(1) == repeated_marked_cols, dim = 1)

        addit_non_marked_row = marked_zero_row[first_bool & second_bool]

        if addit_non_marked_row.shape[0] > 0:
            non_marked_row = torch.concat([non_marked_row.reshape(-1), addit_non_marked_row[0].reshape(-1)])
        else:
            is_need_add_row = False

    repeated_non_marked_row = non_marked_row.repeat(mat.shape[0], 1)
    bool_marked_row = torch.all(arange_index_row != repeated_non_marked_row, dim = 1)
    marked_rows = arange_index_row[bool_marked_row].squeeze(0)

    return marked_zero, marked_rows, marked_cols

def adjust_matrix(mat: Tensor, cover_rows: Tensor, cover_cols: Tensor) -> Tensor:
    bool_cover = torch.zeros_like(mat)
    bool_cover[cover_rows.long()] = True
    bool_cover[:, cover_cols.long()] = True

    non_cover = mat[bool_cover != True]
    min_non_cover = non_cover.min()

    mat[bool_cover != True] = mat[bool_cover != True] - min_non_cover

    double_bool_cover = torch.zeros_like(mat)
    double_bool_cover[cover_rows.long(), cover_cols.long()] = True

    mat[double_bool_cover == True] = mat[double_bool_cover == True] + min_non_cover

    return mat

def hungarian_algorithm(mat: Tensor) -> Tensor:
    dim = mat.shape[0]
    cur_mat = mat

    cur_mat = cur_mat - cur_mat.min(0, keepdim = True)[0]
    cur_mat = cur_mat - cur_mat.min(1, keepdim = True)[0]

    zero_count = 0
    while zero_count < dim:
        ans_pos, marked_rows, marked_cols = mark_matrix(cur_mat)
        zero_count = len(marked_rows) + len(marked_cols)

        if zero_count < dim:
            cur_mat = adjust_matrix(cur_mat, marked_rows, marked_cols)
    return ans_pos

if __name__ == "__main__":
    # Example 1
    mat = torch.tensor(
        [[7, 6, 2, 9, 2],
        [6, 2, 1, 3, 9],
        [5, 6, 8, 9, 5],
        [6, 8, 5, 8, 6],
        [9, 5, 6, 4, 7]], device = device)

    ans_pos = hungarian_algorithm(mat)
    print(ans_pos)

    res = mat[ans_pos[:, 0].long(), ans_pos[:, 1].long()]
    print(res)

    print(res.sum())

    print('==============')

    # Example 2
    mat = torch.tensor(
        [[108, 125, 150],
        [150, 135, 175],
        [122, 148, 250]], device = device)

    ans_pos = hungarian_algorithm(mat)
    print(ans_pos)

    res = mat[ans_pos[:, 0].long(), ans_pos[:, 1].long()]
    print(res)

    print(res.sum())

    print('==============')

    # Example 3
    mat = torch.tensor(
        [[1500, 4000, 4500],
        [2000, 6000, 3500], 
        [2000, 4000, 2500]], device = device)

    ans_pos = hungarian_algorithm(mat)
    print(ans_pos)

    res = mat[ans_pos[:, 0].long(), ans_pos[:, 1].long()]
    print(res)

    print(res.sum())

    print('==============')

    # Example 4
    mat = torch.tensor(
        [[5, 9, 3, 6],
        [8, 7, 8, 2], 
        [6, 10, 12, 7],
        [3, 10, 8, 6]], device = device)

    ans_pos = hungarian_algorithm(mat)
    print(ans_pos)

    res = mat[ans_pos[:, 0].long(), ans_pos[:, 1].long()]
    print(res)

    print(res.sum())