from typing import List

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def pad_stack(tensor_list: List[Tensor]):
    """
    pad each tensor on the input to the max value in shape[1] and
    concatenate them in a single tensor.
    Input:
        list of tensors [Ni,Pi]
    Output:
        tensor [sum(Ni),max(Pi)]
    """
    _max = max([t.shape[1] for t in tensor_list])
    batched = torch.cat([F.pad(t, (0, _max - t.shape[1])) for t in tensor_list])
    return batched


def sample_points(masks, masks_ids, n_pts, n_samples):
    # select n_pts per mask to focus on instances
    # plus random points up to n_samples
    sampled = []
    for ids, mm in zip(masks_ids, masks):
        # Collect mask indices
        m_idx_list = []
        for id in ids:
            if id.shape[0] > 0:  # Only process non-empty ids
                if id.shape[0] > n_pts:
                    # Ensure indices are within bounds
                    perm_size = min(id.shape[0], n_pts)
                    perm = torch.randperm(id.shape[0])[:perm_size]
                    m_idx_list.append(id[perm])
                else:
                    m_idx_list.append(id)
        
        # Concatenate mask indices if we have any
        if m_idx_list:
            m_idx = torch.cat(m_idx_list)
        else:
            m_idx = torch.tensor([], dtype=torch.long)
        
        # Ensure m_idx doesn't exceed the mask dimensions
        if mm.shape[1] > 0:
            # Filter out indices that exceed mask dimensions
            valid_mask = m_idx < mm.shape[1]
            m_idx = m_idx[valid_mask]
        
        # Generate random indices to fill up to n_samples
        remaining = n_samples - m_idx.shape[0]
        if remaining > 0 and mm.shape[1] > 0:
            r_idx = torch.randint(mm.shape[1], [remaining]).to(m_idx)
            idx = torch.cat((m_idx, r_idx))
        else:
            idx = m_idx
        
        # Final bounds check
        if mm.shape[1] > 0:
            idx = idx[idx < mm.shape[1]]
        
        sampled.append(idx)
    return sampled
