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
        max_points = mm.shape[1] if mm.shape[1] > 0 else 0
        
        for id in ids:
            if id.shape[0] > 0 and max_points > 0:  # Only process non-empty ids with valid masks
                # Filter indices to be within bounds FIRST
                valid_id = id[id < max_points]
                if valid_id.shape[0] > 0:
                    if valid_id.shape[0] > n_pts:
                        # Sample from valid indices only
                        perm_size = min(valid_id.shape[0], n_pts)
                        perm = torch.randperm(valid_id.shape[0])[:perm_size]
                        m_idx_list.append(valid_id[perm])
                    else:
                        m_idx_list.append(valid_id)
        
        # Concatenate mask indices if we have any
        if m_idx_list:
            m_idx = torch.cat(m_idx_list)
            # Extra safety check
            m_idx = m_idx[m_idx < max_points] if max_points > 0 else m_idx
        else:
            m_idx = torch.tensor([], dtype=torch.long)
        
        # Generate random indices to fill up to n_samples
        remaining = n_samples - m_idx.shape[0]
        if remaining > 0 and max_points > 0:
            # Ensure we don't sample out of bounds
            r_idx = torch.randint(0, max_points, [remaining]).to(m_idx.device if len(m_idx) > 0 else 'cpu')
            if len(m_idx) > 0:
                idx = torch.cat((m_idx, r_idx))
            else:
                idx = r_idx
        else:
            idx = m_idx[:n_samples] if len(m_idx) > n_samples else m_idx
        
        # Final safety check - ensure all indices are within bounds
        if max_points > 0 and len(idx) > 0:
            idx = idx[(idx >= 0) & (idx < max_points)]
        
        sampled.append(idx)
    return sampled
