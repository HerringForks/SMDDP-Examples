# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from contextlib import contextmanager

import torch
import smdistributed.dataparallel.torch.distributed as dist

# def init_distributed(cuda):
#     """
#     Initializes distributed backend.

#     :param cuda: (bool) if True initializes nccl backend, if False initializes
#         gloo backend
#     """
#     world_size = int(os.environ.get('WORLD_SIZE', 1))
#     distributed = (world_size > 1)
#     if distributed:
#         backend = 'nccl' if cuda else 'gloo'
#         torch.distributed.init_process_group(backend=backend,
#                                              init_method='env://')
#         assert torch.distributed.is_initialized()
#     return distributed


def barrier():
    """
    Call dist.barrier() if distritubed is in use
    """
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def get_rank():
    """
    Gets distributed rank or returns zero if distributed is not initialized.
    """
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0
    return rank


def get_world_size():
    """
    Gets total number of distributed workers or returns one if distributed is
    not initialized.
    """
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
    else:
        world_size = 1
    return world_size


def all_reduce_item(value, op='sum'):
    """
    All-reduces single scalar value if distributed is in use
    """
    if dist.is_available() and dist.is_initialized():
        if op == 'sum' or op == 'mean':
            dop = dist.ReduceOp.SUM
        elif op == 'min':
            dop = dist.ReduceOp.MIN
        elif op == 'max':
            dop = dist.ReduceOp.MAX
        elif op == 'product':
            dop = dist.ReduceOp.PRODUCT
        else:
            raise RuntimeError('Unsupported reduce op')

        #backend = dist.get_backend()
        #if backend == dist.Backend.NCCL:
        #    device = torch.device('cuda')
        #elif backend == dist.Backend.GLOO:
        #    device = torch.device('cpu')
        #else:
        #    raise RuntimeError('Unsupported distributed backend')

        device = torch.device('cuda')
        tensor = torch.tensor(value, device=device)
        dist.all_reduce(tensor, dop)
        if op == 'mean':
            tensor /= get_world_size()
        ret = tensor.item()
    else:
        ret = value
    return ret


@contextmanager
def sync_workers():
    """
    Yields distributed rank and synchronizes all workers on exit.
    """
    rank = get_rank()
    yield rank
    barrier()
