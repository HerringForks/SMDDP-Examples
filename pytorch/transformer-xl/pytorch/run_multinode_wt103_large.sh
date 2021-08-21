#!/bin/bash

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

export OMPI_MCA_btl_tcp_if_exclude="docker0,lo"
export PMIX_MCA_gds=hash
export SAGEMAKER_INSTANCE_TYPE="ml.p4d.24xlarge"
export NCCL_SOCKET_IFNAME="^lo,docker"

# Required to use EFA on EC2
export FI_EFA_USE_DEVICE_RDMA=1

WORLD_SIZE_JOB=$SLURM_NTASKS
RANK_NODE=$SLURM_NODEID
PROC_PER_NODE=8
MASTER_ADDR_JOB=$SLURM_SUBMIT_HOST
MASTER_PORT_JOB="12234"

if [[ $1 == 'train' ]] || [[ $1 == 'all' ]]; then
    echo 'Run training...'
    python -m torch.distributed.launch \
      --nproc_per_node=$PROC_PER_NODE \
      --nnodes="$WORLD_SIZE_JOB" \
      --node_rank="$RANK_NODE" \
      --master_addr="${MASTER_ADDR_JOB}" \
      --master_port=${MASTER_PORT_JOB} \
      train.py --config_file wt103_large.yaml ${@:2}
fi 

if [[ $1 == 'eval' ]] || [[ $1 == 'all' ]]; then
    echo 'Run evaluation...'
    python eval.py \
        --config_file wt103_large.yaml \
        --config 8dgx2_16gpu_fp16 \
        ${@:2}
fi

if [[ $1 != 'train' ]] && [[ $1 != 'eval' ]] && [[ $1 != 'all' ]]; then
    echo 'unknown argment 1'
fi
