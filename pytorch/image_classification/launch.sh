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

echo 'Run training...'
python -m torch.distributed.launch \
      --nproc_per_node=$PROC_PER_NODE \
      --nnodes="$WORLD_SIZE_JOB" \
      --node_rank="$RANK_NODE" \
      --master_addr="${MASTER_ADDR_JOB}" \
      --master_port=${MASTER_PORT_JOB} \
      /workspace/resnet/launch.py --model resnext101-32x4d --precision AMP --mode benchmark_training \
        --platform DGX1V /imagenet --epochs 1 --prof 100 --workspace /workspace/resnet --raport-file benchmark.json