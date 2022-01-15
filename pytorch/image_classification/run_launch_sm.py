import argparse
import os
import subprocess
import json

os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['FI_EFA_USE_DEVICE_RDMA'] = '1'

def parse_args():
    parser = argparse.ArgumentParser(description='Get model info')
    parser.add_argument('--num_nodes', type=int, help='Number of nodes')
    parser.add_argument("--model", default=None, type=str, required=True)
    parser.add_argument("--mode", default=None, type=str, required=True)
    parser.add_argument("--precision", default=None, type=str, required=True)
    parser.add_argument("--platform", default=None, type=str, required=True)
    parser.add_argument(
        "--data-backend",
        metavar="BACKEND",
        default="pytorch",
        help="data backend: (default: pytorch)",
    )
    parser.add_argument(
        "--epochs",
        default=90,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--prof", type=int, default=-1, metavar="N", help="Run only N iterations"
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="random seed used for numpy and pytorch"
    )
    parser.add_argument(
        "--raport-file",
        default="experiment_raport.json",
        type=str,
        help="file in which to store JSON experiment raport",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default="./",
        metavar="DIR",
        help="path to directory where checkpoints will be stored",
    )

    args, _ = parser.parse_known_args()
    print (_)
    return args


def invoke_train(cmd):
    process = subprocess.Popen(cmd, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True)
    while True:
        output = process.stdout.readline()
        if process.poll() is not None:
            break
        if output:
            print(output.decode("utf-8").strip())
    rc = process.poll()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)


if __name__ == "__main__":
    args = parse_args()
    num_nodes = args.num_nodes
    model = args.model
    mode = args.mode
    precision = args.precision
    platform = args.platform
    epochs = args.epochs
    prof = args.prof
    workspace = args.workspace
    data_backend = args.data_backend
    batch_size = args.batch_size
    optimizer_batch_size = args.optimizer_batch_size
    seed = args.seed
    raport_file = args.raport_file

    main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'launch.py'))

    num_gpus = int(os.environ["SM_NUM_GPUS"])
    hosts = json.loads(os.environ["SM_HOSTS"])
    current_host = os.environ["SM_CURRENT_HOST"]
    rank = hosts.index(current_host)
    work_dir = '/opt/ml/code'
    data_dir = os.environ["SM_CHANNEL_TRAIN"]

    cmd = f"python -m torch.distributed.launch --nnodes={num_nodes} --node_rank={rank} --nproc_per_node={num_gpus} \
        --master_addr={hosts[0]} --master_port='12345' \
    {main_path} --model {model} --precision {precision} --mode {mode} --platform {platform} {data_dir} --epochs {epochs} \
        --prof {prof} --workspace {workspace} --data-backend {data_backend} --seed {seed} --raport-file {raport_file}"


    print (cmd)
    invoke_train(cmd)
