python ./multiproc.py --nproc_per_node 8 ./launch.py --model resnet50 --precision TF32 --mode convergence --platform DGXA100 /imagenet --epochs 90 --mixup 0.0 --workspace ${1:-./} --raport-file raport.json
