export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get install -y libsndfile1 sox && \
    apt-get install -y --no-install-recommends numactl && \
    rm -rf /var/lib/apt/lists/*

COMMIT_SHA=f546575109111c455354861a0567c8aa794208a2 && \
    git clone https://github.com/HawkAaron/warp-transducer deps/warp-transducer && \
    cd deps/warp-transducer && \
    git checkout $COMMIT_SHA && \
    sed -i 's/set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30 -O2")/#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30 -O2")/g' CMakeLists.txt && \
    sed -i 's/set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_75,code=sm_75")/set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_80,code=sm_80")/g' CMakeLists.txt && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make VERBOSE=1 && \
    export CUDA_HOME="/usr/local/cuda" && \
    export WARP_RNNT_PATH=`pwd` && \
    export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME && \
    export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH" && \
    export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH && \
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH && \
    export CFLAGS="-I$CUDA_HOME/include $CFLAGS" && \
    cd ../pytorch_binding && \
    python3 setup.py install && \
    rm -rf ../tests test ../tensorflow_binding && \
    cd ../../..

# smddp: dali dataloader
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110

# smddp: install a specific version of apex that works with dlc pytorch image
pip uninstall --yes apex
cd /root && rm -rf apex && git clone https://github.com/NVIDIA/apex && cd apex && git checkout 59d2f7ac2385f20105513cdc76010f996f731af0 && python setup.py install --cuda_ext --cpp_ext --transducer --deprecated_fused_adam --distributed_lamb

pip install --no-cache --disable-pip-version-check -U -r /workspace/rnnt/requirements.txt

# smddp: some meta files for training data
mkdir /sentencepieces && cp /shared/SMDDP-Examples/pytorch/rnnt/librispeech1023* /sentencepieces

pip install nvtx
