# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# smddp: use dlc pytorch 1.8.1
ARG FROM_IMAGE_NAME=763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.8.1-gpu-py36-cu111-ubuntu18.04
FROM ${FROM_IMAGE_NAME}

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get install -y libsndfile1=1.0.28-4ubuntu0.18.04.2 sox=14.4.2-3ubuntu0.18.04.1 && \
    apt-get install -y --no-install-recommends numactl=2.0.11-2.1ubuntu0.1 && \
    rm -rf /var/lib/apt/lists/*

RUN COMMIT_SHA=f546575109111c455354861a0567c8aa794208a2 && \
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
RUN pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110

# smddp: install a specific version of apex that works with dlc pytorch image
RUN pip uninstall --yes apex
RUN cd /root && rm -rf apex && git clone https://github.com/NVIDIA/apex && cd apex && git checkout 59d2f7ac2385f20105513cdc76010f996f731af0 && python setup.py install --cuda_ext --cpp_ext --transducer --deprecated_fused_adam --distributed_lamb

WORKDIR /workspace/rnnt

COPY requirements.txt .
RUN pip install --no-cache --disable-pip-version-check -U -r requirements.txt

COPY . .

# smddp: some meta files for training data
RUN mkdir /sentencepieces && cp librispeech1023* /sentencepieces

RUN pip install nvtx
