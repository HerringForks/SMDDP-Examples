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

# smddp: use dlc pytorch 1.10.2
ARG region=us-east-1
ARG FROM_IMAGE_NAME=763104351884.dkr.ecr.${region}.amazonaws.com/pytorch-training:1.10.2-gpu-py38-cu113-ubuntu20.04-sagemaker-v1.1
FROM ${FROM_IMAGE_NAME}

WORKDIR /workspace/rnnt

RUN mkdir -p /shared && cd /shared \
    && git clone --recursive  https://github.com/HerringForks/SMDDP-Examples \
    && cp -r SMDDP-Examples/pytorch/rnnt /workspace \
    && bash /workspace/rnnt/requirements.sh

