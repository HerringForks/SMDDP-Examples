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

# Use PT 1.8.1 DLC for SMDDP
ARG FROM_IMAGE_NAME=763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.8.1-gpu-py36-cu111-ubuntu18.04
FROM ${FROM_IMAGE_NAME}

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN pip install --global-option="--cpp_ext" --global-option="--cuda_ext" git+git://github.com/NVIDIA/apex.git#egg=apex

WORKDIR /workspace/transformer-xl/pytorch

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir git+https://github.com/NVIDIA/dllogger.git#egg=dllogger

# ADD pytorch/ /workspace/transformer-xl/pytorch/
