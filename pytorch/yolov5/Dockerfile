FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.9.1-gpu-py38-cu111-ubuntu20.04

# Install python dependencies
RUN pip3 install tensorboard

# Create working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Copy contents
COPY . /usr/src/app

