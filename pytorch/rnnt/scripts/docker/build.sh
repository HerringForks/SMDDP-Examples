#!/bin/bash
  
region=$1

docker build . --rm -t mlperf/rnn_speech_recognition_smddp --build-arg "region=${region}"

