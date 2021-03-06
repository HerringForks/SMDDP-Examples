# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import nvidia.dali
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import multiprocessing
import numpy as np
import torch
import math
# smddp:
import smdistributed.dataparallel.torch.distributed as dist

class PipelineParams:
    def __init__(
            self,
            sample_rate=16000,
            max_duration=float("inf"),
            normalize_transcripts=True,
            trim_silence=False,
            speed_perturbation=None
        ):
        pass

class SpeedPerturbationParams:
    def __init__(
            self,
            min_rate=0.85,
            max_rate=1.15,
            p=1.0,
        ):
        pass

class PertCoeffIterator():
    def __init__(self, batch_size, pert_coeff):
        self.batch_size = batch_size
        self.pert_coeff = pert_coeff.view(-1, pert_coeff.size(-1))
        self.i = 0
        self.n = self.pert_coeff.size(0)

    def __iter__(self):
        return self

    def __next__(self):
        if self.i == self.n:
            raise StopIteration
        ret = self.pert_coeff[self.i]
        self.i += 1
        return ret


class DaliPipeline(nvidia.dali.pipeline.Pipeline):
    def __init__(self, *,
                 pipeline_type,
                 device_id,
                 num_threads,
                 batch_size,
                 file_root: str,
                 sampler,
                 sample_rate,
                 resample_range: list,
                 window_size,
                 window_stride,
                 nfeatures,
                 nfft,
                 dither_coeff,
                 silence_threshold,
                 preemph_coeff,
                 max_duration,
                 seed,
                 preprocessing_device="gpu",
                 synthetic_seq_len=None,
                 in_mem_file_list=True,
                 pre_sort=False,
                 dont_use_mmap=False):
        super().__init__(batch_size, num_threads, device_id, seed)

        self._dali_init_log(locals())

        
        if dist.is_initialized() and not sampler.dist_sampler:
            shard_id = dist.get_rank()
            n_shards = dist.get_world_size()
        else:
            shard_id = 0
            n_shards = 1

        self.preprocessing_device = preprocessing_device.lower()
        assert self.preprocessing_device == "cpu" or self.preprocessing_device == "gpu", \
            "Incorrect preprocessing device. Please choose either 'cpu' or 'gpu'"

        self.resample_range = resample_range

        train_pipeline = pipeline_type == 'train'
        self.train = train_pipeline
        self.sample_rate = sample_rate
        self.dither_coeff = dither_coeff
        self.nfeatures = nfeatures
        self.max_duration = max_duration
        self.do_remove_silence = True if silence_threshold is not None else False

        if pipeline_type == "train" and pre_sort:
            pci = PertCoeffIterator(batch_size, sampler.pert_coeff)

        shuffle = train_pipeline and not sampler.is_sampler_random()
        if synthetic_seq_len is None:
            if in_mem_file_list:
                self.read = ops.readers.File( name="Reader", dont_use_mmap=dont_use_mmap,
                                            pad_last_batch=(pipeline_type == 'val'), 
                                            device="cpu", file_root=file_root, files=sampler.files, 
                                            labels=sampler.labels, shard_id=shard_id,
                                            num_shards=n_shards, shuffle_after_epoch=shuffle) 
            else:   
                self.read = ops.readers.File( name="Reader", dont_use_mmap=dont_use_mmap,
                                            pad_last_batch=(pipeline_type == 'val'), 
                                            device="cpu", file_root=file_root, 
                                            file_list=sampler.get_file_list_path(), 
                                            shard_id=shard_id,
                                            num_shards=n_shards, shuffle_after_epoch=shuffle)
        else:
            # sampler.get_file_list_path()    # still want to model this cost
            self.read = ops.readers.File(name="Reader", dont_use_mmap=dont_use_mmap, pad_last_batch=(pipeline_type == 'val'), device="cpu", file_root=file_root, file_list="/workspace/rnnt/rnnt_dali.file_list.synth", shard_id=shard_id,
                                       num_shards=n_shards, shuffle_after_epoch=shuffle)

        # TODO change ExternalSource to Uniform for new DALI release
        if resample_range is not None and not (resample_range[0] == 1 and resample_range[1] == 1):
            if pre_sort:
                self.speed_perturbation_coeffs = ops.ExternalSource(source=pci)    
            else:
                self.speed_perturbation_coeffs = ops.random.Uniform(device="cpu", range=resample_range)
        else:
            self.speed_perturbation_coeffs = None


        


        self.decode = ops.decoders.Audio(device="cpu", sample_rate=self.sample_rate if resample_range is None else None,
                                       dtype=types.FLOAT, downmix=True)

        self.normal_distribution = ops.random.Normal(device=preprocessing_device)

        self.preemph = ops.PreemphasisFilter(device=preprocessing_device, preemph_coeff=preemph_coeff)

        self.spectrogram = ops.Spectrogram(device=preprocessing_device, nfft=nfft,
                                           window_length=window_size * sample_rate,
                                           window_step=window_stride * sample_rate)

        self.mel_fbank = ops.MelFilterBank(device=preprocessing_device, sample_rate=sample_rate, nfilter=self.nfeatures,
                                           normalize=True)

        self.log_features = ops.ToDecibels(device=preprocessing_device, multiplier=np.log(10), reference=1.0,
                                           cutoff_db=math.log(1e-20))

        self.get_shape = ops.Shapes(device=preprocessing_device)

        self.normalize = ops.Normalize(device=preprocessing_device, axes=[1])

        self.pad = ops.Pad(device=preprocessing_device, fill_value=0)

        # Silence trimming
        self.get_nonsilent_region = ops.NonsilentRegion(device="cpu", cutoff_db=silence_threshold)
        self.trim_silence = ops.Slice(device="cpu", normalized_anchor=False, normalized_shape=False, axes=[0])
        self.to_float = ops.Cast(device="cpu", dtype=types.FLOAT)

        # synthesize data
        self.synthetic_seq_len = synthetic_seq_len
        if self.synthetic_seq_len is not None:
            self.constant = ops.Constant(device=preprocessing_device, dtype=types.FLOAT, fdata=100.0, shape=synthetic_seq_len[0])

    @classmethod
    def from_config(cls, pipeline_type, device_id, batch_size, file_root: str, sampler, config_data: dict,
                    config_features: dict, seed, device_type: str = "gpu", do_resampling: bool = True,
                    num_cpu_threads=multiprocessing.cpu_count(), synthetic_seq_len=None,
                    in_mem_file_list=True, pre_sort=False, dont_use_mmap=False):

        max_duration = config_data['max_duration']
        sample_rate = config_data['sample_rate']
        silence_threshold = -60 if config_data['trim_silence'] else None

        # TODO Take into account resampling probablity
        # TODO     config_features['speed_perturbation']['p']

        if do_resampling and config_data['speed_perturbation'] is not None:
            resample_range = [config_data['speed_perturbation']['min_rate'],
                              config_data['speed_perturbation']['max_rate']]
        else:
            resample_range = None

        window_size = config_features['window_size']
        window_stride = config_features['window_stride']
        nfeatures = config_features['n_filt']
        nfft = config_features['n_fft']
        dither_coeff = config_features['dither']
        preemph_coeff = .97

        return cls(pipeline_type=pipeline_type,
                   device_id=device_id,
                   preprocessing_device=device_type,
                   num_threads=num_cpu_threads,
                   batch_size=batch_size,
                   file_root=file_root,
                   sampler=sampler,
                   sample_rate=sample_rate,
                   resample_range=resample_range,
                   window_size=window_size,
                   window_stride=window_stride,
                   nfeatures=nfeatures,
                   nfft=nfft,
                   dither_coeff=dither_coeff,
                   silence_threshold=silence_threshold,
                   preemph_coeff=preemph_coeff,
                   max_duration=max_duration,
                   synthetic_seq_len=synthetic_seq_len,
                   seed=seed,
                   in_mem_file_list=in_mem_file_list,
                   pre_sort=pre_sort,
                   dont_use_mmap=dont_use_mmap
        )

    @staticmethod
    def _dali_init_log(args: dict):
        if (not dist.is_initialized() or (
                dist.is_initialized() and dist.get_rank() == 0)):  # print once
            max_len = max([len(ii) for ii in args.keys()])
            fmt_string = '\t%' + str(max_len) + 's : %s'
            print('Initializing DALI with parameters:')
            for keyPair in sorted(args.items()):
                print(fmt_string % keyPair)

    def _remove_silence(self, inp):
        begin, length = self.get_nonsilent_region(inp)
        out = self.trim_silence(inp, self.to_float(begin), self.to_float(length))
        return out

    def define_graph(self):
        audio, label = self.read()
        if not self.train or self.speed_perturbation_coeffs is None:
            audio, sr = self.decode(audio)
        else:
            resample_coeffs = self.speed_perturbation_coeffs() * self.sample_rate
            audio, sr = self.decode(audio, sample_rate=resample_coeffs)

        if self.do_remove_silence:
            audio = self._remove_silence(audio)

        # Max duration drop is performed at DataLayer stage

        # replace the audio signal with synthetic data
        if self.synthetic_seq_len is not None:
            audio = self.constant()

        if self.preprocessing_device == "gpu":
            audio = audio.gpu()

        if self.dither_coeff != 0.:
            audio = audio + self.normal_distribution(audio) * self.dither_coeff

        audio = self.preemph(audio)

        audio = self.spectrogram(audio)
        audio = self.mel_fbank(audio)
        audio = self.log_features(audio)

        audio_len = self.get_shape(audio)

        audio = self.normalize(audio)
        audio = self.pad(audio)

        # When modifying DALI pipeline returns, make sure you update `output_map` in DALIGenericIterator invocation
        return audio.gpu(), label, audio_len.gpu()

