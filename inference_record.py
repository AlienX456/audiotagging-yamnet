# Copyright 2019 The TensorFlow Authors All Rights Reserved.
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
# ==============================================================================

"""Inference demo for YAMNet."""
from __future__ import division, print_function

import sys

import numpy as np
import resampy
import soundfile as sf

import params
import yamnet as yamnet_model

import threading

import sounddevice as sd

import time

records = []

fs = 44100

def record():
  print('starting recording, waiting 1:30 to load the model')
  time.sleep(90)
  #record config
  duration = 4  # seconds 
  records_total = 0

  while(True):
    records_total += 1
    print('Total records '+str(records_total))
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.int16)
    sd.wait()
    records.append(myrecording)  



def inference():

  #loading model
  print('Loading YAMNET model')
  yamnet = yamnet_model.yamnet_frames_model(params)
  yamnet.load_weights('yamnet.h5')
  yamnet_classes = yamnet_model.class_names('yamnet_class_map.csv')

  infered = 0


  while(True):

    current_list = list(records)

    print('Reading current records for inference....')

    if current_list==[]:
      print('No current records')
    else:
      for record in current_list:
        infered += 1
        wav_data = record
        records.remove(record)

        assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
        waveform = wav_data / 32768.0  # Convert to [-1.0, +1.0]

        # Convert to mono and the sample rate expected by YAMNet.
        if len(waveform.shape) > 1:
          waveform = np.mean(waveform, axis=1)
        if fs != params.SAMPLE_RATE:
          waveform = resampy.resample(waveform, fs, params.SAMPLE_RATE)

        # Predict YAMNet classes.
        # Second output is log-mel-spectrogram array (used for visualizations).
        # (steps=1 is a work around for Keras batching limitations.)
        scores, _ = yamnet.predict(np.reshape(waveform, [1, -1]), steps=1)
        # Scores is a matrix of (time_frames, num_classes) classifier scores.
        # Average them along time to get an overall classifier output for the clip.
        prediction = np.mean(scores, axis=0)
        # Report the highest-scoring classes and their scores.
        top5_i = np.argsort(prediction)[::-1][:5]
        print(str(infered), ':\n' + 
              '\n'.join('  {:12s}: {:.3f}'.format(yamnet_classes[i], prediction[i])
                          for i in top5_i))
    time.sleep(1)


if __name__ == '__main__':
  trecord = threading.Thread(target=record)
  tinference = threading.Thread(target=inference)
  trecord.daemon = True
  tinference.daemon = True
  trecord.start()
  tinference.start()
  try:
    while True:
        pass
  except KeyboardInterrupt:
    print("KeyboardInterrupt has been caught.")

