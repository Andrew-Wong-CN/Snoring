# -*- coding:utf-8 -*-
"""
作者：Ameixa
日期：2022年10月01日
"""
import os
import librosa
import numpy
import numpy as np
import scipy.io.wavfile

sample_rate = 16000  # sample rate
sample_rate_original = 32000  # original sample rate
n_fft = 2048  # fft sample points
frame_length = 0.128  # frame length(seconds)
frame_shift = 0.064  # frame shift(seconds)
hop_length = int(sample_rate * frame_shift)  # (number of sample points)
win_length = int(sample_rate * frame_length)
n_mels = 80  # Number of Mel banks to generate
power = 1.2
preemphasis = .97
max_db = 100
ref_db = 20
top_db = 15  # the threshold below reference to consider as silence


def get_mel_phase(y_32k):
    """

    :param:
        :type audio file (32k)
    :return:
        mel: 幅度谱（频谱）A 2d array of shape (T,n_mels)
        phase: 相位谱
    """

    # loading sound life
    # y_32k, sr = librosa.load(fpath, sr=sample_rate_original)

    #  resampling
    y = librosa.resample(y_32k, sample_rate_original, sample_rate)

    # Trimming causes the output is not same shape
    # y, _ = librosa.effects.trim(y, top_db=top_db)

    # Pre-emphasis
    y = np.append(y[0], y[1:] - preemphasis * y[:-1])

    # stft
    linear = librosa.stft(y=y,
                          n_fft=n_fft,
                          hop_length=hop_length,
                          win_length=win_length)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2,T)

    # compute a mel-scale spectrogram
    mel = librosa.feature.melspectrogram(S=mag, n_mels=n_mels)
    mel = librosa.power_to_db(mel, ref=numpy.max)
    mel = numpy.transpose(mel, (1, 0))

    # get phase
    phase = np.angle(mel)

    return mel, phase

def get_mel_phase_batch(y_batch):
    mels = []
    phases = []
    for i in y_batch:
        mel, phase = get_mel_phase(i)
        mels.append(mel)
        phases.append(phase)
    return numpy.asarray(mels), numpy.asarray(phases)



def separate_channels(audio_data):
    '''

    :param y: dual channel audio
    :return: left and right channels
    '''
    shape = audio_data.shape
    left = audio_data[..., 0:shape[1]+1:2]
    right = audio_data[..., 1:shape[1]+1:2]
    return left, right

def concat_mel_and_phase(data):
    concat_data = []
    for i in range(data[0].shape[0]):
        concat_data.append(list(map(lambda x: x[i], data)))
    return numpy.asarray(concat_data)
