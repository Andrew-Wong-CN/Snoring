# -*- coding:utf-8 -*-
"""
作者：Ameixa
日期：2022年10月01日
"""
import os
import librosa
import numpy
import numpy as np
import scipy.io.wavfile as wav

sample_rate = 16000  # sample rate
sample_rate_original = 32000  # original sample rate
n_fft = 2048  # fft sample points
frame_length = 0.128  # frame length(seconds)
frame_shift = 0.064  # frame shift(seconds)
hop_length = int(sample_rate * frame_shift)  # (number of sample points)
win_length = int(sample_rate * frame_length)
n_mels = 20  # Number of Mel banks to generate
power = 1.2
pre_emphasis = .97
max_db = 100
ref_db = 20
top_db = 15  # the threshold below reference to consider as silence


def get_mel_phase(y):
    """
    :param:
        :type audio file (32k)
    :return:
        mel: 幅度谱（频谱）A 2d array of shape (T,n_mels)
        phase: 相位谱
    """

    # loading sound life
    # y_32k, sr = librosa.load(fpath, sr=sample_rate_original)

     # resampling
    # y = librosa.resample(y, orig_sr=32000, target_sr=16000)

    # Trimming causes the output is not same shape
    # y, _ = librosa.effects.trim(y, top_db=top_db)

    # Pre-emphasis
    y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])

    # stft
    linear = librosa.stft(y=y,
                          n_fft=n_fft,
                          hop_length=hop_length,
                          win_length=win_length,
                          center=True)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2,T)

    # compute a mel-scale spectrogram
    mel = librosa.feature.melspectrogram(S=mag, n_mels=n_mels)
    mel = librosa.power_to_db(mel, ref=numpy.max)
    mel = numpy.transpose(mel, (1, 0))

    # get phase
    phase = np.angle(mel) # return the angle of the complex argument

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
    # shape = audio_data.shape
    # left = audio_data[..., 0:shape[1]+1:2]
    # right = audio_data[..., 1:shape[1]+1:2]
    left = audio_data[:, 0, :]
    right = audio_data[:, 1, :]
    return left, right


def concat_mel_and_phase(data):
    concat_data = []
    for i in range(data[0].shape[0]):
        concat_data.append(list(map(lambda x: x[i], data)))
    return numpy.asarray(concat_data)

# run the code below to transpose all the 32k sampling audio into 16k sampling audio
if __name__ == "__main__":
    folder = "F:\\Dataset"
    subjects = os.listdir(folder)
    for subject in subjects:
        subject_path = folder + "\\" + subject
        snoring = subject_path + "\\" + "Snoring_16k"
        if not os.path.exists(snoring):
            os.makedirs(snoring)
        audio_list = os.listdir(subject_path + "\\" + "Snoring")
        i = 0
        for audio in audio_list:
            audio_name = str(i) + ".wav"
            audio_path = subject_path + "\\" + "Snoring" + "\\" + audio
            audio_data, sr = librosa.load(audio_path, sr=32000, mono=False)
            audio_data = librosa.resample(audio_data, orig_sr=32000, target_sr=16000)
            audio_data = audio_data.T
            audio_save_path = snoring + "\\" + audio_name
            wav.write(audio_save_path, 16000, audio_data)
            i += 1