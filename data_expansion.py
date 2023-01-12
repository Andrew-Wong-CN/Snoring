# -*- coding:utf-8 -*-
"""
作者：Ameixa
日期：2023年01月07日
"""

import librosa
import numpy as np
import os
import random


def get_SNR(origin, noise):
    """

    :param origin: the audio with snore and noise
    :param noise: the audio with noise only
    :return: snr
    """
    length = min(len(noise), len(origin))
    est_clean = origin[:length] - noise[:length]

    snr = 10 * np.log10((np.sum(est_clean ** 2)) / (np.sum(noise ** 2)))
    return snr


def generate_audio(mpssc, noise, snr, target_file):
    """

    :param mpssc: mpssc file
    :param noise: noise file
    :param snr: specific snr
    :param target_file: address to place generated data
    :return:
    """

    # produce snoring in line with snr
    noise_power = (1 / noise.shape[0]) * np.sum(np.power(noise, 2))
    snore_variance = noise_power / np.power(10, snr)
    snore = (np.sqrt(snore_variance) / np.std(mpssc)) * mpssc

    # splicing audio and output


def main(dataset_new_path, dataset_mpssc_path):
    """

    :param dataset_new_path:
    :param dataset_mpssc_path:
    :return:
    """
    mpssc_list = os.listdir(dataset_mpssc_path)
    new_list = os.listdir(dataset_new_path)

    for snore in mpssc_list:
        for i in range(10):
            for person in new_list:
                for j in range(len(os.listdir(os.path.join(dataset_new_path, person, "Snoring_16k")))):
                    rand_noise = str(random.randint(0, 4))
                    noise,  = librosa.load(os.path.join(dataset_new_path, person, "Noise", rand_noise + ".wav"))
                    rand_snore = str(random.randint(0, 4))
                    snore,  = librosa.load(os.path.join(dataset_new_path, person, "Snore", rand_noise + ".wav"))
                    snr = get_SNR(snore, noise)
                    generate_audio(snore,noise,snr,"target address")



if __name__ == "__main__":
    dataset_new_path = "D:\Ameixa\学习\实验室\Snoring Detection\DataSet_new"
    dataset_mpssc_path = "D:\Ameixa\学习\实验室\Snoring Detection\MPSSC_v1.0\MPSSC_v1.0"

    sound, fs = librosa.load(r"D:\Ameixa\学习\实验室\Snoring Detection\DataSet_new\2022-08-31-M-56\Snoring_16k\10.wav")
    print(sound.shape, fs)
