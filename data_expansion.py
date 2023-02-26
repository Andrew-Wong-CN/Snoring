# -*- coding:utf-8 -*-
"""
作者：Ameixa
日期：2023年01月07日
"""
import librosa
import numpy as np
import os
import random
from prepro import get_mel_phase

def get_snr(origin, noise):
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
    snore_variance = (1 / noise.shape[0]) * np.sum(np.power(noise, 2)) * np.power(10, snr)
    snore = (np.sqrt(snore_variance) / np.sqrt(np.sum(np.power(mpssc, 2)) / mpssc.shape[0])) * mpssc

    # splicing audio and output
    snore_mel_spectrum, _ = get_mel_phase(snore)
    noise_spectrum, _ = get_mel_phase(noise)
    rand = random.randint(0, len(noise_spectrum))
    if rand + len(snore_mel_spectrum) > len(noise_spectrum): # 替换从rand下一个点开始
        outer_num = len(snore_mel_spectrum) - (len(noise_spectrum) - rand)
        noise_spectrum[:outer_num] = snore_mel_spectrum[(len(snore_mel_spectrum)-outer_num):]
        noise_spectrum[rand:] = snore_mel_spectrum[:len(snore_mel_spectrum) - outer_num]
    else:
        noise_spectrum[rand:(len(snore_mel_spectrum)+rand)] = snore_mel_spectrum
    np.save(target_file, noise_spectrum)



def main(dataset_new_path, dataset_mpssc_path):
    """

    :param dataset_new_path:
    :param dataset_mpssc_path:
    :return:
    """
    mpssc_list = os.listdir(dataset_mpssc_path)
    new_list = os.listdir(dataset_new_path)
    i = 0

    for mpssc in mpssc_list:
        for i in range(10):
            for person in new_list:
                for j in range(len(os.listdir(os.path.join(dataset_new_path, person, "Snoring_16k")))):
                    rand_noise = str(random.randint(0, 4))
                    noise, _ = librosa.load(os.path.join(dataset_new_path, person, "Noise", rand_noise + ".wav"), sr=16000)
                    rand_snore = str(random.randint(0, 4))
                    snore, _ = librosa.load(os.path.join(dataset_new_path, person, "Snore", rand_snore + ".wav"))
                    snr = get_snr(snore, noise)
                    generate_audio(snore, noise, snr, os.path.join("F:\\Dataset\\GeneratedData", str(i)+".npy"))
                    print(f"{i} is done")
                    i += 1



if __name__ == "__main__":
    dataset_new_path = "F:\\Dataset"
    dataset_mpssc_path = "F:\\MPSSC_v1.0"

    main(dataset_new_path, dataset_mpssc_path)
