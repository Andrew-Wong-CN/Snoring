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
import pandas as pd

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


def generate_audio(mpssc, noise, snr, npy_file, csv_file):
    """

    :param mpssc: mpssc file
    :param noise: noise file
    :param snr: specific snr
    :param npy_file: address to place generated data
    :return:
    """

    # produce snoring in line with snr
    snore_variance = (1 / noise.shape[0]) * np.sum(np.power(noise, 2)) * np.power(10, snr)
    snore = (np.sqrt(snore_variance) / np.sqrt(np.sum(np.power(mpssc, 2)) / mpssc.shape[0])) * mpssc

    # splicing audio and output
    snore_mel_spectrum, _ = get_mel_phase(snore)
    noise_spectrum, _ = get_mel_phase(noise)
    label = np.zeros(len(noise_spectrum), dtype=int)
    rand = random.randint(0, len(noise_spectrum))
    if rand + len(snore_mel_spectrum) > len(noise_spectrum): # 替换从rand下一个点开始
        outer_num = len(snore_mel_spectrum) - (len(noise_spectrum) - rand)
        noise_spectrum[:outer_num] = snore_mel_spectrum[(len(snore_mel_spectrum)-outer_num):]
        label[:outer_num] = np.ones(outer_num, dtype=int)
        noise_spectrum[rand:] = snore_mel_spectrum[:len(snore_mel_spectrum) - outer_num]
        label[rand:] = np.ones(len(snore_mel_spectrum) - outer_num, dtype=int)
    else:
        noise_spectrum[rand:(len(snore_mel_spectrum)+rand)] = snore_mel_spectrum
        label[rand:(len(snore_mel_spectrum)+rand)] = np.ones(len(snore_mel_spectrum), dtype=int)
    df = pd.DataFrame({"label":label})
    df.to_csv(csv_file)
    np.save(npy_file, noise_spectrum)



def main(dataset_new_path, dataset_mpssc_path):
    """

    :param dataset_new_path:
    :param dataset_mpssc_path:
    :return:
    """
    mpssc_list = os.listdir(dataset_mpssc_path)
    new_list = os.listdir(dataset_new_path)
    m = 0

    for mpssc in mpssc_list:
        for i in range(10):
            for person in new_list:
                for j in range(len(os.listdir(os.path.join(dataset_new_path, person, "Snoring_16k")))):
                    rand_noise = str(random.randint(0, 4))
                    noise, _ = librosa.load(os.path.join(dataset_new_path, person, "Noise", rand_noise + ".wav"), sr=16000)
                    rand_snore = str(random.randint(0, 4))
                    snore, _ = librosa.load(os.path.join(dataset_new_path, person, "Snore", rand_snore + ".wav"), sr=16000)
                    snr = get_snr(snore, noise)
                    mpssc_snore, _ = librosa.load(os.path.join(dataset_mpssc_path, mpssc), sr=16000)
                    generate_audio(mpssc_snore, noise, snr,
                                   os.path.join("F:\\GeneratedData\\npy", str(m)+".npy"),
                                   os.path.join("F:\\GeneratedData\\csv", str(m)+".csv"))
                    print(f"{m} is done")
                    m += 1



if __name__ == "__main__":
    dataset_new_path = "F:\\Dataset"
    dataset_mpssc_path = "F:\\MPSSC_v1.0\\wav_norm_float32"

    main(dataset_new_path, dataset_mpssc_path)
