# -*- coding:utf-8 -*-
"""
作者：Ameixa
日期：2022年10月01日
"""
import librosa.display
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import librosa
from sklearn import preprocessing

if __name__ == "__main__":
    import soundfile as sf
    import numpy as np
    file, samplerate  = sf.read("D:\\Ameixa\\学习\\实验室\\Snoring Detection\\DataSet\\Subject0905\\Snoring\\100.wav")
    left = file.T[0]
    right = file.T[1]
    plt.figure()
    l = file.shape[0]
    x =[i/8000 for i in range(l)]
    plt.plot(x,left,c='r')
    plt.show()
