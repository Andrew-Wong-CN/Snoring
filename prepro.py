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
mel_frequency_bins = 20  # Number of Mel banks to generate
power = 1.2
pre_emphasis = .97
max_db = 100
ref_db = 20
top_db = 15  # the threshold below reference to consider as silence


def get_mel_phase(y):

    # pre-emphasis
    y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])

    # short time fourier transform
    linear = librosa.stft(y=y,
                          n_fft=n_fft,
                          hop_length=hop_length,
                          win_length=win_length,
                          center=True)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2,T)

    # compute a mel-scale spectrogram
    mel = librosa.feature.melspectrogram(S=mag, n_mels=mel_frequency_bins)
    mel = librosa.power_to_db(mel, ref=numpy.max)
    mel = numpy.transpose(mel, (1, 0))

    # get phase
    phase = np.angle(mel) # return the angle of the complex argument

    return mel, phase


def get_mel_phase_batch(y_batch):
    mel_list = []
    phase_list = []
    for i in y_batch:
        mel, phase = get_mel_phase(i)
        mel_list.append(mel)
        phase_list.append(phase)
    return numpy.asarray(mel_list), numpy.asarray(phase_list)


def separate_channels(y):
    left = y[:, 0, :]
    right = y[:, 1, :]
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
            audio_list = os.listdir(subject_path + "\\" + "Snoring_32k")
            m = 0
            for audio in audio_list:
                audio_name = str(m) + ".wav"
                audio_path = subject_path + "\\" + "Snoring_32k" + "\\" + audio
                audio_data, sr = librosa.load(audio_path, sr=32000, mono=False)
                audio_data = librosa.resample(audio_data, orig_sr=32000, target_sr=16000)
                audio_data = audio_data.T
                audio_save_path = snoring + "\\" + audio_name
                wav.write(audio_save_path, 16000, audio_data)
                print(f"{subject}: {m} is done")
                m += 1
        else:
            pass