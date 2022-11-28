import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


if __name__ == "__main__":
    import soundfile as sf
    file, sr  = sf.read("F:\\Dataset\\2022-09-05-M-31\\Snoring_16k\\81.wav")
    left = file.T[0]
    right = file.T[1]
    plt.figure()
    l = file.shape[0]
    x =[i/8000 for i in range(l)]
    plt.plot(x,left,c='r')
    plt.show()
