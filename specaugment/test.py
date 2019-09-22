import wavio
import torch
import numpy
from specaugment import spec_augment_pytorch, melscale_pytorch
import matplotlib.pyplot as plt

PAD = 0
N_FFT = 512
SAMPLE_RATE = 16000

def get_spectrogram_feature(filepath, train_mode=False):
    (rate, width, sig) = wavio.readwav(filepath)
    sig = sig.ravel()

    stft = torch.stft(torch.FloatTensor(sig),
                      N_FFT,
                      hop_length=int(0.01*SAMPLE_RATE),
                      win_length=int(0.030*SAMPLE_RATE),
                      window=torch.hamming_window(int(0.030*SAMPLE_RATE)),
                      center=False,
                      normalized=False,
                      onesided=True)

    stft = (stft[:,:,0].pow(2) + stft[:,:,1].pow(2)).pow(0.5)

    amag = stft.clone().detach()

    amag = amag.view(-1, amag.shape[0], amag.shape[1])  # reshape spectrogram shape to [batch_size, time, frequency]
    mel = melscale_pytorch.mel_scale(amag, sample_rate=SAMPLE_RATE, n_mels=N_FFT//2+1)  # melspec with same shape

    plt.subplot(1,2,1)
    plt.imshow(mel.transpose(1,2).squeeze())

    p = 1  # always augment
    randp = numpy.random.uniform(0, 1)
    do_aug = p > randp
    if do_aug & train_mode:  # apply augment
        print("augment image")
        mel = spec_augment_pytorch.spec_augment(mel, time_warping_para=80, frequency_masking_para=54,
                                                time_masking_para=100, frequency_mask_num=1, time_mask_num=1)
    feat = mel.view(mel.shape[1], mel.shape[2])  # squeeze back to [frequency, time]
    feat = feat.transpose(0, 1).clone().detach()

    plt.subplot(1,2,2)
    plt.imshow(feat)
    plt.show()  # display it

    del stft, amag, mel
    return feat


filepath = "./train/train/train_data/41_0508_171_0_08412_03.wav"
for ii in range(10): feat = get_spectrogram_feature(filepath, train_mode=True)
