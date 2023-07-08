import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as transforms
from tqdm import tqdm
import seaborn as sns
from config import *
from sklearn.metrics import confusion_matrix


conf = config()

mel_spectrum = transforms.MelSpectrogram(
    sample_rate=conf.sample_rate,
    n_fft=conf.n_fft,
    win_length=conf.win_length,
    hop_length=conf.hop_length,
    center=True,
    pad_mode="reflect",
    power=2.0,
    norm="slaney",
    n_mels=conf.n_mels,
)


def wav_to_mel_log10(filepath):
    wave, _ = torchaudio.load(filepath)
    return torch.log10(mel_spectrum(wave) + 1e-10)


def normalize_std(melspec):
    return (melspec - torch.mean(melspec, dim=(2, 3), keepdim=True)) / torch.std(melspec, dim=(2, 3), keepdim=True)


def label_to_onehot(scene_label, label_list):
    label_temp = torch.zeros(label_list.shape)
    label_temp[label_list == scene_label] = 1
    return label_temp


def get_devices_no(filename, devices):
    return devices.index(filename.split('-')[-1][:-4])


# 下面这段代码还有问题
def label_for_multi(y):
    multi_y = np.zeros((y.shape[0], 3))
    for i in range(y.shape[0]):
        if np.argmax(y[i, :]) == 0 or np.argmax(y[i, :]) == 3 or np.argmax(y[i, :]) == 6:  # Indoor
            multi_y[i, 0] = 1
        elif np.argmax(y[i, :]) == 4 or np.argmax(y[i, :]) == 5 or np.argmax(y[i, :]) == 7 or np.argmax(
                y[i, :]) == 8:
            multi_y[i, 1] = 1
        else:
            multi_y[i, 2] = 1
    return multi_y


def plot_spectrogram(spec, title=None, ylabel="freq_bin", aspect="auto", xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram(dB)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("Frame")
    im = axs.imshow(spec, origin="lower", aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show()

def plot_confusion_matrix(true,predicted,label_list):
    cm = confusion_matrix(true, predicted, normalize="true")
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, square=True, cbar=False, annot=True, cmap="Blues")
    ax.set_xticklabels(label_list, rotation=90)
    ax.set_yticklabels(label_list, rotation=0)
    plt.xlabel("True label")
    plt.ylabel("Predicted label")

# //////////////////////////////////////////////////////
def plot_device_wise_log_losses(loss_all, predicted_all, train_val_y, train_val_devices, devices, label_list):
    results_table = np.zeros((11, len(devices) + 2))

    for label_id, _ in enumerate(label_list):
        label_indx = (train_val_y[:, label_id] == 1)
        results_table[label_id, len(devices) + 1] = (predicted_all[
                                                         label_indx] == label_id).sum() / label_indx.sum() * 100
        results_table[label_id, 0] = loss_all[label_indx].mean()

        for device_id, _ in enumerate(devices):
            device_indx = np.array(train_val_devices) == device_id
            device_wise_indx = np.array(label_indx) * (device_indx)
            results_table[label_id, device_id + 1] = loss_all[device_wise_indx].mean()
            results_table[10, device_id + 1] = loss_all[device_indx].mean()

    results_table[10, len(devices) + 1] = (predicted_all == torch.argmax(train_val_y,
                                                                         dim=1).clone().numpy()).sum() / len(
        predicted_all) * 100
    results_table[10, 0] = loss_all.mean()

    df_results = pd.DataFrame(results_table, columns=["Log Loss", *devices, "Accuracy %"],
                              index=[*label_list, "Ovberall"])
    print(df_results)


def train_data_process(train_csv, root_path, label_list):
    if conf.process_data:
        train_X = np.load(f"pro_data/{conf.process_data_f}train_X.npy")
        train_X = torch.from_numpy(train_X.astype(np.float32)).clone()
        train_y = np.load(f"pro_data/{conf.process_data_f}train_y.npy")
        train_y = torch.from_numpy(train_y.astype(np.float32)).clone()
        train_y_3class = np.load(f"pro_data/{conf.process_data_f}train_y_3class.npy")
        train_y_3class = torch.from_numpy(train_y_3class.astype(np.float32)).clone()

    else:
        train_X = []
        train_y = []


        for filename, scene_label in zip(tqdm(train_csv['filename']), train_csv['scene_label']):
            train_X.append(wav_to_mel_log10(root_path + filename))

            train_y.append(label_to_onehot(scene_label, label_list))


        train_X = torch.stack(train_X)
        train_y = torch.stack(train_y)

        train_y_3class = label_for_multi(train_y)
        train_y_3class = torch.from_numpy(train_y_3class.astype(np.float32)).clone()

        np.save(f"pro_data/{conf.process_data_f}train_X.npy", train_X)
        np.save(f"pro_data/{conf.process_data_f}train_y.npy", train_y)
        np.save(f"pro_data/{conf.process_data_f}train_y_3class.npy", train_y_3class)

    return train_X, train_y, train_y_3class


def val_data_process(val_csv, root_path, label_list):
    if conf.process_data:
        val_X = np.load(f"pro_data/{conf.process_data_f}val_X.npy")
        val_X = torch.from_numpy(val_X.astype(np.float32)).clone()
        val_y = np.load(f"pro_data/{conf.process_data_f}val_y.npy")
        val_y = torch.from_numpy(val_y.astype(np.float32)).clone()
        val_y_3class = np.load(f"pro_data/{conf.process_data_f}val_y_3class.npy")
        val_y_3class = torch.from_numpy(val_y_3class.astype(np.float32)).clone()

    else:
        val_X = []
        val_y = []


        for filename, scene_label in zip(tqdm(val_csv["filename"]), val_csv["scene_label"]):
            mel_spec = wav_to_mel_log10(root_path + filename)
            val_X.append(mel_spec)
            val_y.append(label_to_onehot(scene_label, label_list))


        val_X = torch.stack(val_X)
        val_y = torch.stack(val_y)

        val_y_3class = label_for_multi(val_y)
        val_y_3class = torch.from_numpy(val_y_3class.astype(np.float32)).clone()

        np.save(f"pro_data/{conf.process_data_f}val_X.npy", val_X)
        np.save(f"pro_data/{conf.process_data_f}val_y.npy", val_y)
        np.save(f"pro_data/{conf.process_data_f}val_y_3class.npy", val_y_3class)

    return val_X, val_y, val_y_3class


def apply_diff_freq(X, diff_freq_power, devices_no):
    if random.randrange(0, 13, 1) != 0:  # 1/13skip
        for idx, (X_temp, device_no) in enumerate(zip(X, devices_no)):
            tmp = (device_no == 0) * diff_freq_power[random.randrange(0, len(diff_freq_power), 1), :].unsqueeze(
                0).unsqueeze(2)
            tmp = torch.from_numpy(np.dot(torch.ones((X.shape[3], 1)), tmp[:, :, 0])).clone()
            X[idx, 0, :, :] = X_temp[0, :, :] + tmp.T

    return X


class AudioUtil():
  # ----------------------------
  # Load an audio file. Return the signal as a tensor and the sample rate
  # ----------------------------
    def open(audio_file):
        sig, sr = torchaudio.load(audio_file)
        return (sig, sr)
    def rechannel(aud, new_channel):
        sig, sr = aud

        if (sig.shape[0] == new_channel):
          # Nothing to do
          return aud

        if (new_channel == 1):
          # Convert from stereo to mono by selecting only the first channel
          resig = sig[:1, :]
        else:
          # Convert from mono to stereo by duplicating the first channel
          resig = torch.cat([sig, sig])

        return ((resig, sr))

    def resample(aud, newsr):
        sig, sr = aud

        if (sr == newsr):
              # Nothing to do
            return aud

        num_channels = sig.shape[0]
          # Resample first channel
        resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1, :])
        if (num_channels > 1):
              # Resample the second channel and merge both channels
            retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:, :])
            resig = torch.cat([resig, retwo])

        return ((resig, newsr))


    def pad_trunc(aud, max_ms):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr // 1000 * max_ms

        if (sig_len > max_len):
            # Truncate the signal to the given length
            sig = sig[:, :max_len]

        elif (sig_len < max_len):
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            # Pad with 0s
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            sig = torch.cat((pad_begin, sig, pad_end), 1)

        return (sig, sr)


    def time_shift(aud, shift_limit):
        sig, sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return (sig.roll(shift_amt), sr)


    def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
        sig, sr = aud
        top_db = 80

        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

        # Convert to decibels
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return (spec)


    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec

        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

        return aug_spec

#%%

class SoundDS():
  def __init__(self, df, data_path):
    self.df = df
    self.data_path = str(data_path)
    self.duration = 4000
    self.sr = 16000
    self.channel = 1
    self.shift_pct = 0.4

  # ----------------------------
  # Number of items in dataset
  # ----------------------------
  def __len__(self):
    return len(self.df)

  # ----------------------------
  # Get i'th item in dataset
  # ----------------------------
  def __getitem__(self, idx):
    # Absolute file path of the audio file - concatenate the audio directory with
    # the relative path
    audio_file = self.data_path + self.df.loc[idx, 'relative_path']
    # Get the Class ID
    class_id = self.df.loc[idx, 'classID']

    aud = AudioUtil.open(audio_file)
    # Some sounds have a higher sample rate, or fewer channels compared to the
    # majority. So make all sounds have the same number of channels and same
    # sample rate. Unless the sample rate is the same, the pad_trunc will still
    # result in arrays of different lengths, even though the sound duration is
    # the same.
    #reaud = AudioUtil.resample(aud, self.sr)
    #rechan = AudioUtil.rechannel(reaud, self.channel)

    dur_aud = AudioUtil.pad_trunc(aud, self.duration)
    shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
    sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
    aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

    return aug_sgram, class_id