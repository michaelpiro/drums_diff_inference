import numpy as np
import torchaudio
import torch
# import librosa
# from librosa.filters import mel as librosa_mel_fn
from torchaudio import transforms
# @title audio functions
from configs import configs
from configs.configs import TrainingConfig

""" =========== Constants =============== """""
SAMPLE_RATE = TrainingConfig.SAMPLE_RATE
N_FFT = TrainingConfig.N_FFT
HOP_LENGTH = TrainingConfig.HOP_LENGTH
WIN_LENGTH = TrainingConfig.WIN_LENGTH
N_MELS = TrainingConfig.N_MELS
FMAX = TrainingConfig.FMAX
FMIN = TrainingConfig.FMIN
AUDIO_LEN_SEC = TrainingConfig.AUDIO_LEN_SEC
TARGET_LENGTH = TrainingConfig.TARGET_MEL_LENGTH
NUM_SAMPLES = TrainingConfig.NUM_SAMPLES
TARGET_LENGTH_SEC = TrainingConfig.TARGET_LENGTH_SEC
# DTYPE = TrainingConfig.mixed_precision
DTYPE = 'torch.float32'

TARGET_SR = SAMPLE_RATE


@torch.no_grad()
def mel_spectogram(
        sample_rate,
        hop_length,
        win_length,
        n_fft,
        n_mels,
        f_min,
        f_max,
        power,
        normalized,
        min_max_energy_norm,
        norm,
        mel_scale,
        compression,
        audio,
):
    audio_to_mel = transforms.Spectrogram(
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
        power=power,
        normalized=normalized,
    ).to(audio.device)

    mel_scale = transforms.MelScale(
        sample_rate=sample_rate,
        n_stft=n_fft // 2 + 1,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        norm=norm,
        mel_scale=mel_scale,
    ).to(audio.device)
    spec = audio_to_mel(audio)
    mel = mel_scale(spec)
    rmse = torch.norm(mel, dim=0)

    if min_max_energy_norm:
        rmse = (rmse - torch.min(rmse)) / (torch.max(rmse) - torch.min(rmse))

    if compression:
        mel = dynamic_range_compression(mel)

    return mel


@torch.no_grad()
def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """Dynamic range compression for audio signals
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def to_mono(signal):
    # print(signal.shape[0])

    if signal.shape[0] == 2:
        signal = torch.mean(signal, dim=0, keepdim=False)
    return signal


def get_dynamic_ref(mel_spectrogram):
    return np.mean(mel_spectrogram)


def crop_audio_randomly_training(audio1, audio2, length, generator=None):
    """
    Crops the audio tensor to a specified length at a random start index,
    using a PyTorch generator for randomness.

    Parameters:
    - audio_tensor (torch.Tensor): The input audio tensor.
    - length (int): The desired length of the output tensor.
    - generator (torch.Generator, optional): A PyTorch generator for deterministic randomness.

    Returns:
    - torch.Tensor: The cropped audio tensor of the specified length.
    """

    # Ensure the desired length is not greater than the audio tensor length

    if length > audio1.shape[0]:
        raise ValueError("Desired length is greater than the audio tensor length.")

    # Calculate the maximum start index for cropping
    max_start_index = audio1.size(0) - length

    # Generate a random start index from 0 to max_start_index using the specified generator
    start_index = torch.randint(0, max_start_index + 1, (1,), generator=generator).item()

    # Crop the audio tensor from the random start index to the desired length
    audio1 = audio1[start_index:start_index + length]
    audio2 = audio2[start_index:start_index + length]

    return audio1, audio2


def crop_audio_randomly_inference(audio1, length, generator=None):
    """
    Crops the audio tensor to a specified length at a random start index,
    using a PyTorch generator for randomness.

    Parameters:
    - audio_tensor (torch.Tensor): The input audio tensor.
    - length (int): The desired length of the output tensor.
    - generator (torch.Generator, optional): A PyTorch generator for deterministic randomness.

    Returns:
    - torch.Tensor: The cropped audio tensor of the specified length.
    """

    # Ensure the desired length is not greater than the audio tensor length

    if length > audio1.shape[0]:
        raise ValueError("Desired length is greater than the audio tensor length.")

    # Calculate the maximum start index for cropping
    max_start_index = audio1.size(0) - length

    # Generate a random start index from 0 to max_start_index using the specified generator
    start_index = torch.randint(0, max_start_index + 1, (1,), generator=generator).item()
    # start_index = 0

    # Crop the audio tensor from the random start index to the desired length
    audio1 = audio1[start_index:start_index + length]

    return audio1


def preprocess_training(orig_waveform, sr1, no_drums_waveform, sr2, generator=None):
    if generator is None:
        generator = torch.manual_seed(0)
    # convert to mono
    no_drums_waveform = to_mono(no_drums_waveform).squeeze(0)
    orig_waveform = to_mono(orig_waveform).squeeze(0)

    # if the data is not in the correct sample rate, we resample it.
    if TARGET_SR != SAMPLE_RATE:
        orig_waveform = torchaudio.functional.resample(orig_waveform, sr1, TARGET_SR)
        no_drums_waveform = torchaudio.functional.resample(no_drums_waveform, sr2, TARGET_SR)
    else:
        if sr1 != SAMPLE_RATE:
            orig_waveform = torchaudio.functional.resample(orig_waveform, sr1, SAMPLE_RATE)
        if sr2 != SAMPLE_RATE:
            no_drums_waveform = torchaudio.functional.resample(no_drums_waveform, sr2, SAMPLE_RATE)

    orig_waveform, no_drums_waveform = crop_audio_randomly_training(orig_waveform, no_drums_waveform, NUM_SAMPLES,
                                                                    generator)
    # create log mel spec
    log_mel_orig = mel_spectogram(
        audio=orig_waveform,
        sample_rate=16000,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_fft=N_FFT,
        n_mels=N_MELS,
        f_min=0.0,
        f_max=8000.0,
        power=1,
        normalized=False,
        min_max_energy_norm=True,
        norm="slaney",
        mel_scale="slaney",
        compression=True
    )
    log_mel_orig = torch.permute(log_mel_orig, (1, 0)).unsqueeze(0).unsqueeze(0)
    log_mel_no_drums = mel_spectogram(
        audio=no_drums_waveform,
        sample_rate=16000,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_fft=N_FFT,
        n_mels=N_MELS,
        f_min=0.0,
        f_max=8000.0,
        power=1,
        normalized=False,
        min_max_energy_norm=True,
        norm="slaney",
        mel_scale="slaney",
        compression=True
    )
    log_mel_no_drums = torch.permute(log_mel_no_drums, (1, 0)).unsqueeze(0).unsqueeze(0)
    # log_mel_orig = torch.transpose(log_mel_orig, 0, 1)
    # log_mel_no_drums = torch.transpose(log_mel_no_drums, 0, 1)
    if DTYPE == 'fp16':
        log_mel_orig = log_mel_orig.half()
        log_mel_no_drums = log_mel_no_drums.half()
    else:
        log_mel_orig = log_mel_orig.to(torch.float32)
        log_mel_no_drums = log_mel_no_drums.to(torch.float32)
    return log_mel_orig, log_mel_no_drums


@torch.no_grad()
def preprocess_for_inference(orig_waveform, sr1, generator=None):
    # if generator is None:
    #     generator = torch.manual_seed(0).to(orig_waveform.device)
    # convert to mono
    orig_waveform = to_mono(orig_waveform)

    # if the data is not in the correct sample rate, we resample it.
    if TARGET_SR != SAMPLE_RATE:
        orig_waveform = torchaudio.functional.resample(orig_waveform, sr1, TARGET_SR)
    else:
        if sr1 != SAMPLE_RATE:
            orig_waveform = torchaudio.functional.resample(orig_waveform, sr1, SAMPLE_RATE)
    orig_waveform = crop_audio_randomly_inference(orig_waveform, NUM_SAMPLES, generator)
    # create log mel spec

    log_mel_orig = mel_spectogram(
        audio=orig_waveform,
        sample_rate=16000,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_fft=N_FFT,
        n_mels=N_MELS,
        f_min=0.0,
        f_max=8000.0,
        power=1,
        normalized=False,
        min_max_energy_norm=True,
        norm="slaney",
        mel_scale="slaney",
        compression=True
    )
    log_mel_orig = torch.permute(log_mel_orig, (1, 0)).unsqueeze(0).unsqueeze(0)
    if DTYPE == 'fp16':
        log_mel_orig = log_mel_orig.half()
    else:
        log_mel_orig = log_mel_orig.to(torch.float32)
    return log_mel_orig


def normalize_audio(audio):
    audio = audio - audio.mean()
    audio = audio / (torch.max(torch.abs(audio)) + 1e-8)
    audio = audio * 0.5
    return audio


def load_audio(audio_path):
    audio, sr = torchaudio.load(audio_path)
    return audio, sr

# def show_spectrogram(mel_spectrogram):
#     import matplotlib.pyplot as plt
#     import librosa.display
#     mel_spectrogram = torch.transpose(mel_spectrogram, 0, 1)
#     to_np = mel_spectrogram.cpu().numpy()
#     # fig, ax = plt.subplots(figsize=(10, 4))
#     librosa.display.specshow(to_np, x_axis='time', y_axis='mel',
#                              sr=TARGET_SR,
#                              hop_length=HOP_LENGTH,
#                              win_length=WIN_LENGTH)
