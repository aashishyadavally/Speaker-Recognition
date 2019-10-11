"""
This file contains the :class: `FeatureExtractor` which extract either
Filterbank or MFCC features from the audio files in the given folder,
so as to return the features list and corresponding labels list.

Author:
-------
Aashish Yadavally
"""

import json
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.fftpack import dct


class FeatureExtractor:
    """Extracts either 'Filterbank' or 'MFCC' features for the audio files in
    the given (train/test) folder

    Parameters
    -----------
        action (str):
            The folder containing audio files, for which features need to be
            extracted
        ftype (str):
            One of 'mfcc' and 'fbank', default is 'fbank'

    """
    def __init__(self, action='train', ftype='fbank'):
        """Initializes :class: `FeatureExtractor`
        """
        self.data_folder = Path().resolve().parent / 'data' / action
        self.ftype = ftype
        self.mapping = self.get_audio_speaker_mapping()
        audios = self.data_folder.glob('*.wav')
        self.x, self.y = [], []
        for audio in audios:
            features, labels = self.extract_xy(audio)
            self.x.extend(features)
            self.y.extend(labels)


    def get_audio_speaker_mapping(self):
        """Maps each audio file to it's corresponding speaker

        Returns
        --------
            mapping (dict):
                Dictionary which maps each audio file with the speaker details
        """
        mapping_path = self.data_folder / 'mapping.json'
        mapping = json.loads(mapping_path)[0]
        return mapping

    def extract_xy(self, audio):
        """Extracts filterbank features for each frame in the audio file and
        corresponding speaker id information for the audios in the given
        data folder

        Arguments
        ---------
            audio (str):
                Name of audio file

        Returns
        -------
            features, labels (tuple):
                `features` is list of features for each frame in the audio
                file; `labels` is the list of corresponding speaker ID
        """
        sample_rate, signal = wavfile.read(str(self.data_folder / audio))
        speaker_id = self.mapping[audio]
        # Frame configuration
        frame_size = 0.025
        frame_stride = 0.01
        # Convert from seconds to samples
        frame_length = int(round(frame_size * sample_rate))
        frame_step = int(round(frame_stride * sample_rate))
        num_frames = int(np.ceil(float(np.abs(signal_length - \
                                          frame_length)) / frame_step))

        indices = np.tile(np.arange(0, frame_length),
                          (num_frames, 1)) + np.tile(np.arange(0,
                                                     num_frames * frame_step,
                                                     frame_step),
                                                     (frame_length, 1)).T
        frames = signal[indices.astype(np.int32, copy=False)]

        frames *= np.hamming(frame_length)
   
        NFFT = 512
        # Magnitude of the FFT
        frame_magnitude = np.absolute(np.fft.rfft(frames, NFFT))
        # Power Spectrum
        frame_power = ((1.0 / NFFT) * ((frame_magnitude) ** 2))

        num_filters = 40
        mel_low_freq = 0
        # Convert Hz to Mel
        mel_high_freq = (2595 * np.log10(1 + (sample_rate / 2) / 700))
        # Equally spaced in Mel scale
        mel_points = np.linspace(mel_low_freq, mel_high_freq, num_filters + 2)
        # Convert Mel to Hz
        hz_points = (700 * (10**(mel_points / 2595) - 1))
        bins = np.floor((NFFT + 1) * hz_points / sample_rate)

        filter_bank = np.zeros((num_filters, int(np.floor(NFFT / 2 + 1))))

        for i in range(1, num_filters + 1):
            filter_left = int(bins[i - 1])   # Left
            filter_i = int(bins[i])             # Center
            filter_right = int(bins[i + 1])    # Right

            for j in range(filter_left, filter_i):
                filter_bank[i - 1, j] = (j - bins[i - 1]) / (bins[i] - bins[i - 1])
            for j in range(filter_left, filter_right):
                filter_bank[i - 1, j] = (bins[i + 1] - j) / (bins[i + 1] - bins[i])

        filter_banks = np.dot(frame_power, filter_bank.T)
        # Numerical Stability
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps,
                                filter_banks)
        # dB
        filter_banks = 20 * np.log10(filter_banks)

        if self.ftype == 'fbank':
            return filter_banks, speaker_id * (filter_banks.size)
        else:
            num_ceps = 12
            mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13
            mfcc = mfcc.flatten()
            return mfcc, speaker_id * (filter_banks.size)
