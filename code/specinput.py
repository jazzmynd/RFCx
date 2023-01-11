import os
import numpy as np
import librosa
import tensorflow as tf

class params:
    """Parameters for spectrogram conversion"""
    
    sample_rate: int= 48000              # sample rate of input audio
    stft_window_seconds: float = 0.05    # seconds of audio analyzed in each spectrogram column
    stft_hop_seconds: float = 0.0125     # seconds shifted between spectrogram columns
    mel_bands: int = 224                 # number of frequency bands in the resulting spectrogram
    mel_min_hz: float = 50.0             # minimum frequency in the resulting spectrogram
    mel_max_hz: float = 24000.0          # maximum frequency in the resulting spectrogram
    sample_seconds: float = 3.0          # duration of each sample waveform

    @property
    def frame_length(self):
        return int(self.stft_window_seconds*self.sample_rate)


def wave_to_mel_spec(waveform, params=params):
    """Converts a 1-D waveform into a mel-scaled spectrogram
    
    Args:
        waveform: array with shape [<# samples>,]
        params: class with spectrogram parameter attributes
    
    Returns:
        mel_spectrogram: spectrogram with shape [<# frequency bands>, <# time frames>]
        
    """
    
    # Convert waveform into spectrogram using a Short-Time Fourier Transform.
    # Note that tf.signal.stft() uses a periodic Hann window by default.
    hop_length_samples = int(
      round(params.sample_rate * params.stft_hop_seconds)) # how many audio samples to shift between spectrogram frames (columns)
    fft_length = 2 ** int(np.ceil(np.log(params().frame_length) / np.log(2.0))) # length of fast Fourier transform
    num_spectrogram_bins = fft_length // 2 + 1 # number of spectrogram frequency bins
    magnitude_spectrogram = tf.abs(tf.signal.stft(
      signals=waveform,
      frame_length=params().frame_length,
      frame_step=hop_length_samples,
      fft_length= fft_length))

    # Convert spectrogram into mel spectrogram.
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=params.mel_bands,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=params.sample_rate,
        lower_edge_hertz=params.mel_min_hz,
        upper_edge_hertz=params.mel_max_hz)
    mel_spectrogram = tf.matmul(
      magnitude_spectrogram, linear_to_mel_weight_matrix)
    mel_spectrogram = tf.transpose(mel_spectrogram)
    mel_spectrogram = tf.reshape(mel_spectrogram , [tf.shape(mel_spectrogram)[0] ,tf.shape(mel_spectrogram)[1],1])

    return mel_spectrogram

def wav_to_np_image(path):
    waveform = librosa.load(path)
    return wave_to_mel_spec(waveform)


def load_audio(path):
    """Loads an audio file
    
    Args:
        path: audio file path
    
    Returns:
        y: time series of amplitudes with shape [<# samples>,]
        sr: sample rate of the recording
        
    """
    y, sr = librosa.load(path, sr=None)
    return y, sr

# path = "C://Users//tzy//Desktop//AC297//data//puerto-rico//train//audio//n//"
# image_path = "C://Users//tzy//Desktop//AC297//image_Data//puerto-rico//train//audio//n//"
# for class_name in os.listdir(path):
#     if not os.path.exists(image_path + class_name):
#         os.makedirs(image_path + class_name)
#     for wav in os.listdir(path + class_name):
#         y, sr = load_audio(path + class_name + "//" + wav)
#         X = wave_to_mel_spec(y)
#         np.save(image_path + class_name + "//" + wav[0:-4], X.numpy())
    

# path = "C://Users//tzy//Desktop//AC297//data//puerto-rico//train//audio//p//"
# image_path = "C://Users//tzy//Desktop//AC297//image_Data//puerto-rico//train//audio//p//"
# for class_name in os.listdir(path):
#     if not os.path.exists(image_path + class_name):
#         os.makedirs(image_path + class_name)
#     for wav in os.listdir(path + class_name):
#         y, sr = load_audio(path + class_name + "//" + wav)
#         X = wave_to_mel_spec(y)
#         np.save(image_path + class_name + "//" + wav[0:-4], X.numpy())
    




