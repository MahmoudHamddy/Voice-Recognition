import os
import numpy as np
import scipy
from scipy.io import wavfile
import scipy.fftpack as fft
from scipy.signal import get_window
#import IPython.display as ipd
import matplotlib.pyplot as plt
import wave
import scipy.io.wavfile as wavfile
import io 
import base64 
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import seaborn as sns
import librosa
import librosa.display
import librosa as lr
from librosa.core import stft, amplitude_to_db
from librosa.display import specshow

class variables:
   counter=0

#normalize audio
def normalize_audio(audio):
    audio = audio / np.max(np.abs(audio))
    return audio

#audio framing
def frame_audio(audio, sample_rate, FFT_size=2048, hop_size=10):
    # hop_size in ms
    
    audio = np.pad(audio, int(FFT_size / 2), mode='reflect')
    frame_len = np.round(sample_rate * hop_size / 1000).astype(int)
    frame_num = int((len(audio) - FFT_size) / frame_len) + 1
    frames = np.zeros((frame_num,FFT_size))
    
    for n in range(frame_num):
        frames[n] = audio[n*frame_len:n*frame_len+FFT_size]
    
    return frames

#Discrete Cosine Transform
def dct(dct_filter_num, filter_len):
    basis = np.empty((dct_filter_num,filter_len))
    basis[0, :] = 1.0 / np.sqrt(filter_len)
    
    samples = np.arange(1, 2 * filter_len, 2) * np.pi / (2.0 * filter_len)

    for i in range(1, dct_filter_num):
        basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / filter_len)
        
    return basis

#filter points
def freq_to_mel(freq):
    return 2595.0 * np.log10(1.0 + freq / 700.0)

def met_to_freq(mels):
    return 700.0 * (10.0**(mels / 2595.0) - 1.0)

def get_filter_points(fmin, fmax, mel_filter_num, FFT_size, sample_rate=44100):
    fmin_mel = freq_to_mel(fmin)
    fmax_mel = freq_to_mel(fmax)
    
    print("MEL min: {0}".format(fmin_mel))
    print("MEL max: {0}".format(fmax_mel))
    
    mels = np.linspace(fmin_mel, fmax_mel, num=mel_filter_num+2)
    freqs = met_to_freq(mels)
    
    return np.floor((FFT_size + 1) / sample_rate * freqs).astype(int), freqs   #return filter points , frequences

#filters construction
def get_filters(filter_points, FFT_size):
    filters = np.zeros((len(filter_points)-2,int(FFT_size/2+1)))
    
    for n in range(len(filter_points)-2):
        filters[n, filter_points[n] : filter_points[n + 1]] = np.linspace(0, 1, filter_points[n + 1] - filter_points[n])
        filters[n, filter_points[n + 1] : filter_points[n + 2]] = np.linspace(1, 0, filter_points[n + 2] - filter_points[n + 1])
    
    return filters

 #-----------------------------------------------------------------------------------------------------------------------------------------   

def image(fig,name):
   #canvas=FigureCanvas(fig)
   img=io.BytesIO()
   fig.figure.savefig(img, format='png')
   img.seek(0)
   # Embed the result in the html output.
   data = base64.b64encode(img.getbuffer()).decode("ascii")
   image_file_name='static/assets/img/'+str(name)+str(variables.counter)+'.jpg'
   plt.savefig(image_file_name)
#    return f"<img src='data:image/png;base64,{data}'/>"
   return image_file_name

def spectral_features(audio):
    signal , sample_rate = librosa.load(audio)
    spec_bw = librosa.feature.spectral_bandwidth(y=signal, sr=sample_rate)
    S, phase = librosa.magphase(librosa.stft(y=signal))
    centroid = librosa.feature.spectral_centroid(S=S)
    rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sample_rate, roll_percent=0.99)
    rolloff_min = librosa.feature.spectral_rolloff(y=signal, sr=sample_rate, roll_percent=0.01)
    fig, ax = plt.subplots()
    times = librosa.times_like(spec_bw)
    ax.semilogy(times, spec_bw[0], label='Spectral bandwidth')
    ax.legend()
    ax.label_outer()
    fig.patch.set_facecolor('#e4e8e8')
    img=librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                        y_axis='log', x_axis='time', ax=ax)
    ax.set(title='The Spectral Features ')
    ax.fill_between(times, np.maximum(0, centroid[0] - spec_bw[0]),
                np.minimum(centroid[0] + spec_bw[0], sample_rate/2),
                alpha=0.5, label='Centroid +- bandwidth')
    ax.plot(times, centroid[0], label='Spectral centroid', color='w')
    ax.plot(librosa.times_like(rolloff), rolloff[0], label='Roll-off frequency (0.99)')
    ax.plot(librosa.times_like(rolloff), rolloff_min[0], color='w',
        label='Roll-off frequency (0.01)')
    ax.legend(loc='lower right')
    fig.colorbar(img, ax=ax)
    img=image(fig,"spec")
    return img

def chroma(file):
    y, sr = librosa.load(file,res_type='kaiser_fast')
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    fig=plt.figure(figsize=(6,6))
    img = librosa.display.specshow(chroma_stft, y_axis='chroma', x_axis='time')
    plt.title('ChromaSTFT')
    plt.colorbar()
    chroma_fig = image(fig,'chroma')
    return chroma_fig


def rms(audio):
   signal,sample_rate=librosa.load(audio)
   S, phase = librosa.magphase(librosa.stft(signal))
   rms = librosa.feature.rms(S=S)
   fig, ax = plt.subplots()
   times = librosa.times_like(rms)
   ax.semilogy(times, rms[0], label='RMS Energy')
   ax.set(xticks=[])
   ax.legend()
   ax.label_outer()
   librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                           y_axis='log', x_axis='time', ax=ax)
   ax.set(title='log Power spectrogram')
   img=image(fig,"rms")
   return img

def voice_spec(file):
   fig,ax = plt.subplots(figsize=(6,6))
   plt.xlabel("Time (Sec)")
   plt.ylabel("Frequency (Hz)")
   plt.title("Spectrogram")
   ax=sns.set_style(style='darkgrid')
   sample_rate, signal = wavfile.read(file)
   # select left channel only
   signal = signal[:,0]
   # trim the first 125 seconds
   first = signal[:int(sample_rate*15)]
   powerSpectrum, frequenciesFound, time, imageAxis = plt.specgram(first, Fs=sample_rate)
   img=image(fig,"result")
   return img