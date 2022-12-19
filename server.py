from email import message
from flask import Flask, redirect, url_for, request,render_template,flash,session
import numpy as np
# from sympy import Id
import pickle
import os
#import speech_recognition as spr
import pickle
import librosa
import sounddevice as sd
import wavio as wv
import model
import pydotplus
from sklearn.datasets import load_iris
from sklearn import tree
import graphviz
from dtreeviz.trees import *


app = Flask(__name__,template_folder="templates")

@app.route('/')
def hello_name():
   return render_template('index.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      return 'file uploaded successfully'
def prepare_testing(to_test):
    features=[]
    # reading records
    y, sr = librosa.load(to_test,res_type='kaiser_fast')
    print("Y1", len(y))
    # remove leading and trailing silence
    y, index    = librosa.effects.trim(y)
    print("Y2", len(y))

    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse        = librosa.feature.rms(y=y)
    spec_cent   = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw     = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff     = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr         = librosa.feature.zero_crossing_rate(y)
    mfcc        = librosa.feature.mfcc(y=y, sr=sr)
    to_append   = f'{np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
    for e in mfcc:
        to_append += f' {np.mean(e)}'
    features.append(to_append.split())
    print(features)
    
    for counter in range(0,len(features[0])):
        features[0][counter]=float(features[0][counter])

    return features

def test_model (wav_file):
    # wav_file='k_close_2.wav'
    
    features=prepare_testing(wav_file)
    # model= pickle.load(open('Sentence-model.pkl','rb'))
    model= pickle.load(open('Sentence-model2.pkl','rb'))
    model_output =model.predict(features)

    if model_output==0:
        # result='Close the door'
        result='False'
    elif model_output==1:
        # result='Open the door'
        result='True'
    else:
        result=''
        

    return result


    
def get_model_path(features):
    # clf = pickle.load(open("Speech2-model.pkl",'rb'))
    clf = pickle.load(open("Speech3-model.pkl",'rb'))
    # clf = clf.estimators_[5]
    dot_data = tree.export_graphviz(clf, out_file=None,
                                filled=True, rounded=True,
                                special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)

    # empty all nodes, i.e.set color to white and number of samples to zero
    for node in graph.get_node_list():
        if node.get_attributes().get('label') is None:
            continue
        if 'samples = ' in node.get_attributes()['label']:
            labels = node.get_attributes()['label'].split('<br/>')
            for i, label in enumerate(labels):
                if label.startswith('samples = '):
                    labels[i] = 'samples = 0'
            node.set('label', '<br/>'.join(labels))
            node.set_fillcolor('grey')

    samples = features
    samples = np.array(samples)
    decision_paths = clf.decision_path(samples.reshape(1,-1))

    for decision_path in decision_paths:
        for n, node_value in enumerate(decision_path.toarray()[0]):
            if node_value == 0:
                continue
            node = graph.get_node(str(n))[0]            
            node.set_fillcolor('orange')
            labels = node.get_attributes()['label'].split('<br/>')
            for i, label in enumerate(labels):
                if label.startswith('samples = '):
                    labels[i] = 'samples = {}'.format(int(label.split('=')[1]) + 1)

            node.set('label', '<br/>'.join(labels))

    filename = 'static/assets/img/tree.png'
    graph.write_png(filename)
    return filename

def get_path_as_Histogram(features):
    X = pickle.load(open("X_data",'rb'))
    y = pickle.load(open("Y_data",'rb'))
    clf = pickle.load(open("Speech3-model.pkl",'rb'))
    feature_names = ['mfccs', 'chroma', 'mel', 'contrast', 'tonnetz']*len(X)
    # feature_names = arrange_feat(DataSet)
    target_names = ["others", 'Mahmoud Hamdy','Sherif', 'yassmen', 'bassma']
    print(feature_names[0])
    viz = dtreeviz(clf,
                np.array(X), 
                np.array(y),
                target_name="Member",
                feature_names = feature_names,
                class_names= target_names, 
                title="Wine data set regression",
                fontname="Arial",
                scale=1.5,
                X=features[0])
                # fancy=False)
    filename = 'static/assets/img/Histogramtree.svg'
    viz.save(filename)
    return filename


def predict_sound(file_name):
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    X, index    = librosa.effects.trim(X)
    # Generate Mel-frequency cepstral coefficients (MFCCs) from a time series 
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    # Generates a Short-time Fourier transform (STFT) to use in the chroma_stft
    stft = np.abs(librosa.stft(X))
    # Computes a chromagram from a waveform or power spectrogram.
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    # Computes a mel-scaled spectrogram.
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    # Computes spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    # Computes the tonal centroid features (tonnetz)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    sr=sample_rate).T,axis=0)
    features=[]
    features.append(np.concatenate((mfccs, chroma, mel, contrast, tonnetz),axis=0))
    path = get_model_path(features)
    path2 = get_path_as_Histogram(features)
    # open_model = pickle.load(open("Speech2-model.pkl",'rb'))
    open_model = pickle.load(open("Speech3-model.pkl",'rb'))
    result =open_model.predict(features)[0]
    members = ["others", 'Mahmoud Hamdy','Sherif', 'yassmen', 'bassma']
    try:
        result = members[result]
    except:
        print(result)
        result = "There's an error in recording please try again"
    return result, path, path2



@app.route("/", methods=["GET", "POST"])
def index():
        speech =''
        speaker =''
        file_name=''
        model_path = ''
        y=[]
        sr=[]
        chroma_fig=''
        rms_fig=''
        spectrum=''
        if request.method == "POST":
            record = request.form["recording_icon"]
            if(record!=0):
                # Sampling frequency
                frequency = 44400
                # Recording duration in seconds
                duration = 2
                # to record audio from
                # sound-device into a Numpy
                recording = sd.rec(int(duration * frequency),samplerate = frequency, channels = 2)
                # Wait for the audio to complete
                sd.wait()
                # using wavio to save the recording in .wav format
                # This will convert the NumPy array to an audio
                # file with the given sampling frequency
                wv.write("result.wav", recording, frequency, sampwidth=2)
                file_name="result.wav"
                y, sr = librosa.load(file_name)
                if(len(y)!=0):
                    speaker, model_path, m_path2=predict_sound("result.wav")
                    if(speaker != 'others'):
                        speaker = "Hello "+speaker
                else:
                    speaker = 'Please record audio'

                speech=test_model("result.wav")
                # if(speech=='Open the door'):
                #     # speaker = "True"
                #     speech='True'
                # else:
                #     speech="False"
                #     speaker = ' '

                # get audio from the microphone                                                                       
                # r = spr.Recognizer()                                                                                   
                # with spr.Microphone() as source:                                                                       
                #     # print("Speak:")                                                                                   
                #     audio = r.listen(source)   

                # try:
                #     speech = r.recognize_google(audio)
                #     if(speech != 'Open the door' or speech != "open the door"):
                #         speech = 'Wrong password'
                #     print("You said " + r.recognize_google(audio))
                # except spr.UnknownValueError:
                #     speech = "Could not understand audio"
                #     print("Could not understand audio")
                # except spr.RequestError as e:
                #     print("Could not request results; {0}".format(e))
                # y, sr = librosa.load(file_name)
                # rms_fig=model.rms("result.wav")
                # rms_fig='static/assets/img/rms'+str(model.variables.counter)+'.jpg'
                # chroma_fig=model.chroma("result.wav")
                # chroma_fig='static/assets/img/chroma'+str(model.variables.counter)+'.jpg'
                # spectrum= model.voice_spec("result.wav")
                # spectrum='static/assets/img/result'+str(model.variables.counter)+'.jpg'
                # model.variables.counter+=1
                Total_spec = model.spectral_features(file_name)
                print("Pic Path",Total_spec)
        # return render_template('index.html',speaker=speaker, 
        # speech=speech,chroma_fig=chroma_fig,rms_fig=rms_fig,spectrum=spectrum, model_path=model_path)
        return render_template('index.html', speech=speech,speaker=speaker,file_name=file_name,
            TSpec = Total_spec,  model_path=model_path, m_path2=m_path2)

if __name__ == '__main__':
    app.run(debug=True)   
