import os
import librosa
import random
import torch
import evaluate
import torchaudio
import numpy as np
import pandas as pd
from datasets import Dataset
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from matplotlib import pyplot as plt


from transformers import WhisperTokenizer, WhisperProcessor
from model.feature_extraction_whisper import WhisperFeatureExtractor
from model.modeling_whisper import WhisperForConditionalGeneration

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('device',device)

mp3_path = "./dataset/clips"
data_tsv = "./dataset/inference_purpose.tsv"

model_path = 'saved_train'

def read_data(data_path, data_chunk = 0):
    if data_chunk==0:
        data = pd.read_csv(data_path, sep='\t')
    else:
        data = pd.read_csv(data_path, sep='\t')[:data_chunk]
    data.columns = ["client_id", "path", "sentence", "up_votes", "down_votes", "age", "gender", "accents", "locale", "segment"]
    data = data.drop(["client_id","up_votes", "down_votes", "age", "gender", "accents", "locale", "segment"], axis=1)
    return data

data = read_data(data_tsv, data_chunk=0)

def add_file_path(path):
    audio_path = os.path.join(mp3_path, path)
    return audio_path

data['path'] = data['path'].map(lambda x: add_file_path(x))


feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
tokenizer = WhisperTokenizer.from_pretrained(model_path+'/tokenizer')
processor = WhisperProcessor.from_pretrained(model_path+'/tokenizer')
model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)

def prepare_dataset(path):
    speech_array, sampling_rate = torchaudio.load(path, format="mp3")
    speech_array = speech_array[0].numpy()
    speech_array = librosa.resample(np.asarray(speech_array), orig_sr=sampling_rate, target_sr=16000)
    input_features = feature_extractor(speech_array, sampling_rate=16000, return_tensors="pt").input_features
    return speech_array, input_features

audio_list, sentence =data["path"].tolist(), data["sentence"].tolist()

for index, sen in enumerate(audio_list):
    audio_file, sentence1 = audio_list[index], sentence[index]
    speech_array, input_features = prepare_dataset(audio_file)


    np_feature = input_features[0].detach().cpu().numpy()
    predicted_ids = model.generate(inputs=input_features.to(device))[0]
    transcription = processor.decode(predicted_ids, skip_special_tokens=True)

    print('gt           ', sentence1)
    print('transcription', transcription)
