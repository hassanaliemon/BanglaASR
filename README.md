# BanglaASR

I have fine-tuned whisper small model for bangla dataset of around 23.8 hours and it provides Word Error Rate of around 14.73 which is good for a small model with a small dataset. There is a lot of room to improve the efficiency of the model.

Future Work:

1. training with more data like combining with other data of mozila open source dataset
2. Analyzing and find error prone or lower pronunciated data and trian accordingly
3. Train other whisper models like `Medium` or `Large`
4. Web integration


# Dataset
Download the dataset from [here](https://commonvoice.mozilla.org/bn/datasets), create a folder named `dataset` and put it in `dataset` folder

# Requirements
Please check `requirements.txt` file

# Train
Run `python train.py` to start the training process.

# Infer
Download pretrained model from [huggingface](https://huggingface.co/hassanaliemon/BanglaASR), and set model path and run `python single_test.py` <br>
or run the following code
```python
import librosa
import torch
import torchaudio
import numpy as np

from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from transformers import WhisperFeatureExtractor
from transformers import WhisperForConditionalGeneration

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

audio_path = "https://huggingface.co/hassanaliemon/BanglaASR/resolve/main/test_audio/common_voice_bn_31255511.mp3"
model_path = "hassanaliemon/BanglaASR"


feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
tokenizer = WhisperTokenizer.from_pretrained(model_path)
processor = WhisperProcessor.from_pretrained(model_path)
model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)


speech_array, sampling_rate = torchaudio.load(audio_path, format="mp3")
speech_array = speech_array[0].numpy()
speech_array = librosa.resample(np.asarray(speech_array), orig_sr=sampling_rate, target_sr=16000)
input_features = feature_extractor(speech_array, sampling_rate=16000, return_tensors="pt").input_features

predicted_ids = model.generate(inputs=input_features.to(device))[0]
transcription = processor.decode(predicted_ids, skip_special_tokens=True)

print(transcription)

```

# Note
Feel free to provide your valuable feed back or connect with me at [linkedin](https://www.linkedin.com/in/hassan-ali-emon/)