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
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
from model.tokenization_whisper import WhisperTokenizer
from model.processing_whisper import WhisperProcessor
from model.feature_extraction_whisper import WhisperFeatureExtractor
from model.modeling_whisper import WhisperForConditionalGeneration



# common_voice = Data

model_type = "openai/whisper-small"

mp3_path = "./dataset/clips"

train_tsv = "./dataset/train.tsv"
val_tsv = "./dataset/dev.tsv"


def read_data(data_path, data_chunk = 0):
    if data_chunk==0:
        data = pd.read_csv(data_path, sep='\t')
    else:
        data = pd.read_csv(data_path, sep='\t')[:data_chunk]
    data.columns = ["client_id", "path", "sentence", "up_votes", "down_votes", "age", "gender", "accents", "locale", "segment"]
    data = data.drop(["client_id","up_votes", "down_votes", "age", "gender", "accents", "locale", "segment"], axis=1)
    return data

train_data = read_data(train_tsv, data_chunk=0)
val_data = read_data(val_tsv, data_chunk=0)


def add_file_path(path):
    return os.path.join(mp3_path, path)


# read data from tsv file

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_type)
tokenizer = WhisperTokenizer.from_pretrained(model_type, language="Bengali", task="transcribe")
processor = WhisperProcessor.from_pretrained(model_type, language="Bengali", task="transcribe")



def prepare_dataset(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"], format="mp3")
    speech_array = speech_array[0].numpy()
    speech_array = librosa.resample(np.asarray(speech_array), orig_sr=sampling_rate, target_sr=16000)
    batch["input_features"] = feature_extractor(speech_array, sampling_rate=16000).input_features[0]
    batch["sampling_rate"] = 16000
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

    

train_data['path'] = train_data['path'].map(lambda x: add_file_path(x))

val_data['path'] = val_data['path'].map(lambda x: add_file_path(x))


train_data = Dataset.from_pandas(train_data)
val_data = Dataset.from_pandas(val_data)


train_data2 = train_data.map(prepare_dataset, num_proc=4)
val_data2 = val_data.map(prepare_dataset, num_proc=4)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch



data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}



model = WhisperForConditionalGeneration.from_pretrained(model_type)

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []


training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-bn",  # change to a repo name of your choice
    per_device_train_batch_size=26, #16
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=12000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=26,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=500,
    eval_steps=500,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_data2,
    eval_dataset=val_data2,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

processor.save_pretrained(training_args.output_dir)

trainer.train()#provide a checkpoint path if you want to
