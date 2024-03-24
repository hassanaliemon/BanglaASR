# BanglaASR

I have fine-tuned whisper small model for bangla dataset and it provides Word Error Rate of around 14.73 which is good for a small model with a small dataset. WER can be decreased by:

1. training with more data like combining with other data of mozila open source dataset
2. Analyzing and find error prone or lower pronunciated data and trian accordingly

These are future work

# Dataset
Download the dataset from [here](https://commonvoice.mozilla.org/bn/datasets) and put it in `dataset` folder

# Train
Run `python train.py` to start the training process.
You can also

# Infer
Download pretrained model from [huggingface](https://huggingface.co/hassanaliemon/BanglaASR), and set model path and run `python single_test.py`

# Note
this repo will be updated very soon, till then if you need any help please knock me at [linkedin](https://www.linkedin.com/in/hassan-ali-emon/)