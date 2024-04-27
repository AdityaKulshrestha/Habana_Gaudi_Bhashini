import time
import os
import argparse
import numpy
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import torch
import csv
import multiprocessing
from threading import Lock
import torchaudio.functional as F # For resampling the audio 
from habana_frameworks.torch.hpex.experimental.transformer_engine import recipe
import habana_frameworks.torch.hpex.experimental.transformer_engine as te
import habana_quantization_toolkit
os.environ['PT_HPU_ENABLE_GENERIC_STREAM'] = '1'
os.environ['PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES'] = '0'
import habana_frameworks.torch as ht
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu.graphs as htgraphs

fp8_recipe = recipe.DelayedScaling(margin=0, interval=1)
htcore.hpu_set_env()

# Environment variables
# Note these need to be set before loading habana_framworks package
# Please do not move these from here


hpu = torch.device('hpu')
cpu = torch.device('cpu')


processor = Wav2Vec2Processor.from_pretrained("ai4bharat/indicwav2vec-hindi")
model = Wav2Vec2ForCTC.from_pretrained("ai4bharat/indicwav2vec-hindi")

habana_quantization_toolkit.prep_model(model)
htcore.hpu_initialize(model)

model = model.eval()
model = htgraphs.wrap_in_hpu_graph(model)
model = model.to(hpu)

ds = load_dataset("mozilla-foundation/common_voice_16_0", "hi", split="validation")

sampling_rate = ds.features['audio'].sampling_rate

print("Sampling rate of the dataset", sampling_rate)


# Bucketing technique in Habana 
print("Bucketing Input data")
limit = len(ds)
lengths = [len(ds[i]['audio']['array']) for i in range(0, limit)]
lengths.sort() 
print("max_length", lengths[-1])


# Predicting - Running Completely fine 
# resampled_audio = F.resample(torch.tensor(ds[0]['audio']['array']), 48000, 16000).numpy()
# input_values = processor(resampled_audio, sampling_rate=16000, return_tensors="pt", padding='max_length', truncation=True, max_length = lengths[-1]).input_values
# input_values = input_values.to(hpu, non_blocking = True)
# with torch.autocast(device_type="hpu", dtype = torch.bfloat16):
#     logits = model(input_values).logits.to(cpu, non_blocking = False)
    
# predicted_ids = torch.argmax(logits, dim = -1) 
# print("Prediction", predicted_ids)
# transcription = processor.batch_decode(predicted_ids)[0]
# print(transcription)

def predict(audio):
    
    resampled_audio = F.resample(audio, 48000, 16000).numpy()
    input_values = processor(resampled_audio, sampling_rate=16000, return_tensors="pt", padding='max_length', truncation=True, max_length = lengths[-1]).input_values
    input_values = input_values.to(hpu, non_blocking = True)
    
    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe, force_measurement=True):
        
        logits = model(input_values).logits.to(cpu, non_blocking = True) 
        
    habana_quantization_toolkit.finish_measurements(model)
    
    #with torch.autocast(device_type="hpu"):
        #logits = model(input_values).logits.to(cpu, non_blocking = True) 
        
    predicted_ids = torch.argmax(logits, dim = -1) 
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

start_time = time.time()
output = predict(torch.tensor(ds[1]['audio']['array']))
print(output)
print("Warming up the model time taken", time.time() - start_time)

start_time = time.time() 
output = predict(torch.tensor(ds[3]['audio']['array']))
print(output)
print("Time to process the audio", time.time() - start_time)

start_time = time.time() 
output = predict(torch.tensor(ds[5]['audio']['array']))
print(output)
print("Time to process the audio", time.time() - start_time)