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

# Environment variables
# Note these need to be set before loading habana_framworks package
# Please do not move these from here
os.environ['PT_HPU_ENABLE_GENERIC_STREAM'] = '1'
os.environ['PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES'] = '0'
import habana_frameworks.torch as ht
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu.graphs as htgraphs

hpu = torch.device('hpu')
cpu = torch.device('cpu')


processor = Wav2Vec2Processor.from_pretrained("ai4bharat/indicwav2vec-hindi")
model = Wav2Vec2ForCTC.from_pretrained("ai4bharat/indicwav2vec-hindi")

model = model.eval()

model = htgraphs.wrap_in_hpu_graph(model)
#model = model.to(hpu)

ds = load_dataset("mozilla-foundation/common_voice_16_0", "hi", split="validation")

sampling_rate = ds.features['audio'].sampling_rate

print("Sampling rate of the dataset", sampling_rate)


# Bucketing technique in Habana 
print("Bucketing Input data")
limit = len(ds)
lengths = [len(ds[i]['audio']['array']) for i in range(0, limit)]
lengths.sort() 
print("max_length", lengths[-1])


def predict(audio):
    resampled_audio = F.resample(audio, 48000, 16000).numpy()
    input_values = processor(resampled_audio, sampling_rate=16000, return_tensors="pt", padding='max_length', truncation=True, max_length = lengths[-1]).input_values
    input_values = input_values.to(hpu, non_blocking = True)

    with torch.autocast(device_type="hpu"):
        logits = model(input_values).logits.to(cpu, non_blocking = True)
        
        
    predicted_ids = torch.argmax(logits, dim = -1) 
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

# Warming up the model. 
# start_time = time.time()
# output = predict(torch.tensor(ds[1]['audio']['array']))
# print(output)
# print("Warming up the model time taken", time.time() - start_time)


activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU]

with torch.profiler.profile(
    schedule = torch.profiler.schedule(wait=0, warmup=20, active = 5, repeat = 1),
    activities = activities, 
    on_trace_ready=torch.profiler.tensorboard_trace_handler('logs_new'),
    profile_memory=True
) as profiler:
    for i in range(100):
        model = model.to(hpu)
        output = predict(torch.tensor(ds[i]['audio']['array']))
        print(output)
        htcore.mark_step()
        profiler.step()
        

    

    








