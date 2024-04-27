import torch
import os
os.environ['PT_HPU_ENABLE_GENERIC_STREAM'] = '1'
os.environ['PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES'] = '0'
import habana_frameworks.torch as ht
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu.graphs as htgraphs
from ai4bharat.transliteration import XlitEngine

INPUT_WORD = "namaste kaise hai aap"
print("Input word:", INPUT_WORD)


# xlit_engine = XlitEngine("hi")
# res = xlit_engine.translit_word(INPUT_WORD)
# print("Hindi output:", res)
activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU]


with torch.profiler.profile(
    schedule = torch.profiler.schedule(wait=1, warmup=5, active = 2, repeat = 1),
    activities = activities, 
    on_trace_ready=torch.profiler.tensorboard_trace_handler('logs')
) as profiler:
    
    e = XlitEngine("hi")
    for i in range(50):
        out = e.translit_sentence(INPUT_WORD)
        htcore.mark_step()
        profiler.step()