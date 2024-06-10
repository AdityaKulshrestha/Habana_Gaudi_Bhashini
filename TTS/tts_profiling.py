import io
import sys 
sys.path.append('/YOUR_ABSOLUTE_PATH/Habana_Gaudi_Bhashini/TTS/TTS')
import torch 
from TTS.utils.synthesizer import Synthesizer
from Indic_TTS.inference.src.inference import TextToSpeechEngine
import scipy.io.wavfile
import habana_frameworks.torch as ht
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu.graphs as htgraphs

DEFAULT_SAMPLING_RATE = 16000

activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU]


lang = "bn"
ta_model  = Synthesizer(
    tts_checkpoint=f'checkpoints/{lang}/fastpitch/best_model.pth',
    tts_config_path=f'checkpoints/{lang}/fastpitch/config.json',
    tts_speakers_file=f'checkpoints/{lang}/fastpitch/speakers.pth',
    # tts_speakers_file=None,
    tts_languages_file=None,
    vocoder_checkpoint=f'checkpoints/{lang}/hifigan/best_model.pth',
    vocoder_config=f'checkpoints/{lang}/hifigan/config.json',
    encoder_checkpoint="",
    encoder_config="",
    use_cuda=True,
)

# Setup TTS Engine

models = {
    "bn": ta_model,
}
engine = TextToSpeechEngine(models)

# Bengali TTS inference

with torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=0, warmup=1, active=2, repeat=1),
    activities=activities,
    on_trace_ready=torch.profiler.tensorboard_trace_handler('logs'), 
    profile_memory=True) as profiler:
    
    
    for i in range(10):
        hindi_raw_audio = engine.infer_from_text(
            input_text="নমস্কার, আপনি কেমন আছেন?",
            lang="bn",
            speaker_name="male"
        )
        htcore.mark_step()
        profiler.step()
# byte_io = io.BytesIO()
# scipy.io.wavfile.write(byte_io, DEFAULT_SAMPLING_RATE, hindi_raw_audio)

# with open("bn_audio.wav", "wb") as f:
#     f.write(byte_io.read())
