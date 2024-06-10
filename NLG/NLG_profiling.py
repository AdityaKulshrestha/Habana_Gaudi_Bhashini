from transformers import MBartForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers import AlbertTokenizer, AutoTokenizer
import os
import torch
os.environ['PT_HPU_ENABLE_GENERIC_STREAM'] = '1'
os.environ['PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES'] = '0'
import habana_frameworks.torch as ht
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu.graphs as htgraphs

activities = [torch.profiler.ProfilerActivity.CPU]
activities.append(torch.profiler.ProfilerActivity.HPU)

hpu = torch.device('hpu')
cpu = torch.device('cpu')

tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBART", do_lower_case=False, use_fast=False, keep_accents=True)

# Or use tokenizer = AlbertTokenizer.from_pretrained("ai4bharat/IndicBART", do_lower_case=False, use_fast=False, keep_accents=True)

# Or use model = MBartForConditionalGeneration.from_pretrained("ai4bharat/IndicBART")

model = MBartForConditionalGeneration.from_pretrained("ai4bharat/IndicBART")

# Some initial mapping
bos_id = tokenizer._convert_token_to_id_with_added_voc("<s>")
eos_id = tokenizer._convert_token_to_id_with_added_voc("</s>")
pad_id = tokenizer._convert_token_to_id_with_added_voc("<pad>")
# To get lang_id use any of ['<2as>', '<2bn>', '<2en>', '<2gu>', '<2hi>', '<2kn>', '<2ml>', '<2mr>', '<2or>', '<2pa>', '<2ta>', '<2te>']

# First tokenize the input and outputs. The format below is how IndicBART was trained so the input should be "Sentence </s> <2xx>" where xx is the language code. Similarly, the output should be "<2yy> Sentence </s>".
inp = tokenizer("I am a boy </s> <2en>", add_special_tokens=False, return_tensors="pt", padding=True).input_ids # tensor([[  466,  1981,    80, 25573, 64001, 64004]])
inp = inp.to(hpu)
out = tokenizer("<2hi> मैं  एक लड़का हूँ </s>", add_special_tokens=False, return_tensors="pt", padding=True).input_ids # tensor([[64006,   942,    43, 32720,  8384, 64001]])
# Note that if you use any language other than Hindi or Marathi, you should convert its script to Devanagari using the Indic NLP Library.

model_outputs=model(input_ids=inp, decoder_input_ids=out[:,0:-1], labels=out[:,1:])

# For loss
model_outputs.loss ## This is not label smoothed.

# For logits
model_outputs.logits

# For generation. Pardon the messiness. Note the decoder_start_token_id.

model.eval() # Set dropouts to zero
model = htgraphs.wrap_in_hpu_graph(model)

with torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=0, warmup=20, active=5, repeat=1),
    activities=activities,
    on_trace_ready=torch.profiler.tensorboard_trace_handler('logs'),profile_memory=True) as profiler:
    for i in range(100):
        model = model.to(hpu)
        inp = tokenizer("I am a boy </s> <2en>", add_special_tokens=False, return_tensors="pt", padding=True).input_ids # tensor([[  466,  1981,    80, 25573, 64001, 64004]])
        inp = inp.to(hpu)
        model_output=model.generate(inp, use_cache=True, num_beams=1, max_length=20, min_length=1, early_stopping=True, pad_token_id=pad_id, bos_token_id=bos_id, eos_token_id=eos_id, decoder_start_token_id=tokenizer._convert_token_to_id_with_added_voc("<2en>"))
        decoded_output=tokenizer.decode(model_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(decoded_output) # I am a boy

        htcore.mark_step()
        profiler.step()

        inp = tokenizer("I am [MASK] </s> <2en>", add_special_tokens=False, return_tensors="pt", padding=True).input_ids.to(hpu)
        model_output=model.generate(inp, use_cache=True, num_beams=4, max_length=20, min_length=1, early_stopping=True, pad_token_id=pad_id, bos_token_id=bos_id, eos_token_id=eos_id, decoder_start_token_id=tokenizer._convert_token_to_id_with_added_voc("<2en>"))
        decoded_output=tokenizer.decode(model_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(decoded_output) # I am happy

        htcore.mark_step()
        profiler.step()

        inp = tokenizer("मैं [MASK] हूँ </s> <2hi>", add_special_tokens=False, return_tensors="pt", padding=True).input_ids.to(hpu)
        model_output=model.generate(inp, use_cache=True, num_beams=4, max_length=20, min_length=1, early_stopping=True, pad_token_id=pad_id, bos_token_id=bos_id, eos_token_id=eos_id, decoder_start_token_id=tokenizer._convert_token_to_id_with_added_voc("<2en>"))
        decoded_output=tokenizer.decode(model_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(decoded_output) # मैं जानता हूँ

        htcore.mark_step()
        profiler.step()

        inp = tokenizer("मला [MASK] पाहिजे </s> <2mr>", add_special_tokens=False, return_tensors="pt", padding=True).input_ids.to(hpu)
        model_output=model.generate(inp, use_cache=True, num_beams=4, max_length=20, min_length=1, early_stopping=True, pad_token_id=pad_id, bos_token_id=bos_id, eos_token_id=eos_id, decoder_start_token_id=tokenizer._convert_token_to_id_with_added_voc("<2en>"))
        decoded_output=tokenizer.decode(model_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

        print(decoded_output) # मला ओळखलं पाहिजे

        htcore.mark_step()
        profiler.step()
