import torch
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer
import os
os.environ['PT_HPU_ENABLE_GENERIC_STREAM'] = '1'
os.environ['PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES'] = '0'
import habana_frameworks.torch as ht
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu.graphs as htgraphs

BATCH_SIZE = 4
DEVICE = torch.device('hpu')


activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU]

en_indic_ckpt_dir = "ai4bharat/indictrans2-en-indic-1B"  # ai4bharat/indictrans2-en-indic-dist-200M
direction = "en-indic"


tokenizer = IndicTransTokenizer(direction=direction)
model = AutoModelForSeq2SeqLM.from_pretrained(
        en_indic_ckpt_dir,
        trust_remote_code=True,
    )

model = htgraphs.wrap_in_hpu_graph(model)
model = model.to(DEVICE)


model.eval()

def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip):
    translations = []
    for i in range(0, len(input_sentences), BATCH_SIZE):
        batch = input_sentences[i : i + BATCH_SIZE]

        # Preprocess the batch and extract entity mappings
        batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)

        # Tokenize the batch and generate input encodings
        inputs = tokenizer(
            batch,
            src=True,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(DEVICE)

        # Generate translations using the model
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )

        # Decode the generated tokens into text
        generated_tokens = tokenizer.batch_decode(generated_tokens.detach().cpu().tolist(), src=False)

        # Postprocess the translations, including entity replacement
        translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)

    return translations

ip = IndicProcessor(inference=True)

en_sents = [
    "When I was young, I used to go to the park every day.",
    "He has many old books, which he inherited from his ancestors.",
    "I can't figure out how to solve my problem.",
    "She is very hardworking and intelligent, which is why she got all the good marks.",
    "We watched a new movie last week, which was very inspiring.",
    #"If you had met me at that time, we would have gone out to eat.",
    #"She went to the market with her sister to buy a new sari.",
    #"Raj told me that he is going to his grandmother's house next month.",
    #"All the kids were having fun at the party and were eating lots of sweets.",
    #"My friend has invited me to his birthday party, and I will give him a gift.",
]
src_lang, tgt_lang = "eng_Latn", "hin_Deva"

# do we need to shift en_sent to hpu
with torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=0, warmup=10, active=5, repeat=1),
    activities=activities,
    on_trace_ready=torch.profiler.tensorboard_trace_handler('logs_new')) as profiler:
    
    for i in range(30):
        hi_translations = batch_translate(en_sents, src_lang, tgt_lang, model, tokenizer, ip)
        
    
    # translations = []
    # for i in range(0, len(input_sentences), BATCH_SIZE):
    #     batch = input_sentences[i : i + BATCH_SIZE]

    #     # Preprocess the batch and extract entity mappings
    #     batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)

    #     # Tokenize the batch and generate input encodings
    #     inputs = tokenizer(
    #         batch,
    #         src=True,
    #         truncation=True,
    #         padding="longest",
    #         return_tensors="pt",
    #         return_attention_mask=True,
    #     ).to(DEVICE)

    #     # Generate translations using the model
    #     with torch.no_grad():
    #         generated_tokens = model.generate(
    #             **inputs,
    #             use_cache=True,
    #             min_length=0,
    #             max_length=256,
    #             num_beams=5,
    #             num_return_sequences=1,
    #         )

    #     # Decode the generated tokens into text
    #     generated_tokens = tokenizer.batch_decode(generated_tokens.detach().cpu().tolist(), src=False)

    #     # Postprocess the translations, including entity replacement
    #     translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)


        htcore.mark_step()
        profiler.step()
        