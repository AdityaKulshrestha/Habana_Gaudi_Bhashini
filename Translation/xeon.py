import streamlit as st
import pandas as pd
import time
import os
import torch
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig

class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
st.set_page_config(layout="wide")
        
BATCH_SIZE = 1
DEVICE = "cpu"
quantization = None

def initialize_model_and_tokenizer(ckpt_dir, direction, quantization):
    if quantization == "4-bit":
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization == "8-bit":
        qconfig = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    else:
        qconfig = None

    tokenizer = IndicTransTokenizer(direction=direction)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=qconfig,
    )

    if qconfig == None:
        model = model.to(DEVICE)

    model.eval()

    return tokenizer, model


def indic_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip):
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
                use_cache=False,
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

@st.cache_resource
def load_model():
    en_indic_ckpt_dir = "ai4bharat/indictrans2-en-indic-1B"  # ai4bharat/indictrans2-en-indic-dist-200M
    en_indic_tokenizer, en_indic_model = initialize_model_and_tokenizer(en_indic_ckpt_dir, "en-indic", quantization)

    ip = IndicProcessor(inference=True)
    return en_indic_tokenizer, en_indic_model, ip 


en_indic_tokenizer, en_indic_model, ip = load_model()

src_lang, tgt_lang = "eng_Latn", "hin_Deva"       
        
# Set the page to wide layout


# Initialize the profanity masker

# Streamlit app
st.title("📑 ai4bharat/Translator")

# Description table as a single dictionary
description_table = {
    "Component": "Translator",
    "Description":"Enables accurate and contextually aware translation across multiple languages with precision.",
}

description_table2 = {
    "Model": "ai4bharat/indictrans2-en-indic-1B",
    
}

message = """
component Translator{
    service indic_translate{
        [in] string text;
        [out] string translated_text;
        [out] int error_code;
    }
}
"""

# Display the table with all details in the first row
st.table(description_table)

st.write("Interface Definition Language (IDL)")
# Print the message with the same indentation and format
st.code(message, language='plaintext')

st.table(description_table2)
src_lang, tgt_lang = "eng_Latn", "hin_Deva"  
# Performance section
performance_expander = st.expander("Performance", expanded=False)
with performance_expander:
    warmup_criteria = st.number_input("Enter warmup criteria:", min_value=0, value=10, step=1)
    runs_criteria = st.number_input("Enter runs criteria:", min_value=1, value=100, step=1)
    if st.button("Start Runs"):
        # Load the CSV file
        sentences_df = pd.read_csv('sentences.csv') # Assuming 'sentences.csv' is the name of your CSV file
        # Extract the required number of sentences for warmup
        warmup_sentences = sentences_df['text'].head(warmup_criteria).tolist()
        
        # Perform masking during the warmup phase without displaying anything
        for sentence in warmup_sentences:
            print("This is debugging", sentence)
            translated_sentence = indic_translate([sentence], src_lang, tgt_lang, en_indic_model,en_indic_tokenizer,ip)

            
        print("working")
        # Prepare to collect metrics for the runs criteria loop
        total_time = 0
        sentence_times = []
        
        # Start the runs criteria loop
        start_time = time.time()
        for i in range(runs_criteria):
            # Select a sentence for masking (you can modify this to use a different sentence for each run)
            sentence = sentences_df['text'].iloc[0]
            
            # Start the timer for this run
            sentence = [sentence]
            
            translated_sentence = indic_translate(sentence, src_lang, tgt_lang, en_indic_model,en_indic_tokenizer,ip)

            
            
            # Calculate performance metrics for this run
            
            
        total_time = time.time() - start_time
        # Calculate average time per sentence
        average_time_per_sentence = total_time / runs_criteria
        
        # Display the total time taken and the average time per sentence
        description_table1 = {
    "Total Time Taken": f"{total_time:.4f} seconds",
    "Average Time Per Sentence": f"{average_time_per_sentence:.4f} seconds",
}

        st.table(description_table1)

src_lang, tgt_lang = "eng_Latn", "hin_Deva"       


# Functionality section
functionality_expander = st.expander("Functionality", expanded=False)
with functionality_expander:
    
    default_sentence = "I am Shivpal and I work in Intel and I live in Bangalore"
    
    # Display the default sentence in the text input field
    user_input = st.text_input("Enter your sentence here:", default_sentence)
    user_input = [user_input]
    if st.button("📑Translate"):
        if user_input:
            translated_sentence = indic_translate(user_input, src_lang, tgt_lang, en_indic_model,en_indic_tokenizer,ip)
            st.write("Translated Sentence:")
            st.write(translated_sentence)
            
        else:
            st.write("Please enter a sentence.")
