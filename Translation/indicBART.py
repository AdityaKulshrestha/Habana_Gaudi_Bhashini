from transformers import AutoTokenizer, AutoModelForTokenClassification

from transformers import AutoTokenizer, BertForMaskedLM
import torch
import os
os.environ['PT_HPU_ENABLE_GENERIC_STREAM'] = '1'
os.environ['PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES'] = '0'
import habana_frameworks.torch as ht
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu.graphs as htgraphs

hpu = torch.device('hpu')
cpu = torch.device('cpu')

tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicNER")
model = AutoModelForTokenClassification.from_pretrained("ai4bharat/IndicNER")

inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")
inputs = inputs.to(hpu)
model = model.eval()

model = htgraphs.wrap_in_hpu_graph(model)
model = model.to(hpu)

with torch.no_grad():
    logits = model(**inputs).logits

# retrieve index of [MASK]
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
print(tokenizer.decode(predicted_token_id))

labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]
# mask labels of non-[MASK] tokens
labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)

outputs = model(**inputs, labels=labels)
round(outputs.loss.item(), 2)