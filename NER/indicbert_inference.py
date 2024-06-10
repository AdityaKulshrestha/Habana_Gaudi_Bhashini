import time 
import torch 
import os 

from transformers import AutoTokenizer, AutoModelForTokenClassification
from habana_frameworks.torch.dynamo.compile_backend.experimental import enable_compiled_autograd
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu.graphs as htgraphs

# os.environ['PT_HPU_ENABLE_GENERIC_STREAM'] = '1'
# os.environ['PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES'] = '0'
# Throwed error core memory dump!

#enable_compiled_autograd()

hpu = torch.device('hpu')
cpu = torch.device('cpu')


tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicNER")

model = AutoModelForTokenClassification.from_pretrained("ai4bharat/IndicNER")

model = model.eval()

model = htgraphs.wrap_in_hpu_graph(model)
model = model.to(hpu)

def get_predictions( sentence, tokenizer, model ):
  # Let us first tokenize the sentence - split words into subwords
  tok_sentence = tokenizer(sentence, return_tensors='pt', padding = 'max_length', max_length = 512)

  with torch.autocast(device_type="hpu"):
    # we will send the tokenized sentence to the model to get predictions
    logits = model(**tok_sentence).logits.argmax(-1).to(cpu, non_blocking = True)
    
    # We will map the maximum predicted class id with the class label
    predicted_tokens_classes = [model.config.id2label[t.item()] for t in logits[0]]
    
    predicted_labels = []
    
    previous_token_id = 0
    # we need to assign the named entity label to the head word and not the following sub-words
    word_ids = tok_sentence.word_ids()
    for word_index in range(len(word_ids)):
        if word_ids[word_index] == None:
            previous_token_id = word_ids[word_index]
        elif word_ids[word_index] == previous_token_id:
            previous_token_id = word_ids[word_index]
        else:
            predicted_labels.append( predicted_tokens_classes[ word_index ] )
            previous_token_id = word_ids[word_index]
    
    return predicted_labels


sentence = 'लगातार हमलावर हो रहे शिवपाल और राजभर को सपा की दो टूक, चिट्ठी जारी कर कहा- जहां जाना चाहें जा सकते हैं'

start_time = time.time()
predicted_labels = get_predictions(sentence=sentence, 
                                   tokenizer=tokenizer,
                                   model=model
                                   )

for index in range(len(sentence.split(' '))):
  print( sentence.split(' ')[index] + '\t' + predicted_labels[index] )
  
print("\n\n Time taken : ", time.time() - start_time)


sentence = 'ಶರಣ್ ರ ನೀವು ನೋಡಲೇಬೇಕಾದ ಟಾಪ್ 5 ಕಾಮಿಡಿ ಚಲನಚಿತ್ರಗಳು'

start_time = time.time()
predicted_labels = get_predictions(sentence=sentence, 
                                   tokenizer=tokenizer,
                                   model=model
                                   )

for index in range(len(sentence.split(' '))):
  print( sentence.split(' ')[index] + '\t' + predicted_labels[index] )

print("\n\n Time taken : ", time.time() - start_time)
