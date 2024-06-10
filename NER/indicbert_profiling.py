import time 
import torch 
import os 

from transformers import AutoTokenizer, AutoModelForTokenClassification
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu.graphs as htgraphs

os.environ['PT_HPU_ENABLE_GENERIC_STREAM'] = '1'
os.environ['PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES'] = '0'


hpu = torch.device('hpu')
cpu = torch.device('cpu')

activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU] 



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



hindi_sentences = [
    "मैं घर जा रहा हूँ।",
    "वह किताब पढ़ रही है।",
    "तुम स्कूल जाओ।",
    "हम खाना बना रहे हैं।",
    "वहाँ सूरज चमक रहा है।",
    "उसने खेल खेला।",
    "उसकी बहन गाना गा रही है।",
    "मेरे पास समय नहीं है।",
    "उसने अपनी माँ से बात की।",
    "उसे खुशी मिली।",
    'लगातार हमलावर हो रहे शिवपाल और राजभर को सपा की दो टूक, चिट्ठी जारी कर कहा- जहां जाना चाहें जा सकते हैं',
    "तुम्हारा दोस्त खेल रहा है।",
    "मेरा नाम अनीता है।",
    "उसने बहुत सोचा।",
    "हम वहाँ जा रहे हैं।",
    "उसके पास पैसे हैं।",
    "तुमने क्या देखा?",
    "उसका बच्चा रो रहा है।",
    "मेरे दोस्त स्कूल जा रहे हैं।",
    "उसकी माँ रसोई में है।"
]



with torch.profiler.profile(
    schedule = torch.profiler.schedule(wait=0, warmup=5, active = 2, repeat = 1),
    activities = activities, 
    on_trace_ready=torch.profiler.tensorboard_trace_handler('logs')
) as profiler:
    

    

    for i in range(20):
        predicted_labels = get_predictions(sentence=hindi_sentences[i], 
                                    tokenizer=tokenizer,
                                    model=model
                                    )
        htcore.mark_step()
        profiler.step()







# for index in range(len(sentence.split(' '))):
#   print( sentence.split(' ')[index] + '\t' + predicted_labels[index] )
  
# print("\n\n Time taken : ", time.time() - start_time)


# sentence = 'ಶರಣ್ ರ ನೀವು ನೋಡಲೇಬೇಕಾದ ಟಾಪ್ 5 ಕಾಮಿಡಿ ಚಲನಚಿತ್ರಗಳು'

# start_time = time.time()
# predicted_labels = get_predictions(sentence=sentence, 
#                                    tokenizer=tokenizer,
#                                    model=model
#                                    )

# for index in range(len(sentence.split(' '))):
#   print( sentence.split(' ')[index] + '\t' + predicted_labels[index] )

# print("\n\n Time taken : ", time.time() - start_time)
