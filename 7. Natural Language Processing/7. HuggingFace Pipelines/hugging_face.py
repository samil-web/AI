from os import truncate
from transformers import pipeline
from transformers import AutoTokenizer,AutoModelForSequenceClassification

import torch
import torch.nn.functional  as F

model_name = 'distilbert-base-uncased-finetuned-sst-2-english'

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis",model = model,tokenizer = tokenizer)

result = classifier("Build, train and deploy state of the art models powered by the reference open source in natural language processing.",)
print(result)

tokens = tokenizer.tokenize("Build, train and deploy state of the art models powered by the reference open source in natural language processing.")
token_ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = tokenizer('Build, train and deploy state of the art models powered by the reference open source in natural language processing.')
print(f'   Tokens:{tokens}')
print(f'Token IDs:{token_ids}')
print(f'Input IDs:{input_ids}')

X_train = "Build, train and deploy state of the art models powered by the reference open source in natural language processing."

batch = tokenizer(X_train,padding = True,truncation = True,max_length = 512,return_tensors = 'pt')
print(batch)
# batch is a dict and it contains input_ids as key and tensors as value
with torch.no_grad():
    outputs = model(**batch,labels = torch.tensor([1]))#inside of batch we have token ids
    print(outputs)
    predictions = F.softmax(outputs.logits,dim= 1)
    print(predictions)
    labels = torch.argmax(predictions,dim=1)
    print(labels)
    labels = [model.config.id2label[label_id] for label_id in labels.tolist()]
    print(labels)

save_directory = "saved"
tokenizer.save_pretrained(save_directory)
model.save_pretrainned(save_directory)

tokenizer = AutoTokenizer.from_pretrained(save_directory)
model = AutoModelForSequenceClassification.from_pretrained(save_directory)
