# STEP 0 - PRE REQUISITES

# python -m spacy download en_core_web_lg

# TBD: Import libraries
import os
import spacy
import random
from spacy.util import minibatch, compounding
from pathlib import Path
from spacy.training import Example
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


nlp = spacy.load('en_core_web_lg')
# TBD: Load preferred model
print(nlp)
with open("food.txt") as file:
    dataset = file.read()

# TBD: Load the dataset and test it as-is
doc = nlp(dataset)
print('Entities:',[(ent.text,ent.label_) for ent in doc.ents])

# STEP 1 - TRAIN DATA

# Prepare training data

# TBD: define all the entities by extracting the words and their indexes from the dataset
# expected format is the following:  ("sentence", {"entities": [0,10, "FOOD"]})

words = ['ketchup','pasta','carrot','pizza',
        'garlic','tomato sauce','basil','carbonara',
        'eggs','cheek fat','pancakes','parmigiana','eggplant',
        'fettucine','heavy cream','polenta','risotto','espresso', 
        'arrosticini','spaghetti','fiorentina steak','pecorino',
        'maccherone','nutella','amaro','pistachio','coca-cola', 
        'wine','pastiera','watermelon','cappucino','ice cream',
        'soup','Lemon','chocolate','pineapple']

train_data = []

with open('food.txt') as file:
    dataset = file.readlines()
    for sentence in dataset:
        print('######')
        print('sentence: ',sentence)
        print('######')
        sentence = sentence.lower()
        entities = []
        for word in words:
            word = word.lower()
            if word in sentence:
                start_index = sentence.index(word)
                end_index = len(word) + start_index
                print('word: ',word)
                print('start index:',start_index)
                print('end_index:',end_index)
                pos = (start_index,end_index,'FOOD')
                entities.append(pos)
        element = (sentence.rstrip('\n'),{'entities':entities})

        train_data.append(element)
        print('---------')
        print('element:',element)

        #('this is my sentence:',{'entities':[0,4,'PREP']})
        # ('this is my sentence:',{'entities':[6,8,'VERB']})
# STEP 2 - UPDATE MODEL

ner = nlp.get_pipe('ner')

for _,annotations in train_data:
    for ent in annotations.get('entities'):
        ner.add_label(ent[2])

# TBD: load the needed pipeline

# TBD: define the annotations

# TBD: train the model


pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

# TBD: define the number of iterations, the batch size and the drop according to your experience or using an empirical value
# Train model
with nlp.disable_pipes(*unaffected_pipes):
    for iteration in range(0):
        print("Iteration #" + str(iteration))

        # Data shuffle for each iteration
        random.shuffle(train_data)
        losses = {}
        batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
        for batch in spacy.util.minibatch(train_data, size=3):
            for text, annotations in batch:
                # Create an Example object
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                # Update the model
                nlp.update([example], losses=losses, drop=0)
        print("Losses:", losses)

# Save the model
output_dir = Path('/ner/')
nlp.to_disk(output_dir)
print('Saved correctly')
# TBD:

# STEP 3 - TEST THE UPDATED MODEL
print('Loading model....')
# Load updated model
nlp_updated = spacy.load(output_dir)

doc =nlp_updated("I don't like pizza with pineapple.")
# TBD: test with a old sentence
print("entities:", [(ent.text, ent.label_) for ent in doc.ents])

# TBD: test with a new sentence and an old brand
doc =nlp_updated("in carbonara, parmigiano is not used.")
# TBD: test with a new sentence and a new brand
doc =nlp_updated("Fabio likes full-stack development")
print("entities:", [(ent.text, ent.label_) for ent in doc.ents])