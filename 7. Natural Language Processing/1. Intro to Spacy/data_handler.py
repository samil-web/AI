
from _typeshed import OpenTextModeUpdating
from os import system
import numpy as np

def preprocess(pth):
    with open('data/Suruculuk.txt','r') as f:
        lines = f.readlines()

        for i in range(len(lines)):
            # print(lines[i])
            lines[i] = lines[i].replace(".","").replace(",","").replace(";","").replace("!","").replace("?","").replace("-","").replace("_","").replace("\"","").replace("\'","")

        with open('data/preprocessed.txt','w') as p:
            p.writelines(lines)
preprocess("data/processed.txt")

def get_sentences(pth):
    # with open('data/processed.txt','r') as f:
    #     f.readlines
    f = open(pth,'r')

    lines = f.readlines()
    sentences = [line.lower().split() for line in lines]

    f.close()
    print(len(sentences))
    return sentences

sentences = get_sentences('data/processed.txt')

def clean_sentences(sentences):
    
    i = 0
    while i < len(sentences):
        if sentences[i] == []:
            sentences.pop()
        else:
            i += 1
    print(len(sentences))
    return sentences
cleaned_sentences = clean_sentences(sentences)

def get_tokens(sentences):
    vocab = []
    for sentence in sentences:
        for token in sentence:
            if token not in vocab:
                vocab.append(token)
            
    w2i = {w:i for (i,w) in enumerate(vocab)}
    i2w = {i:w for (i,w) in enumerate(vocab)}
    return w2i,i2w,len(vocab)
w2i,i2w,len_vocab = get_tokens(cleaned_sentences)

print(w2i['maksimal'])
print(i2w[0])
print(len_vocab)

def get_pair(sentences,w2i,r):
    pairs = []

    for sentence in sentences:
        tokens = [w2i[word] for word in sentence]# list of index of words in txt file
        # print(tokens)
        for center in range(len(tokens)):#
            for context in range(-r,r+1):
                context_word = center + context 

                if context_word<0 or context_word >=(len(tokens)) or context_word == center:
                    continue
                else:
                    pairs.append((tokens[center],tokens[context_word]))

    return np.array(pairs) 
            
get_pair(sentences = cleaned_sentences,w2i=w2i,r = 2)

def get_dataset():
    sentences = get_sentences('data/processed.txt')
    clean_sents = clean_sentences(sentences)

    w2i,i2w,len_vocab = get_tokens(clean_sents)
    print(i2w[0],i2w[1])
    pairs = get_pair(clean_sents,w2i,4)

    return pairs,len_vocab

pairs,_ = get_dataset()
print(pairs[4])