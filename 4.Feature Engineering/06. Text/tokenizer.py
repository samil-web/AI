
def get_sentences(pth):
    
    f = open(pth, 'r')
    lines = f.readlines()
    sentences = [ line.split() for line in lines]
    
    return sentences

def clean(sentences):
    
    i = 0
    while i < len(sentences):
        
        if sentences[i] == []:
            sentences.pop(i)
        else:
            i += 1
    return sentences


def get_ditcs(sentences):
    
    vocab = []
    
    for sentence in sentences:
        for token in sentence:
            if token not in vocab:
                vocab.append(token)
                
    w2i = { w: i for (i, w) in enumerate(vocab) }
    i2w = { i: w for (i, w) in enumerate(vocab) }
    
    return w2i, i2w, len(vocab)

sent = get_sentences("blakepoems.txt")
sent = clean(sent)

w2i, i2w, l = get_ditcs(sent)


print(i2w[0])
print(w2i['You'])