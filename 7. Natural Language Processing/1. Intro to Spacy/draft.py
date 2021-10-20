import spacy 
import re
from spacy import displacy
nlp = spacy.load('en_core_web_lg')

print(nlp.pipeline)
# Process sentences 'Hello, world. Antonio is learning Python.' using spaCy
doc = nlp(u'Hello, world. Antonio is learning Python')

for token in doc:
    print(token.text)

# The spaCy model automatically divide the text in sentences:
for sent in doc.sents:
    print(sent)

# We know that "im" is a wrong way to write "I'm" and,
#  since there are two verbs in that text,
#  there are also two sentences. Let's see how spaCy performs:

doc = nlp('Im antonio im learning python')
for sentence in doc.sents:
    print(sentence)

# Looking at the output of `nlp.pipeline` above, we can see there are 
#  a tagger,a dependency parser and the entity recognizer. 
# Lets check the entities of the following sentence:
tokens = nlp("Let's go to L.A.!")
for token in tokens:
    print(token.text)

from spacy.tokenizer import Tokenizer
tokenizer = Tokenizer(vocab = nlp.vocab)
tokens = tokenizer("Let's go to L.A.!")
for token in tokens:
    print(token)

doc = nlp('Apple is a $1000b company.')
for ent in doc.ents:
    print(ent,ent.label_)

# Removing stop words
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
print("Number of stop words: %d" % len(spacy_stopwords))
print("First ten stop words: %s" % list(spacy_stopwords)[:10])

text = """He determined to drop his litigation with the monastry, and relinguish his claims to the wood-cuting and 
fishery rihgts at once. He was the more ready to do this becuase the rights had become much less valuable, and he had 
indeed the vaguest idea where the wood and river in question were."""

doc = nlp(text)

tokens = [token.text for token in doc if not token.is_stop]
for token in tokens:
    print(token)
# For adding customized stop words:
customize_stop_words = ['computing','filtered']
for w in customize_stop_words:
    nlp.vocab[w].is_stop = True

### Stemming and Lemmatization
nlp = spacy.load('en_core_web_lg')
# nlp.add_pipe(nlp.create_pipe('merge_entities'))
doc = nlp(
    u"""He determined to drop his litigation with the monastry, and relinguish his claims to the wood-cuting and 
fishery rihgts at once. He was the more ready to do this becuase the rights had become much less valuable, and he had 
indeed the vaguest idea where the wood and river in question were."""
)
lemma_words = []
for token in doc:
    if token.is_stop:
        continue
    lemma_words.append(token.lemma_)
print(lemma_words)

text = """He determined to drop his litigation with the monastry, and relinguish his claims to the wood-cuting and 
fishery rihgts at once. He was the more ready to do this becuase the rights had become much less valuable, and he had 
indeed the vaguest idea where the wood and river in question were."""
import string
text_no_punct = "".join([char for char in text if char not in string.punctuation])
text_no_punct

st = "hhheeeLLLLooo hoooowww areee youuu?????"
text = re.sub(r"(.)\1+", r"\1", st)
text

sentence = nlp('Antonio is learning Python in Strive School.')
for token in sentence:
    print(token.pos_)

for token in sentence:
    print(token.tag_)
    
# While the output of the `.pos_` attribute is easy to decrypt (`PROPN`: proper noun,
# `AUX`: Auxiliary verb,
# `VERB`: verb,
# `ADP`: Adposition,
# `PUNCT`: Punctuation), the `.tag_`'s output is more cryptic. For this, you can use the `spacy.explain()` function to get the intuition behind that:

for token in sentence:
    print(spacy.explain(token.tag_))

for token in sentence:
    print(f'{token.text:{12}} {token.pos_:{10}} {token.tag_:{8}} {spacy.explain(token.tag_)}')

sentence = nlp('Antonio is learning Python Programming Language')
num_pos  = sentence.count_by(spacy.attrs.POS)
num_pos

sentence.vocab[96].text

for ID,frequency in num_pos.items():
    print(f'{ID} stands for {sentence.vocab[ID].text:{8}}:{frequency}')

doc = nlp('Antonio is learning Python Programming Language at Strive School')
displacy.render(doc,style = 'ent')

doc = nlp('Baku is the capital of Azerbaijan')
displacy.render(doc,style = 'ent')

sentence = nlp(u'Manchester United is looking to sign Harry Kane for $90 million. David demand 100 Million Dollars')
displacy.render(sentence, style='ent', jupyter=True)