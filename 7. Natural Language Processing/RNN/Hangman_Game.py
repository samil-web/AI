import streamlit as st
import pandas as pd


header = st.container()

import torch
import string
import torch.nn as nn


all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1 # Plus EOS marker

all_categories = ['French', 'Czech', 'Dutch', 'Polish', 'Scottish', 'Chinese', 'English', 'Italian', 'Portuguese', 
            'Japanese', 'German', 'Russian', 'Korean', 'Arabic', 'Greek', 'Vietnamese', 'Spanish', 'Irish']
n_categories = len(all_categories)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)




rnn = torch.load('data/RNN.pth')

def categoryTensor(category):
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor
def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

# LongTensor of second letter to end (EOS) for target
def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return torch.LongTensor(letter_indexes)


max_length = 20

# Sample from a category and starting letter
def sample(category, start_letter='A'):
    with torch.no_grad():  # no need to track history in sampling
        category_tensor = categoryTensor(category)
        input = inputTensor(start_letter)
        hidden = rnn.initHidden()
        #hidden = 'a'

        output_name = start_letter

        for i in range(max_length):
            output, hidden = rnn(category_tensor, input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensor(letter)

        return output_name

# Get multiple samples from one category and multiple starting letters
def samples(category, start_letters='ABC'):
    for start_letter in start_letters:
        print(sample(category, start_letter))

#print(sample('Arabic','A'))

with header:
    st.title('Hangman Game')

import string
def split(word):
    return [char for char in word]
lang = st.selectbox('enter language of choice',options = ['Arabic','Chinese','Czech','Dutch','English','French','German','Greek','Irish','Italian','Japanese','Korean','Polish','Portuguese','Russian','Scottish','Spanish','Vietnamese'] )
st.text(lang)
char = st.select_slider('choose a letter',options = split(string.ascii_lowercase.upper()), value='A')


name = sample(lang,char)


#from helper import replacer
#from rnn_model import samples
def replacer(s, newstring, index, nofail=False):
    # raise an error if index is outside of the string
    if not nofail and index not in range(len(s)):
        raise ValueError("index outside given string")

    # if not erroring, but the index is still not in the correct range..
    if index < 0:  # add it to the beginning
        return newstring + s
    if index > len(s):  # add it to the end
        return s + newstring

    # insert the new string between "slices" of the original
    return s[:index] + newstring + s[index + 1:]
    
a = name
if len(a) > 1:
    a = replacer(a,'_', 1)
if len(a) > 4:
    a = replacer(a,'_', 4)
if len(a) > 6:
    a = replacer(a,'_', 6)

st.text(a)

count = 0
inp = ''
inp = st.text_input('guess the name','')
inp = inp.capitalize()


if inp == name:
    st.text('You must be omnipotent')
elif inp == '':
    st.text_input('try again','')
    if inp == name:
        st.text('You must be omnipotent')
    else:
        st.text('nope')








