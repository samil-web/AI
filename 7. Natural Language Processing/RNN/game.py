from unicodedata import category
import torch
import torch.nn as nn
import matplotlib.pyplot as plt 
from rnn_model import samples
from model import RNN
import string

# from utils import ALL_LETTERS,N_LETTERS
# from utils import load_data,letter_to_tensor
# from utils import line_to_tensor,random_training_example
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

# model = torch.load('data/RNN.pth')

# samples('German','D')