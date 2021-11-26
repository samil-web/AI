import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
# from BS4 import BeautifulSoup
URL = 'https://m.facebook.com/story.php?story_fbid=4465924213474547&id=639288689471471'
page = requests.get(URL)

soup = BeautifulSoup(page.content,'html.parser')

