from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
#instantiating model
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

#converts prompt into a sequence of numbers
#example
tokens = tokenizer.encode("I liked it", return_tensors="pt")
result = model(tokens)
int(torch.argmax(result.logits)) +1
r = requests.get("https://www.ratemyprofessors.com/ShowRatings.jsp?tid=2335479")
soup = BeautifulSoup(r.text, "html.parser")
regex = re.compile('."Comments."')
results = soup.find_all("div",{'class':regex})
reviews = [result.text for result in results]
import pandas as pd 
import numpy as np
df = pd.DataFrame(np.array(reviews),columns = ['review'])
def score(review):
    tokens = tokenizer.encode(review, return_tensors = 'pt')
    result = model(tokens)
    return int(torch.argmax(result.logits))+1
df['sentiment'] = df['review'].apply(lambda x: score(x[:512]))
