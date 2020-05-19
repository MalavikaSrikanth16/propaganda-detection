import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
nltk.download('punkt')
nltk.download('wordnet')
wordnet_lemmatizer = WordNetLemmatizer()

CSV_NAME = "causal_datasets\\climate_change_data.csv"

df = pd.read_csv(CSV_NAME, encoding='cp1252')
causal_list_df = pd.read_csv("causal_words.csv")
causal_list_df.fillna('', inplace=True)
causal_links = list(causal_list_df['Causal links'])
causal_verbs = list(causal_list_df['Causal verbs'])
causal_links = [l for l in causal_links if l != '']
causal_verbs = [v for v in causal_verbs if v != '']
causal_links_presence = []
causal_verbs_presence = []
found_causal_links_list = []
found_causal_verbs_list = []

for i, row in df.iterrows():
    sent = row['Sentences']
    sent = sent.lower()
    exclude = set(string.punctuation)
    sent = ''.join(ch for ch in sent if ch not in exclude)
    causal_link_presence = 0
    found_causal_links = []
    #Going through all causal links in list and checking for presence in sentence
    for link in causal_links:
        link = ' ' + link + ' '
        if link in sent:
            found_causal_links.append(link)
            causal_link_presence = 1
    causal_links_presence.append(causal_link_presence)
    found_causal_links_list.append(found_causal_links)
    #Going through the lemmas of the sentence to see if any of them are causal verbs
    tokens = word_tokenize(sent)
    causal_verb_presence = 0
    found_causal_verbs = []
    for tok in tokens:
        tok_lemma = wordnet_lemmatizer.lemmatize(tok, pos='v')
        if tok_lemma in causal_verbs:
            causal_verb_presence = 1
            found_causal_verbs.append(tok_lemma)
    causal_verbs_presence.append(causal_verb_presence)
    found_causal_verbs_list.append(found_causal_verbs)

df['Causal link presence'] = causal_links_presence
df['Causal verb presence'] = causal_verbs_presence
df['Found causal links'] = found_causal_links_list
df['Found causal verbs'] = found_causal_verbs_list

df.to_csv(CSV_NAME)