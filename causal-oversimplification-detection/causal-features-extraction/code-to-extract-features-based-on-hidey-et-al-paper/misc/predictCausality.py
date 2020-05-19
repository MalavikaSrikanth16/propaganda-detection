#read data (may be in crowdflower format)
#tokenize words and sentences, parse, POS-tag, etc
#split sentences on altlex using known causal altlexes
#modify data points as needed (splitDependencies)
#add features
#make interaction features
#make predictions using default, bootstrapped model

import sys
import pandas as pd
sys.path.append("C:\\Users\\malav\\OneDrive\\Documents\\Malavika\\Propaganda detection")
from altlex.utils.readers.plaintextIterator import PlaintextIterator

from altlex.featureExtraction.altlexHandler import AltlexHandler
from altlex.semantics.frameNetManager import FrameNetManager


if __name__ == '__main__':
    data_filename = "C:\\Users\\malav\\OneDrive\\Documents\\Malavika\\Propaganda detection\\causal_datasets\\climate_change_data.csv" #sys.argv[1]

    #a sentence iterator returns parsed sentences
    df = pd.read_csv(data_filename)
    sentences_raw = list(df['Sentences'].values)
    print(len(sentences_raw))
    sentenceIterator = PlaintextIterator(data=sentences_raw)
    altlexHandler = AltlexHandler()

    sentences = list(sentenceIterator)
    labels_sent_level = altlexHandler.findAltlexes(sentences)

    # output = [' '.join(i['words']) for i in sentences]
    # for sent, label in zip(output, labels_sent_level):
    #     print(sent, " : ", label)
    df['Pre-trained causal classifier labels'] = labels_sent_level

    df.to_csv("C:\\Users\\malav\\OneDrive\\Documents\\Malavika\\Propaganda detection\\causal_datasets\\climate_change_data.csv")
    print("Done")

    #Calculating framnet causal and anticausal scores on entire sentence for every sentence.
    # fn = FrameNetManager(verbose=True)
    # causal_scores_list = []
    # anticausal_scores_list = []
    # for index, metadata in enumerate(sentences):
    #     causal_score, anticausal_score = fn.score_modified(metadata['stems'], metadata['pos'])
    #     causal_scores_list.append(causal_score)
    #     anticausal_scores_list.append(anticausal_score)
    #
    # print(len(causal_scores_list))
    #
    # df['Framenet causal score'] = causal_scores_list
    # df['Framenet anticausal score'] = anticausal_scores_list
    #
    # df.to_csv("C:\\Users\\malav\\OneDrive\\Documents\\Malavika\\Propaganda detection\\causal_datasets\\climate_change_data.csv")
