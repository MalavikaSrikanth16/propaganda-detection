import spacy
from spacy.lang.en import English
from spacy.tokens import Doc

from nltk.stem import SnowballStemmer

from altlex.utils.dependencyUtils import tripleToList

from altlex.semantics.semaforHandler import TCPClientSemaforHandler

class PlaintextIterator(object):
    nlp = spacy.load("en_core_web_sm")
    stemmer = SnowballStemmer('english')
    frame_parser = TCPClientSemaforHandler()

    def __init__(self, filename=None, data=None, frames=False):
        self._filename = filename
        self._data = data
        assert(data is not None or filename is not None)

        self.frames = frames
        self.nlp.add_pipe(self.set_custom_boundaries, before="parser")

    def addMetadata(self, sentence, metadata):
        return metadata

    def _iterSentences(self, data):
        for line in data:
            yield line
    
    def _iterData(self):
        if self._data is not None:
            for sentence in self._iterSentences(self._data):
                yield sentence
        else:
            with open(self._filename) as f:
                for sentence in self._iterSentences(f):
                    yield sentence

    def set_custom_boundaries(self, doc):
        doc[0].is_sent_start = True
        for i in range(1,len(doc)):
            doc[i].is_sent_start = False
        return doc
    
    def __iter__(self):
        for line in self._iterData():
            parsed_line = self.nlp(str(str(line).strip()))
            #print(parsed_line.is_parsed)

            offset = 0
            for sentence in parsed_line.sents:      #Might split a sentence into multiple sentences?
                metadata = dict(original=str(sentence),
                                deps=[],
                                lemmas=[],
                                pos=[],
                                parse=[],
                                words=[],
                                ner=[],
                                stems=[])

                for index,word in enumerate(sentence):
                    metadata['lemmas'].append(word.lemma_)
                    metadata['pos'].append(word.tag_)
                    metadata['ner'].append('O')
                    metadata['stems'].append(self.stemmer.stem(str(word)))
                    metadata['words'].append(str(word))

                    head = word.head.i-offset
                    if word.dep_ == 'ROOT':
                        head = -1
                    metadata['deps'].append((head, index, word.dep_))

                offset += len(sentence)

                metadata['dependencies'] = tripleToList(metadata['deps'], len(metadata['words']))

                if self.frames:
                    metadata['frames'] = self.frame_parser.get_frames(sentence)

                metadata = self.addMetadata(line, metadata)
                    
                yield metadata
        
class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab
        
    def __call__(self, text):
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

class TokenizedIterator(PlaintextIterator):
    def __init__(self, *args, **kwargs):
        super(TokenizedIterator, self).__init__(*args, **kwargs)
        
        self.nlp.tokenizer = WhitespaceTokenizer(self.nlp.vocab)    

    def addMetadata(self, sentence, metadata):
        metadata['label'] = sentence.label
        metadata['data_type'] = sentence.data_type
        metadata['prev_len'] = sentence.prev_len
        metadata['altlex_len'] = sentence.altlex_len
        metadata['curr_len'] = sentence.curr_len        
        
        return metadata
        
class LabeledSentence:
    def __init__(self, prev, altlex, curr, label, data_type=None):
        self.prev = prev.strip()
        self.altlex = altlex.strip()
        self.curr = curr.strip()
        self.label = label
        self.data_type = data_type

        self.sentence = ' '.join([self.prev, self.altlex, self.curr])
        self.prev_len = len(self.prev.split(' '))
        self.altlex_len = len(self.altlex.split(' '))
        self.curr_len = len(self.curr.split(' '))
        
    def __str__(self):
        return self.sentence
  
    def __unicode__(self):
        return self.sentence
    
    def __repr__(self):
        return self.sentence
    
class TrainSetIterator(TokenizedIterator):

    def _iterSentences(self, data):
        for line in data:
            label, eng_prev, eng_altlex, eng_curr, sim_prev, sim_altlex, sim_curr = line.split('\t')
            yield LabeledSentence(eng_prev, eng_altlex, eng_curr, label, data_type='english')
            yield LabeledSentence(sim_prev, sim_altlex, sim_curr, label, data_type='simple')
            
class TestSetIterator(TokenizedIterator):

    def _iterSentences(self, data):
        for line in data:
            prev, altlex, curr, label = line.split('\t')
            yield LabeledSentence(prev, altlex, curr, label)
