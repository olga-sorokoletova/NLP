#this file contains a model corresponding to the first approach, since this model allows to gain the best performance on the development dataset
#train function and other part of a pipeline not needed for the evaluation procedure can be found in the corresponding notebook baseline.ipynb

import numpy as np
from typing import List, Tuple, Dict, Optional

from model import Model

from tqdm.notebook import tqdm

import torch
from torch import nn

import os

import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

#fix embedding dimension as global variable
n_embedding = 50

#fix path to the model folder
model_folder = './model/'
embedding_path = os.path.join(model_folder, 'glove.6B.50d.txt')
weights_path = os.path.join(model_folder, 'best_weights.pt')

#to store hyperparameters outside the model
class HParams():
  # mean emmbedding for the sentence 1, embedding for the separator, mean embedding for the sentence 2
  n_features = n_embedding*3
  n_hidden = 128
  # do not used in the final model. but used during experiments
  dropout = 0.0  
hparams = HParams()


def build_model(device: str) -> Model:
    # to inherit nn.Module before wrapping to Model
    model = WiCClassifier(hparams, device)
    model.to(device)
    
    student_model = StudentModel(model, device)
    return student_model

class WiCClassifier(nn.Module):

    def __init__(self,
                 hparams, device):
        super().__init__()

        self.device = device

        # classification layers (yes, 3 goes before 2:) I am sorry, can't already change, because then they will not map to the corresponding weights)
        self.lin1 = torch.nn.Linear(hparams.n_features, hparams.n_hidden)
        self.lin3 = torch.nn.Linear(hparams.n_hidden, hparams.n_hidden)
        self.lin2 = torch.nn.Linear(hparams.n_hidden, 1)
        
        # dropout layer for the experiments, but it is not used in the model with best weights
        self.dropout = torch.nn.Dropout(hparams.dropout)

        # Binary Cross-Entropy
        self.loss_fn = torch.nn.BCELoss()
        
        # initialize counter
        self.global_epoch = 0


    def forward(self,
                x: torch.Tensor, y: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:

        # construct layers
        out = self.lin1(x)
        out = torch.relu(out)
        out = self.lin3(out)
        out = torch.relu(out)
        out = self.lin2(out).squeeze(1)

        # smooth output to [0, 1]
        pred = torch.sigmoid(out)

        # output dictionary
        result = {'logits': out, 'pred': pred}

        if y is not None:
            loss = self.loss(pred, y)
            # add loss to output dictionary
            result['loss'] = loss

        return result

    def loss(self, pred, y):
        return self.loss_fn(pred, y)


class RandomBaseline(Model):

    options = [
        ('True', 40000),
        ('False', 40000),
    ]

    def __init__(self):

        self._options = [option[0] for option in self.options]
        self._weights = np.array([option[1] for option in self.options])
        self._weights = self._weights / self._weights.sum()

    def predict(self, sentence_pairs: List[Dict]) -> List[str]:
        return [str(np.random.choice(self._options, 1, p=self._weights)[0]) for x in sentence_pairs]


class StudentModel(Model):

    def __init__(self, model, device):

        self.device = device

        # load vocabulary
        self.vocab = self.word_vectors(embedding_path)

        self.model = model

        # load weights
        self.model.load_state_dict(torch.load(weights_path, map_location=torch.device(self.device)))


    def word_vectors(self, embedding_path: str) -> Dict:

        word_vectors = dict()

        with open(embedding_path) as f:

            for i, line in tqdm(enumerate(f)):

                word, *vector = line.strip().split(' ')

                vector = torch.tensor([float(c) for c in vector])
            
                word_vectors[word] = vector
            
        return word_vectors



    def sentence2vector(self, sentence: str, start_idx: str, end_idx: str) -> Optional[torch.Tensor]:

        # extract target word to delete it from stop words set (and to do some other experiments, but there obly for stop words set)
        target_word = sentence[int(start_idx): int(end_idx)]
        # hypthen handling, example of issue: sample 322 in train dataset, sentence 2
        cleaned_sentence = re.sub('-', ' ', sentence)
        # slash handling, example of issue: sample 1987 in train dataset, sentence 2
        cleaned_sentence = re.sub('/', ' ', cleaned_sentence)
        # dash handling, example of issue: sample 7026 in train dataset, sentence 2
        cleaned_sentence = re.sub('—', ' ', cleaned_sentence)
        # numbers removal
        unnumbered_sentence = re.sub(r'\d+', '', cleaned_sentence)
        # punctuation removal
        cleaned_sentence = unnumbered_sentence.translate(str.maketrans('', '', string.punctuation))
        # lowercasing
        lowered_sentence = cleaned_sentence.lower()
        # tokenizing
        word_tokens = word_tokenize(lowered_sentence)
        # stop words removal
        stop_words = set(stopwords.words('english'))
        # exception for the target words containing in the stop words set, e.g. only
        if target_word in stop_words:
          stop_words.remove(target_word)
        filtered_sentence = [word for word in word_tokens if not word in stop_words]
        # get embeddings for each word if it exists in vocabulary
        sentence_word_vector = [self.vocab[w] for w in filtered_sentence if w in self.vocab]
        if len(sentence_word_vector) == 0:
            return None
        sentence_word_vector = torch.stack(sentence_word_vector)
        # mean of embeddings for the whole sentence
        return torch.mean(sentence_word_vector, dim=0)

    def concatenate(self,
                    sentence1: str, start1: str, end1: str, 
                    sentence2: str, start2: str, end2: str) -> Optional[torch.Tensor]:
        # mean embedding for the sentence 1
        mean_embedding1 = self.sentence2vector(sentence1, start1, end1)
        # mean embedding for the sentence 2
        mean_embedding2 = self.sentence2vector(sentence2, start2, end2)
        # add separator and concatenate
        return torch.cat([mean_embedding1, self.vocab['.'], mean_embedding2])

    def predict(self, sentence_pairs: List[Dict]) -> List[str]:

        predictions = []

        for d in sentence_pairs:

          # extract relevant fields from the input dictionaries
          sentence1 = d['sentence1']
          start1 = d['start1']
          end1 = d['end1']

          sentence2 = d['sentence2']
          start2 = d['start2']
          end2 = d['end2']

          # apply preprocessing
          sentences_vector = self.concatenate(sentence1, start1, end1, sentence2, start2, end2).to(self.model.device)

          # feed to the model
          out = self.model(sentences_vector.unsqueeze(0))
          # get prediction in the correct format
          pred = str(torch.round(out['pred'].squeeze(0)).item()==1.)
          # collect predictions
          predictions.append(pred)

        return predictions


