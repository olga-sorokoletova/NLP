import numpy as np
from typing import List, Tuple, Dict

from model import Model
import random

import torch
import nltk
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer

nltk.download('punkt')

def build_model_b(device: str) -> Model:
    return StudentModel(device)

def build_model_ab(device: str) -> Model:
    return StudentModel(device, 'ab')

def build_model_cd(device: str) -> Model:
    return StudentModel(device, 'cd')

class RandomBaseline(Model):

    options_sent = [
        ('positive', 793+1794),
        ('negative', 701+638),
        ('neutral',  365+507),
        ('conflict', 39+72),
    ]

    options = [
        (0, 452),
        (1, 1597),
        (2, 821),
        (3, 524),
    ]

    options_cat_n = [
        (1, 2027),
        (2, 402),
        (3, 65),
        (4, 6),
    ]

    options_sent_cat = [
        ('positive', 1801),
        ('negative', 672),
        ('neutral',  411),
        ('conflict', 164),
    ]

    options_cat = [
        ("anecdotes/miscellaneous", 939),
        ("price", 268),
        ("food", 1008),
        ("ambience", 355),
        ("service", 248),
    ]

    def __init__(self, mode = 'b'):

        self._options_sent = [option[0] for option in self.options_sent]
        self._weights_sent = np.array([option[1] for option in self.options_sent])
        self._weights_sent = self._weights_sent / self._weights_sent.sum()

        if mode == 'ab':
            self._options = [option[0] for option in self.options]
            self._weights = np.array([option[1] for option in self.options])
            self._weights = self._weights / self._weights.sum()
        elif mode == 'cd':
            self._options_cat_n = [option[0] for option in self.options_cat_n]
            self._weights_cat_n = np.array([option[1] for option in self.options_cat_n])
            self._weights_cat_n = self._weights_cat_n / self._weights_cat_n.sum()

            self._options_sent_cat = [option[0] for option in self.options_sent_cat]
            self._weights_sent_cat = np.array([option[1] for option in self.options_sent_cat])
            self._weights_sent_cat = self._weights_sent_cat / self._weights_sent_cat.sum()

            self._options_cat = [option[0] for option in self.options_cat]
            self._weights_cat = np.array([option[1] for option in self.options_cat])
            self._weights_cat = self._weights_cat / self._weights_cat.sum()

        self.mode = mode

    def predict(self, samples: List[Dict]) -> List[Dict]:
        preds = []
        for sample in samples:
            pred_sample = {}
            words = None
            if self.mode == 'ab':
                n_preds = np.random.choice(self._options, 1, p=self._weights)[0]
                if n_preds > 0 and len(sample["text"].split(" ")) > n_preds:
                    words = random.sample(sample["text"].split(" "), n_preds)
                elif n_preds > 0:
                    words = sample["text"].split(" ")
            elif self.mode == 'b':
                if len(sample["targets"]) > 0:
                    words = [word[1] for word in sample["targets"]]
            if words:
                pred_sample["targets"] = [(word, str(np.random.choice(self._options_sent, 1, p=self._weights_sent)[0])) for word in words]
            else: 
                pred_sample["targets"] = []
            if self.mode == 'cd':
                n_preds = np.random.choice(self._options_cat_n, 1, p=self._weights_cat_n)[0]
                pred_sample["categories"] = []
                for i in range(n_preds):
                    category = str(np.random.choice(self._options_cat, 1, p=self._weights_cat)[0]) 
                    sentiment = str(np.random.choice(self._options_sent_cat, 1, p=self._weights_sent_cat)[0]) 
                    pred_sample["categories"].append((category, sentiment))
            preds.append(pred_sample)
        return preds


class StudentModel(Model):

''' In the corresponding notebooks separate prediction functions and commentaries for each mode can be found. '''
    
    def __init__(self, device, mode = 'b'):

          self.mode = mode
          self.tokenizer_a = AutoTokenizer.from_pretrained("distilbert-base-uncased")
          self.tokenizer_b = AutoTokenizer.from_pretrained("bert-large-uncased")
          self.tokenizer_cd = AutoTokenizer.from_pretrained("bert-large-uncased")
          self.device = device

          if mode == 'b':
              self.model_b = torch.load(f"./model/absa_b.pt", map_location = torch.device(device))
          if mode == 'ab':
              self.model_a = torch.load(f"./model/absa_a.pt", map_location = torch.device(device))
              self.model_b = torch.load(f"./model/absa_b.pt", map_location = torch.device(device))
          if mode == "cd":   
              self.model_a = torch.load(f"./model/absa_a.pt", map_location = torch.device(device))
              self.model_b = torch.load(f"./model/absa_b.pt", map_location = torch.device(device))
              self.model_cd = torch.load(f"./model/absa_cd.pt", map_location = torch.device(device))

        
    def extract(self, samples: List[Dict]) -> List[List]:

        predictions = []
        test_predictions = []

        self.model_a.eval()

        for sample in samples:
        
            sentence = sample["text"]
            aspects = word_tokenize(sentence)
            num_aspects = len(aspects)

            tokenized = self.tokenizer_a(num_aspects * [sentence], aspects, padding = True, truncation = True, max_length = 512, return_tensors = "pt")

            with torch.no_grad():

                input_ids = tokenized["input_ids"].to(self.device)
                attention_mask = tokenized["attention_mask"].to(self.device)

                output = self.model_a(input_ids = input_ids, attention_mask = attention_mask)

                logits = output["logits"]

                pred = torch.sigmoid(logits).argmax(-1)
                y_pred = pred.cpu()

                predicted_list = list()
                current_str = str()

                for i, aspect in enumerate(aspects):
                    if y_pred[i] == 1 and current_str != '':
                        current_str = current_str + ' ' + aspect
                    else:
                      if y_pred[i] == 1:
                        current_str = aspect
                      else:
                        if current_str != '':
                          predicted_list.append(current_str)
                          current_str = ''

                if current_str != '':
                    predicted_list.append(current_str)


                predictions.append(predicted_list)
        
        
        return predictions
    
    def predict(self, samples: List[Dict]) -> List[Dict]:


        predictions = []

        if self.mode == 'b':

            sentiment_types = ["positive", "negative", "neutral", "conflict"]
            self.model_b.eval()

            for sample in samples:

                predicted_dict = dict()
        
                sentence = sample["text"]

                aspects = [target[1] for target in sample['targets']]
                num_aspects = len(aspects)

                if aspects == []:
                  predicted_dict["targets"] = []
                  predictions.append(predicted_dict)
                  continue

                tokenized = self.tokenizer_b(num_aspects * [sentence], aspects, padding = True, truncation = True, max_length = 512, return_tensors = "pt")

                with torch.no_grad():

                    input_ids = tokenized["input_ids"].to(self.device)
                    attention_mask = tokenized["attention_mask"].to(self.device)

                    output = self.model_b(input_ids = input_ids, attention_mask = attention_mask)

                    logits = output["logits"]

                    pred = torch.softmax(logits, -1).argmax(-1)
                    y_pred = pred.cpu()

                    predicted_list = [(aspect, sentiment_types[pred]) for (aspect, pred) in zip(aspects, y_pred)]
                    predicted_dict["targets"] = predicted_list

                    predictions.append(predicted_dict)

        if self.mode == 'ab':

            sentiment_types = ["positive", "negative", "neutral", "conflict"]
            extractions = self.extract(samples)
            self.model_b.eval()

            for sample, aspects in zip(samples, extractions):

                predicted_dict = dict()
        
                sentence = sample["text"]

                num_aspects = len(aspects)

                if aspects == []:
                  predicted_dict["targets"] = []
                  predictions.append(predicted_dict)
                  continue

                tokenized = self.tokenizer_b(num_aspects * [sentence], aspects, padding = True, truncation = True, max_length = 512, return_tensors = "pt")

                with torch.no_grad():

                    input_ids = tokenized["input_ids"].to(self.device)
                    attention_mask = tokenized["attention_mask"].to(self.device)

                    output = self.model_b(input_ids = input_ids, attention_mask = attention_mask)

                    logits = output["logits"]

                    pred = torch.softmax(logits, -1).argmax(-1)
                    y_pred = pred.cpu()

                    predicted_list = [(aspect, sentiment_types[pred]) for (aspect, pred) in zip(aspects, y_pred)]
                    predicted_dict["targets"] = predicted_list


                    predictions.append(predicted_dict)

        if self.mode == "cd":

            sentiment_types_cd = ["none", "positive", "negative", "neutral", "conflict"]
            categories = ["anecdotes/miscellaneous", "price", "food", "ambience", "service"]
            num_categories = 5
            sentiment_types_ab = ["positive", "negative", "neutral", "conflict"]
            extractions = self.extract(samples)
            self.model_b.eval()
            self.model_cd.eval()

            for sample, aspects in zip(samples, extractions):

                predicted_dict = dict()
        
                sentence = sample["text"]

                num_aspects = len(aspects)

                if aspects == []:
                  predicted_dict["targets"] = aspects

                else:
                    tokenized = self.tokenizer_b(num_aspects * [sentence], aspects, padding = True, truncation = True, max_length = 512, return_tensors = "pt")

                    with torch.no_grad():

                        input_ids = tokenized["input_ids"].to(self.device)
                        attention_mask = tokenized["attention_mask"].to(self.device)

                        output = self.model_b(input_ids = input_ids, attention_mask = attention_mask)

                        logits = output["logits"]

                        pred = torch.softmax(logits, -1).argmax(-1)
                        y_pred = pred.cpu()

                        predicted_list = [(aspect, sentiment_types_ab[pred]) for (aspect, pred) in zip(aspects, y_pred)]
                        predicted_dict["targets"] = predicted_list


                tokenized_cd = self.tokenizer_cd(num_categories * [sentence], categories, padding = True, truncation = True, max_length = 512, return_tensors = "pt")

                with torch.no_grad():

                    input_ids = tokenized_cd["input_ids"].to(self.device)
                    attention_mask = tokenized_cd["attention_mask"].to(self.device)

                    output = self.model_cd(input_ids = input_ids, attention_mask = attention_mask)

                    logits = output["logits"]

                    pred = torch.softmax(logits, -1).argmax(-1)
                    y_pred = pred.cpu()

                    predicted_list = list()
                    predicted_list = [(category, sentiment_types_cd[pred]) for (category, pred) in zip(categories, y_pred) if pred > 0]
                    predicted_dict["categories"] = predicted_list


                predictions.append(predicted_dict)
            
        return predictions
