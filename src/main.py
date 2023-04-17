import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.tokenize import sent_tokenize
import re
import nltk
nltk.download('punkt')

class Seq2SeqTranslator:
    def __init__(self, model_names, device='cpu',batch_size=64):
        self.tokenizers = {}
        self.models = {}
        self.batch_size = batch_size
        self.device = device
        for lang, model_name in model_names.items():
            self.tokenizers[lang] = AutoTokenizer.from_pretrained(model_name)
            self.models[lang] = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def _split_into_sentences(self, paragraph):
        for sent in re.findall(u'[^!?。\.\!\?]+[!?。\.\!\?]?', paragraph, flags=re.U):
            yield sent.strip().replace('\n', '')

    def translate(self, language, query):
        if language in self.tokenizers and language in self.models:
            tokenizer = self.tokenizers[language]
            model = self.models[language]
            translated_sentences=[]
            if language=='envi':
                prefix = 'en: '
                sentences = [prefix + sent for sent in sent_tokenize(query)]
                for i in range(0, len(sentences), self.batch_size):
                    batch_input = sentences[i:i+self.batch_size]
                    input_ids = tokenizer.batch_encode_plus(batch_input, return_tensors="pt", padding=True, truncation=True).to(self.device)
                    outputs = model.generate(input_ids['input_ids'])
                    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    translated_sentences.extend([s.strip('vi:') for s in decoded])
                output_document = "".join(translated_sentences)
            elif language=='zhvi':
                prefix = ''
                sentences = [prefix + sent for sent in list(self._split_into_sentences(query))]
                for i in range(0, len(sentences), self.batch_size):
                    batch_input = sentences[i:i+self.batch_size]
                    input_ids = tokenizer.batch_encode_plus(batch_input, return_tensors="pt", padding=True, truncation=True).to(self.device)
                    outputs = model.generate(input_ids['input_ids'])
                    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    translated_sentences.extend(decoded)
                output_document = " ".join(translated_sentences)
            return output_document
        else:
            raise ValueError(f'Model for language {language} not found.')