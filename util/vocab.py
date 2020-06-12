from util.common import download
from util.tokenizer import tokenizer

import sentencepiece as spm
import gluonnlp as nlp


# koBERT vocab download
def koBertVocab():
  cachedir='~/reforBert/'

  vocab_info = tokenizer
  vocab_file = download(vocab_info['url'],
                         vocab_info['fname'],
                         vocab_info['chksum'],
                         cachedir=cachedir)
  vocab_b_obj = nlp.vocab.BERTVocab.from_sentencepiece(vocab_file,
                                                       padding_token='[PAD]')
  return vocab_b_obj


# paul-hyun vocab loader
def load_vocab(file):
  vocab = spm.SentencePieceProcessor()
  vocab.load(file)
  return vocab
