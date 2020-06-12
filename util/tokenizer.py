from util.common import download

tokenizer = {
    'url':
    'https://kobert.blob.core.windows.net/models/kobert/tokenizer/kobert_news_wiki_ko_cased-ae5711deb3.spiece',
    'fname': 'kobert_news_wiki_ko_cased-1087f8699e.spiece',
    'chksum': 'ae5711deb3'
}

def koBERTTokenizer(cachedir='./cache/'):
  """Get KoBERT Tokenizer file path after downloading
  """
  model_info = tokenizer
  return download(model_info['url'],
                  model_info['fname'],
                  model_info['chksum'],
                  cachedir=cachedir)