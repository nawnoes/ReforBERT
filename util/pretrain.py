import torch
import torch.nn.functional as F
from transformers import BertTokenizer, PreTrainedTokenizer
from ReforBERT.util.tokenizer import koBERTTokenizer
from gluonnlp.data import SentencepieceTokenizer, SentencepieceDetokenizer
from gluonnlp.data import BERTSPTokenizer
from ReforBERT.util.vocab import koBertVocab


# Reformer pytorch 오리지널 mask token 사용.
def orgin_mask_tokens(tokenizer, inputs: torch.Tensor, mlm_probability=0.15, pad=True):
  """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
  """ 
  Masked Language Model을 위한 마스킹데이터 생성

  마스킹된 입력 input과
  마스킹의 정답인 label을 반환
  """
  # 라벨 생성
  labels = inputs.clone()

  # mlm_probability defaults to 0.15 in Bert
  probability_matrix = torch.full(labels.shape, mlm_probability)

  # sentencepiece 토크나이저에서
  special_tokens_mask = [
    tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
  ]
  print(special_tokens_mask)

  probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
  if tokenizer._pad_token is not None:
    padding_mask = labels.eq(tokenizer.pad_token_id)
    probability_matrix.masked_fill_(padding_mask, value=0.0)
  masked_indices = torch.bernoulli(probability_matrix).bool()
  labels[~masked_indices] = -100  # We only compute loss on masked tokens

  # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
  indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
  inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

  # 10% of the time, we replace masked input tokens with random word
  indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
  random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
  inputs[indices_random] = random_words[indices_random]

  if pad:
    input_pads = tokenizer.max_len - inputs.shape[-1] # 인풋의 패딩 갯수 계산
    label_pads = tokenizer.max_len - labels.shape[-1] # 라벨의 패딩 갯수 계산

    inputs = F.pad(inputs, pad=(0, input_pads), value=tokenizer.pad_token_id)
    labels = F.pad(labels, pad=(0, label_pads), value=tokenizer.pad_token_id)

  # The rest of the time (10% of the time) we keep the masked input tokens unchanged
  return inputs, labels

def mask_special_tokens(tokenizer, str):
  """
  tokenizer내 special 토큰들에 대해 input을 마스킹
  tokenizer는 BERTSPTokenizer 사용.

  input은 torch tensor
  """
  vocab = tokenizer.vocab # 토크나이저의 사
  special_tokens = vocab.reserved_tokens # 스페셜 토큰 목록
  special_token_indices = vocab.to_indices(special_tokens) #스페셜 토큰을 index로 변환

  tokens =str.squeeze(0).tolist()
  result = [[int(val in special_token_indices) for val in tokens]]

  return result

# 토큰 인덱스의 양끝에 [CLS] token_indices [SEP] 토큰 추
def add_special_tokens(tokenizer, token_indices):
  vocab = tokenizer.vocab  # 토크나이저의 사
  cls_token = vocab.cls_token  #
  sep_token = vocab.sep_token  #

  cls_index = vocab.to_indices(cls_token)
  sep_index = vocab.to_indices(sep_token)

  result = [cls_index] + token_indices + [sep_index]

  return result


# kobert에서 사용하는 tokenizer로 mask_token 구현
def kobert_mask_tokens(tokenizer, inputs: torch.Tensor, max_len = 512, mlm_probability=0.15, pad=True):
  """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
  """ 
  Masked Language Model을 위한 마스킹데이터 생성

  마스킹된 입력 input과
  마스킹의 정답인 label을 반환
  """
  # 라벨 생성
  labels = inputs.clone()
  # 사전
  vocab = tokenizer.vocab
  padding_token_id = vocab.to_indices(vocab.padding_token)
  unknown_token_id = vocab.to_indices(vocab.unknown_token)
  # mlm_probability defaults to 0.15 in Bert
  probability_matrix = torch.full(labels.shape, mlm_probability)

  # sentencepiece 토크나이저에서
  special_tokens_mask = mask_special_tokens(tokenizer,inputs)
  print(special_tokens_mask)

  probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
  if vocab.padding_token is not None:
    padding_mask = labels.eq(padding_token_id)
    probability_matrix.masked_fill_(padding_mask, value=0.0)
  # 마스크할 부분을 베르누이 함수를 통해 True로 설
  masked_indices = torch.bernoulli(probability_matrix).bool()

  #mask가 되어있지 않은 곳에 [UNK] 토큰 삽입
  labels[~masked_indices] =  unknown_token_id # We only compute loss on masked tokens

  # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
  indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
  inputs[indices_replaced] = vocab.to_indices(vocab.mask_token)

  # 10% of the time, we replace masked input tokens with random word
  indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
  random_words = torch.randint(len(vocab), labels.shape, dtype=torch.long)
  inputs[indices_random] = random_words[indices_random]

  if pad:
    input_pads = max_len - inputs.shape[-1] # 인풋의 패딩 갯수 계산
    label_pads = max_len - labels.shape[-1] # 라벨의 패딩 갯수 계산

    inputs = F.pad(inputs, pad=(0, input_pads), value=padding_token_id)
    labels = F.pad(labels, pad=(0, label_pads), value=padding_token_id)

  # The rest of the time (10% of the time) we keep the masked input tokens unchanged
  return inputs, labels

if __name__=='__main__':
  # KoBERT 토크나이저
  tok_path = koBERTTokenizer()
  sentencepieceTokenizer = SentencepieceTokenizer(tok_path)
  sentencepieceDetokenizer = SentencepieceDetokenizer(tok_path)

  bertVocab = koBertVocab()
  bertTokenizer =BERTSPTokenizer(tok_path, bertVocab, lower=False)

  tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
  test_ko_str = '오늘은 날이 매우 좋은 날'
  test = 'Hello, my dog is cute'

  tokens = sentencepieceTokenizer(test_ko_str)
  tokens_index =bertVocab(tokens)
  added_specialtokens_index = add_special_tokens(bertTokenizer,tokens_index)

  input_tensor = torch.tensor(added_specialtokens_index)
  # inputs, labels = orgin_mask_tokens(tokenizer,tok.unsqueeze(0), pad=True)
  inputs, labels = kobert_mask_tokens(bertTokenizer, input_tensor.unsqueeze(0), pad=True)

  print('inputs index2token: ',bertVocab.to_tokens(inputs.squeeze().tolist()))
  print('labels index2token: ',bertVocab.to_tokens(labels.squeeze().tolist()))

