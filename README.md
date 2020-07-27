# ReforBERT
Transformer를 개선한 Reformer를 이용한 BERT (ver.pytorch)

##  Introduction
2020년 트랜스포머를 개선한 리포머 발표. 
리포머는 트랜스포머의 제한 사항들을 **LSH**, **RevNet**, **Chunk**을 통해 개선하였다. 
BERT나 GPT2와 같은 큰 모델들은 많은 컴퓨팅과 메모리를 필요로 하여, 고가의 장비 없이 직접 학습시키는데 많은 제한이 있었다.
[lucidrains/reformer-pytorch](https://github.com/lucidrains/reformer-pytorch)를 이용하여 
리포머를 이용한 **BERT**를 만들고 colab을 통해 LM을 학습해 다양한 downstream task에 테스트.
  
## Architecture
### 1. Data
한국어 위키피디아 데이터
```
지미 카터
제임스 얼 "지미" 카터 주니어(, 1924년 10월 1일 ~ )는 민주당 출신 미국 39번째 대통령 (1977년 ~ 1981년)이다.
지미 카터는 조지아주 섬터 카운티 플레인스 마을에서 태어났다. 조지아 공과대학교를 졸업하였다. 그 후 해군에 들어가 전함·원자력·잠수함의 승무원으로 일하였다. 1953년 미국 해군 대위로 예편하였고 이후 땅콩·면화 등을 가꿔 많은 돈을 벌었다. 그의 별명이 "땅콩 농부" (Peanut Farmer)로 알려졌다.
1962년 조지아 주 상원 의원 선거에서 낙선하나 그 선거가 부정선거 였음을 입증하게 되어 당선되고, 1966년 조지아 주 지사 선거에 낙선하지만 1970년 조지아 주 지사를 역임했다. 대통령이 되기 전 조지아주 상원의원을 두번 연임했으며, 1971년부터 1975년까지 조지아 지사로 근무했다. 조지아 주지사로 지내면서, 미국에 사는 흑인 등용법을 내세웠다.
...중략...
```
#### Pretrain
한국어 위키 덤프
##### 전처리
  1. 위키 미러에서 덤프 다운로드
  2. sentencepiece를 이용해 8000개의 vocab 생성
  3. JSON 형태로 데이터 변환 
  ```json
    {"tokens": ["[CLS]", "\uc6d4", "\u25811", "[MASK]", "\u2581~", "\u2581)", "\ub294", "\u2581\ubbfc\uc8fc", "\ub2f9", "\u2581\ucd9c\uc2e0", "\u2581\ubbf8\uad6d", "\u25813", "9", "\ubc88\uc9f8", "[MASK]", "\u2581(19", "7", "7", "\ub144", "\u2581~", "\u25811981", "\ub144", ")", "\uc774\ub2e4", ".", "\u2581\uc9c0", "\ubbf8", "\u2581\uce74", "\ud130", "\ub294", "\u2581\uc870\uc9c0", "\uc544", "\uc8fc", "\u2581\uc12c", "\ud130", "\u2581\uce74", "\uc6b4", "\ud2f0", "[MASK]", "[MASK]", "\uce69", "[MASK]", "\u2581\ub9c8\uc744", "\uc5d0\uc11c", "\u2581\ud0dc\uc5b4\ub0ac\ub2e4", ".", "\u2581\uc870\uc9c0", "\uc544", "\u2581\uacf5", "\uacfc", "\ub300\ud559\uad50", "\ub97c", "\u2581\uc878\uc5c5", "\ud558\uc600\ub2e4", ".", "\u2581\uadf8", "\u2581\ud6c4", "\u2581\ud574", "\uad70\uc5d0", "\u2581\uad00", "\u2581\uc804", "\ud568", "\u00b7", "\uc6d0", "\uc790", "\ub825", "\u00b7", "\uc7a0", "\uc218", "\ud568", "\uc758", "\u2581\uc2b9", "\ubb34", "\uc6d0\uc73c\ub85c", "\u2581\uc77c", "\ud558\uc600\ub2e4", ".", "\u2581195", "3", "\ub144", "\u2581\ubbf8\uad6d", "\u25811930", "\u2581\ub300", "\uc704\ub85c", "\u2581\uc608", "\ud3b8", "\ud558\uc600\uace0", "\u2581\uc774\ud6c4", "\u2581\ub545", "\ucf69", "\u00b7", "\uba74", "\ud654", "\u2581\ub4f1\uc744", "\u2581\uac00", "\uafd4", "\u2581\ub9ce\uc740", "\u2581\ub3c8", "\uc744", "\u2581\ubc8c", "\uc5c8\ub2e4", ".", "\u2581\uadf8\uc758", "\u2581\ubcc4", "\uba85\uc774", "\u2581\"", "\ub545", "\ucf69", "\u2581\ub18d", "\ubd80", "\"", "\u2581(", "P", "e", "an", "ut", "\u2581F", "ar", "m", "er", ")", "\ub85c", "[MASK]", "\uc84c\ub2e4", "[MASK]", "[MASK]", "[MASK]", "[MASK]", "\u2581\uc870\uc9c0", "[MASK]", "\u2581\uc8fc", "\u2581\uc0c1", "\uc6d0", "\u2581\uc758\uc6d0", "\u2581\uc120\uac70", "\uc5d0\uc11c", "[MASK]", "[MASK]", "\ud558\ub098", "\u2581\uadf8", "[MASK]", "[MASK]", "\u2581\ubd80\uc815", "\uc120\uac70", "[MASK]", "[MASK]", "[MASK]", "\u2581\uc785", "\uc99d", "\ud558\uac8c", "\u2581\ub418\uc5b4", "\u2581\ub2f9\uc120", "\ub418\uace0", ",", "\u2581196", "6", "\ub144", "\u2581\uc870\uc9c0", "\uc544", "\u2581\uc8fc", "\u2581\uc9c0", "\uc0ac", "\u2581\uc120\uac70", "\uc5d0", "\u2581\ub099", "\uc120", "\ud558\uc9c0\ub9cc", "\u25811970", "\ub144", "\u2581\uc870\uc9c0", "\uc544", "\u2581\uc8fc", "\u2581\uc9c0", "\uc0ac\ub97c", "\u2581\uc5ed\uc784", "\ud588\ub2e4", ".", "\u2581\ub300\ud1b5\ub839", "\uc774", "\u2581\ub418", "\uae30", "\u2581\uc804", "[MASK]", "[MASK]", "[MASK]", "\u2581\uc0c1", "\uc6d0\uc758", "\uc6d0\uc744", "\u2581\ub450", "\ubc88", "\u2581\uc5f0", "\uc784", "\ud588\uc73c\uba70", ",", "[MASK]", "[MASK]", "\u25811975", "\ub144\uae4c\uc9c0", "\u2581\uc870\uc9c0", "\uc544", "\u2581\uc9c0", "\uc0ac\ub85c", "\u2581\uadfc\ubb34", "\ud588\ub2e4", ".", "\u2581\uc870\uc9c0", "\uc544", "\u2581\uc8fc", "\uc9c0", "\uc0ac\ub85c", "\u2581\uc9c0", "\ub0b4", "\uba74\uc11c", ",", "\u2581\ubbf8\uad6d", "\uc5d0", "\u2581\uc0ac\ub294", "\u2581\ud751", "\uc778", "[MASK]", "[MASK]", "[MASK]", "\u2581\ub0b4", "\uc138", "\uc6e0\ub2e4", ".", "[SEP]", "\u25811976", "\ub144", "[MASK]", "\u2581\uc120\uac70", "\uc5d0", "\u2581\ubbfc\uc8fc", "\ub2f9", "\u2581\ud6c4\ubcf4", "\ub85c", "\u2581\ucd9c", "\ub9c8", "\ud558\uc5ec", "\u2581\ub3c4", "\ub355", "\uc8fc\uc758", "\u2581\uc815\ucc45", "\uc73c\ub85c", "\u2581\ub0b4", "\uc138", "\uc6cc", ",", "\u2581\ud3ec", "\ub4dc\ub97c", "[MASK]", "[MASK]", "\u2581\ub2f9\uc120", "\ub418\uc5c8\ub2e4", ".", "[SEP]"], 
    "segment": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
    "is_next": 1, 
    "mask_idx": [2, 3, 14, 38, 39, 40, 41, 55, 59, 81, 122, 123, 124, 125, 126, 127, 128, 129, 136, 137, 138, 140, 141, 144, 145, 146, 182, 183, 184, 194, 195, 219, 220, 221, 229, 250, 251], 
    "mask_label": ["\u25811", "\uc77c", "\u2581\ub300\ud1b5\ub839", "\u2581\ud50c", "\ub808", "\uc778", "\uc2a4", "\u2581\uadf8", "\u2581\ub4e4\uc5b4\uac00", "\u2581\ud574\uad70", "\u2581\uc54c\ub824", "\uc84c\ub2e4", ".", "\u2581196", "2", "\ub144", "\u2581\uc870\uc9c0", "\uc544", "\u2581\ub099", "\uc120", "\ud558\ub098", "\u2581\uc120\uac70", "\uac00", "\u2581", "\uc600", "\uc74c\uc744", "\u2581\uc870\uc9c0", "\uc544", "\uc8fc", "\u25811971", "\ub144\ubd80\ud130", "\u2581\ub4f1", "\uc6a9", "\ubc95\uc744", "\u2581\ub300\ud1b5\ub839", "\u2581\ub204", "\ub974\uace0"]}
  ``` 
### Vocab 및 Tokenizer
SentencePiece Tokenizer 및 위키로 만든 8007개의 Vocab
```
[PAD]	0
[UNK]	0
[BOS]	0
[EOS]	0
[SEP]	0
[CLS]	0
[MASK]	0
▁1	-0
▁이	-1
으로	-2
에서	-3
▁있	-4
▁2	-5
▁그	-6
▁대	-7
▁사	-8
이다	-9
었다	-10
...중략...
```
### 2. Model
Reformer-pytorch의 Reformer 사용.

#### 2.1 Config
```
    count = 10            # 학습 데이터 분할 크기 kowiki_bert_{}.json
    learning_rate = 5e-5  # Learning Rate
    n_epoch = 20          # Num of Epoch
    batch_size = 128      # 배치 사이즈
    device ="cpu"         # cpu or cuda

    vocab_size = 8007     # vocab 크기
    max_seq_len = 512     # 최대 입력 길이
    embedding_size = 768  # 임베딩 사이
    batch_size = 128      # 학습 시 배치 크기
    depth = 12            # reformer depth
    heads = 8             # reformer heads

    train_save_step = 100 # 학습 저장 주기
```


### 3. Pretrain
기본 BERT의 Masked Language Model과 Next Sentence Prediction을 사전학습에 사용.

#### 3.1 Masked Language Model

#### 3.2 Next Sentence Prediction

#### 3.3 학습
1. 학습 모델  
Colab pro에서 P100을 이용해 학습을 진행해 보니 base 모델을 학습하는데, 많은 시간이 소요됨. 따라서 `BERT-Small` 모델을 통해 학습을 다시 진행
  
|   |H=128|H=256|H=512|H=768|
|---|:---:|:---:|:---:|:---:|
| **L=2**  |[**2/128 (BERT-Tiny)**]|[2/256]|[2_512]|[2_768]|
| **L=4**  |[4/128]|[**4/256 (BERT-Mini)**]|[**4/512 (BERT-Small)**]|[4/768]|
| **L=6**  |[6/128]|[6/256]|[6/512]|[6/768]|
| **L=8**  |[8/128]|[8/256]|[**8/512 (BERT-Medium)**]|[8/768]|
| **L=10** |[10/128]|[10/256]|[10/512]|[10/768]|
| **L=12** |[12/128]|[12/256]|[12/512]|[**12/768 (BERT-Base)**]|

## Train Environment
Colab GPU 메모리 12G 이상.
> 12G 이하로는 학습 되지 않음. 
 
## 학습결과 
|                    | Size  | **NSMC**<br/>(acc) | **Naver NER**<br/>(F1) | **PAWS**<br/>(acc) | **KorNLI**<br/>(acc) | **KorSTS**<br/>(spearman) | **Question Pair**<br/>(acc) | **KorQuaD (Dev)**<br/>(EM/F1) |
| :----------------- | :---: | :----------------: | :--------------------: | :----------------: | :------------------: | :-----------------------: | :-------------------------: | :---------------------------: |
| ReforBERT          |       |                    |                        |                    |                      |                           |                             |                               |

##  License
MIT

##  Author
Seonghwan Kim 

## Log
| 일자 | 내용| 비고 |
|---|---|---|
|20.04.25| 시작 | |
|20.05.08| Pretrain 코드 테스트| |
|20.05.10| Colab에서 Pretrain 테스트| |
|20.05.11| Colab 학습 중지 후 재개 부분 추가| |
|2020.05.11~20.05.24| **BERT-base**모델 Colab 학습 진행| 학습시간이 오래 걸리는 문제로 **BERT-small**로 변경 |
|20.7.12| **BERT-small**모델 Colab 학습 진행| |
|20.7.27| **BERT-base**모델 nipa 학습 진행| |




# Reference
[lucidrains/reformer-pytorch](https://github.com/lucidrains/reformer-pytorch)  
[BERT(Bidirectional Encoder Representations from Transformers) 구현하기 (1/2)](https://paul-hyun.github.io/bert-01/)  
[SKTBrain/KoBERT](https://github.com/SKTBrain/KoBERT)  
[monologg/KoELECTRA](https://github.com/monologg/KoELECTRA)
