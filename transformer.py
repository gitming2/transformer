import torch
torch.cuda.is_available() # GPU 할당여부 확인. GPU가 정상적으로 할당되면 True가 출력.

"""### 1️⃣ Dataset & Tokenizing
- torch.utils.data.Dataset class 를 상속받아 news_dataset class를 정의해줍니다.
- 데이터를 불러오는 load_data 와 tokenizing을 진행하는 <br> construct_tokenized_dataset 를 정의해줍니다.
-  위에서 정의된 함수들을 활용하여, 데이터셋을 불러와 tokenizing 한후에, <br> torch dataset class로 변환해줍니다.
- 이후 train data /validation data 는 7.5 : 2.5로 나눠줍니다.

####1-1. <b> news_dataset class </b> 정의
"""

import os
import pandas as pd
import torch

class news_dataset(torch.utils.data.Dataset):
    """dataframe을 torch dataset class로 변환"""
    def __init__(self, news_dataset, labels):
        self.dataset = news_dataset
        self.labels = labels

    def __getitem__(self,idx):
        item = {
            key: val[idx].clone().detach() for key, val in self.dataset.items() # key와 value를 각각 인덱스에 맞게 넣어줌
        }
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

"""####1-2. <b>load_data , construct_tokenized_dataset</b> 함수 정의
- "데이터를 불러오고" "토크나이징 해주는" 함수들
"""

def load_data(dataset_dir):
    """csv file을 dataframe으로 load"""
    dataset = pd.read_csv(dataset_dir)[:500] # 일단 500개만 불러옴
    print("dataframe 의 형태")
    print(dataset.head())
    return dataset

def construct_tokenized_dataset(dataset,tokenizer, max_length):
    """[뉴스제목 + [SEP] + 뉴스본문]형태로 토크나이징""" # 이렇게 엮어줄 생각!
    concat_entity = []
    for title, body in zip(dataset["newsTitle"],dataset["newsContent"]):
        total = str(title) + "[SEP]" + str(body) # 여기서 엮어줌 # 문장 구분을 위한 [SEP] 특수 토큰
        concat_entity.append(total) # 각 문장을 concat_entity 리스트에 붙여주기
    print("tokenizer 에 들어가는 데이터 형태")
    print(concat_entity[:5])
    tokenized_senetences = tokenizer( # 붙여놓은 걸 가져와서 토크나이징 해주기
        concat_entity,
        return_tensors = "pt",
        padding = True,
        truncation = True,
        max_length = max_length,
        add_special_tokens = True,
        # return_token_type_ids=False, # BERT 이후 모델(RoBERTa 등) 사용할때 False
    )
    print("tokenizing 된 데이터 형태")
    print(tokenized_senetences[:5])
    return tokenized_senetences

"""####1-3. prepare_dataset 함수 정의
- 앞서 정의한 함수들 기반으로 데이터셋 준비하는 함수

🌟 전처리과정 🌟<blockquote>
1. train.csv / test.csv 파일을 pd.dataframe 로 다운로드 해준다. <br>
2. train/validation set을 나눠준다. (7.5:2.5) <br>
3. label 값을 따로 저장해준다. <br>
4. 제목과 본문 데이터만 정제한후에 tokenizing 해준다. <br>
5. tokenizing 된 데이터를 news_dataset class로 반환해준다. <br>
"""

def prepare_dataset(dataset_dir, tokenizer,max_len):
    """학습(train)과 평가(test)를 위한 데이터셋을 준비"""
    # load_data
    train_dataset = load_data(os.path.join(dataset_dir, "train.csv"))
    test_dataset = load_data(os.path.join(dataset_dir, "test.csv"))

    # split train / validation = 7.5 : 2.5
    train_dataset, val_dataset = train_test_split(train_dataset,test_size=0.25,random_state=42,stratify=train_dataset['label']) # stratify: 나눌 때, 클래스 비율도 동일하게 나눠줌!

    # split label
    train_label = train_dataset['label'].values
    val_label = val_dataset['label'].values
    test_label = test_dataset['label'].values

    # tokenizing dataset
    tokenized_train = construct_tokenized_dataset(train_dataset, tokenizer, max_len)
    tokenized_val = construct_tokenized_dataset(val_dataset, tokenizer, max_len)
    tokenized_test = construct_tokenized_dataset(test_dataset, tokenizer, max_len)
    print("--- tokenizing Done ---") # 이렇게 로깅을 해주면 좋음!

    # make dataset for pytorch.
    news_train_dataset = news_dataset(tokenized_train, train_label)
    news_val_dataset = news_dataset(tokenized_val, val_label)
    news_test_dataset = news_dataset(tokenized_test, test_label)
    print("--- dataset class Done ---")

    return news_train_dataset , news_val_dataset, news_test_dataset , test_dataset

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')

df = load_data("./data/train.csv")

tokenized_data = construct_tokenized_dataset(df,tokenizer,120)

for key ,value in tokenized_data.items():
    print(key)

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained('klue/bert-base')


"""### 2️⃣ Model & Trainer
- huggingface 에서 사전학습된(pre-trained) 모델을 불러옵니다.
- huggingface 의 Trainer 모듈을 정의하고 학습에 사용될 Arguments 들을 지정해줍니다.
"""

import os
import random
import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
)
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
from transformers.optimization import get_cosine_with_hard_restarts_schedule_with_warmup

"""####2-1. compute_metrics 함수 정의
- 학습 중 validation 할때 사용될 평가지표 정의하는 함수
- 해당 실습에서는 Accuracy와 F1 Score를 Metric으로 사용
"""

def compute_metrics(pred):
    """validation을 위한 metrics function"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)

    # calculate f1 score using sklearn's function
    f1 = f1_score(labels, preds, average='micro')

    return {
        "accuracy": acc,
        "f1": f1,
    }

"""####2-2.load_tokenizer_and_model_for_train 함수 정의
- 학습에 사용될 토크나이저와 모델을 불러오는 함수
"""

def load_tokenizer_and_model_for_train():
    """학습(train)을 위한 사전학습(pretrained) 토크나이저와 모델을 huggingface에서 load"""
    # load model and tokenizer
    MODEL_NAME = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # setting model hyperparameter # config를 따로 만들어주는 이유: label이 2가 아닐 수도 있어서;
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 2
    print(model_config)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, config=model_config
    )
    print("--- Modeling Done ---")
    return tokenizer , model

"""####2-3.load_trainer_for_train 함수 정의
- 학습에 사용될 Trainer 모듈을 정의하고 Arguments들을 지정해준다.
"""

def load_trainer_for_train(model,news_train_dataset,news_val_dataset):
    """학습(train)을 위한 huggingface trainer 설정"""
    training_args = TrainingArguments(
        output_dir=args.save_path + "results",  # output directory
        save_total_limit=args.save_limit,  # number of total save model.
        save_steps=args.save_step,  # model saving step.
        num_train_epochs=args.epochs,  # total number of training epochs
        learning_rate=args.lr,  # learning_rate
        per_device_train_batch_size=args.batch_size,  # batch size per device during training
        per_device_eval_batch_size=2,  # batch size for evaluation
        warmup_steps=args.warmup_steps,  # number of warmup steps for learning rate scheduler
        weight_decay=args.weight_decay,  # strength of weight decay
        logging_dir=args.save_path + "logs",  # directory for storing logs
        logging_steps=args.logging_step,  # log saving step.
        eval_strategy="steps",  # evaluation strategy to adopt during training # val에서의 eval
            # `no`: No evaluation during training.
            # `steps`: Evaluate every `eval_steps`.
            # `epoch`: Evaluate every end of epoch.
        eval_steps=args.eval_step,  # evaluation step.
        load_best_model_at_end=True,
    )

    ## Add callback & optimizer & scheduler
    MyCallback = EarlyStoppingCallback(
        early_stopping_patience=3, early_stopping_threshold=0.001
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.weight_decay,
        amsgrad=False,
    )
    print("--- Set training arguments Done ---")

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=news_train_dataset,  # training dataset
        eval_dataset=news_val_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,  # define metrics function
        callbacks=[MyCallback],
        optimizers=(
            optimizer,
            get_cosine_with_hard_restarts_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=args.warmup_steps,
                    num_training_steps=len(news_train_dataset) * args.epochs,
            ),
        ),
    )
    print("--- Set Trainer Done ---")

    return trainer

"""####2-4.train 함수 정의
- 실험세팅 후 앞서 정의한 함수들 활용하여 학습을 진행하는 함수

🌟 학습동작과정 🌟
<blockquote>
1. 실험에 영향을 주는 모든 seed를 고정해준다. <br>
2. 사용할 gpu를 device에 할당해준다. <br>
3. tokenizer와 model을 불러온후, model을 device에 할당해준다. <br>
4. 학습에 사용될 news_dataset 을 불러온다.<br>
5. 학습에 사용될 Trainer 를 불러온다.<br>
6. 학습을 진행한후에 best_model을 저장해준다. <br>
"""

def train():
    """모델을 학습(train)하고 best model을 저장"""
    # fix a seed
    pl.seed_everything(seed=42, workers=False)

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # set model and tokenizer
    tokenizer , model = load_tokenizer_and_model_for_train()
    model.to(device)

    # set data
    news_train_dataset, news_val_dataset, news_test_dataset, test_dataset = prepare_dataset(args.dataset_dir,tokenizer,args.max_len) # 평가할 때 쓰려고 test_dataset 따로 빼둠

    # set trainer
    trainer = load_trainer_for_train(model,news_train_dataset,news_val_dataset)

    # train model
    print("--- Start train ---")
    trainer.train()
    print("--- Finish train ---")
    model.save_pretrained("./best_model")

"""####2-5.arguments 지정 및 학습 진행
- RoBERTa 모델 간략 설명

<blockquote>
1. BERT 모델의 변형으로, 학습 데이터의 양을 크게 늘리고 학습률(learning rate)을 조정하며, <br>문장의 길이를 다양화하여 성능을 향상시켰다. <br>
2. BERT의 사전학습 방법 중 하나인 NSP(Next Sentence Prediction)을 제거하였다.
"""

class args (): # 학습에 사용되는 파라미터들을 여기에 모아놓음
  """학습(train)과 추론(infer)에 사용되는 arguments 관리하는 class"""
  dataset_dir = "./data"
  model_type = "roberta" # 다른 모델 사용 가능 e.g) "bert" , "electra" ···
  model_name = "klue/roberta-large" # 다른 모델 사용 가능 e.g) "klue/bert-base" , "monologg/koelectra-base-finetuned-nsmc" ···
  save_path = "./"
  save_step = 200
  logging_step = 200
  eval_step = 100
  save_limit = 5
  seed = 42
  epochs = 1 # 10
  batch_size = 8 # 메모리 상황에 맞게 조절 e.g) 16 or 32
  max_len = 256
  lr = 3e-5
  weight_decay = 0.01
  warmup_steps = 300
  scheduler = "linear"
  model_dir = "./best_model" #추론 시, 저장된 모델 불러오는 경로 설정

train()

"""### 3️⃣ Inference & Evaluation
- 학습완료된(fine-tuned) 모델을 불러와서 추론(infer)을 진행합니다.
- 추론된 예측값들과 정답값을 비교하여 평가(evaluation)합니다.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
import pandas as pd
import torch
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

"""####3-1.load_model_for_inference 함수 정의
- 학습된(fine-tuned) 모델의 체크포인트(checkpoint)를 불러오는 함수 <br>
<b>(이때, 토크나이저는 기존과 동일하게 huggingface 에서 불러온다. )</b>

"""

def load_model_for_inference():
    """추론(infer)에 필요한 모델과 토크나이저 load """
    # load tokenizer
    Tokenizer_NAME = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)

    ## load my model
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir) # 여기선 model name말고, directory 넣어줌. 왜? 학습한 걸 불러와야 되기 때문!

    return tokenizer, model

"""####3-2. inference 함수 정의
- 학습된(fine-tuned)모델을 통해 평가 데이터의 예측값 추론해내는 함수

🌟 추론(infer) 계산과정 🌟

<blockquote>
1. model.eval , torh.no_grad 를 통해 모델을 추론 모드로 변경 <br>
2. 모델에 입력값으로 input_ids 와 attention mask를 <b>gpu에 할당한 후</b> 입력으로 주고 결과값(outputs) 생성 <br>
3. 결과값(outputs) 중 logits 값을 cpu로 할당한 후, argmax 를 통해 예측 레이블(label) 생성 <br>
4. 생성된 레이블(label) 을 concat 하여 리스트 형태로 반환
"""

def inference(model, tokenized_sent, device):
    """학습된(trained) 모델을 통해 결과를 추론하는 function"""
    dataloader = DataLoader(tokenized_sent, batch_size=args.batch_size, shuffle=False)
    model.eval()
    output_pred = []
    for i, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            outputs = model(
                input_ids=data["input_ids"].to(device),
                attention_mask=data["attention_mask"].to(device),
            )
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)

        output_pred.append(result)
    return (np.concatenate(output_pred).tolist(),)

"""####3-3.infer_and_eval 함수 정의
- 학습된(fine-tuned) 모델로 추론(infer)한 후 예측한 결과를 평가(evaluation)하는 함수

<blockquote>
1. 사용할 gpu를 device에 할당해준다. <br>
2. tokenizer와 model을 불러온후, model을 device에 할당해준다. <br>
3. 추론에 사용될 news_dataset 을 불러온다.<br>
4. model 과 news_dataset을 입력으로 주고 추론(infer)을 진행한다. <br>
5. test data 의 레이블(label)과 예측값(pred)을 비교하여 평가지표를 계산한다.<br>
6. 최종 예측값을 csv 형태로 저장해준다. <br>
"""

def infer_and_eval():
    """학습된 모델로 추론(infer)한 후에 예측한 결과(pred)를 평가(eval)"""
    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # set model & tokenizer
    tokenizer, model = load_model_for_inference()
    model.to(device)

    # set data
    news_train_dataset, news_val_dataset, news_test_dataset, test_dataset = prepare_dataset(args.dataset_dir,tokenizer,args.max_len)

    # predict answer
    pred_answer = inference(model, news_test_dataset, device)  # model에서 class 추론
    print("--- Prediction done ---")

    # evaluate between label and prediction
    labels = test_dataset['label'].values
    pred = pred_answer[0]

    acc = accuracy_score(labels, pred)
    f1 = f1_score(labels, pred, average='macro')
    print(f" ----- accuracy:{acc * 100:.1f}% -----")
    print(f"----- f1_score(macro): {f1 * 100:.1f}% ------")

    # make csv file with predicted answer
    output = pd.DataFrame(
        {
            "title": test_dataset["newsTitle"],
            "cleanBody": test_dataset["newsContent"],
            "clickbaitClass": pred,
        }
    )

    # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    result_path = "./prediction/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    output.to_csv(
        os.path.join(result_path,"result.csv"), index=False
    )
    print("--- Save result ---")
    return output # 밑에서 찍어보려고 추가한 거임(없어도 됨)

"""- 추론 및 평가 진행 후 결과값 10개까지 출력"""

output_df = infer_and_eval()
output_df.head(10)