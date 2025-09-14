import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

class news_dataset(torch.utils.data.Dataset):
    """DataFrame을 torch.utils.data.Dataset으로 변환"""
    def __init__(self, tokenized_dataset, labels):
        self.dataset = tokenized_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: val[idx].clone().detach() for key, val in self.dataset.items()
        }
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def load_data(dataset_dir):
    """CSV 파일을 DataFrame으로 로드"""
    dataset = pd.read_csv(dataset_dir)
    return dataset

def construct_tokenized_dataset(dataset, tokenizer, max_length):
    """[뉴스제목 + [SEP] + 뉴스본문] 형태로 토크나이징"""
    concat_entity = [str(title) + "[SEP]" + str(body) for title, body in zip(dataset["newsTitle"], dataset["newsContent"])]
    
    tokenized_sentences = tokenizer(
        concat_entity,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    return tokenized_sentences

def prepare_dataset(dataset_dir, tokenizer, max_len):
    """학습 및 평가를 위한 데이터셋 준비"""
    train_df_full = load_data(os.path.join(dataset_dir, "train.csv"))
    test_df = load_data(os.path.join(dataset_dir, "test.csv"))

    train_df, val_df = train_test_split(train_df_full, test_size=0.25, random_state=42, stratify=train_df_full['label'])

    train_label = train_df['label'].values
    val_label = val_df['label'].values
    test_label = test_df['label'].values

    tokenized_train = construct_tokenized_dataset(train_df, tokenizer, max_len)
    tokenized_val = construct_tokenized_dataset(val_df, tokenizer, max_len)
    tokenized_test = construct_tokenized_dataset(test_df, tokenizer, max_len)

    train_dataset = news_dataset(tokenized_train, train_label)
    val_dataset = news_dataset(tokenized_val, val_label)
    test_dataset = news_dataset(tokenized_test, test_label)
    
    return train_dataset, val_dataset, test_dataset, test_df