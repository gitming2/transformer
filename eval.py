import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score

from config import args
from data import prepare_dataset

def inference(model, test_dataset, device, batch_size):
    """학습된 모델로 결과를 추론"""
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    output_pred = []

    for data in tqdm(dataloader, desc="추론 진행"):
        with torch.no_grad():
            outputs = model(
                input_ids=data["input_ids"].to(device),
                attention_mask=data["attention_mask"].to(device),
            )
        logits = outputs.logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)
        output_pred.extend(result)
        
    return output_pred

def main():
    """추론 및 평가 실행"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")

    # 저장된 모델과 토크나이저 로드
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model.to(device)
    
    # 테스트 데이터셋 준비
    _, _, test_dataset, test_df = prepare_dataset(args.dataset_dir, tokenizer, args.max_len)
    
    # 추론 실행
    pred_answer = inference(model, test_dataset, device, args.batch_size)
    print("--- 추론 완료 ---")
    
    # 평가
    labels = test_df['label'].values
    acc = accuracy_score(labels, pred_answer)
    f1 = f1_score(labels, pred_answer, average='macro')
    
    print(f"정확도(Accuracy): {acc * 100:.2f}%")
    print(f"F1 Score (Macro): {f1 * 100:.2f}%")

    # 결과 파일 저장
    output = pd.DataFrame({
        "newsTitle": test_df["newsTitle"],
        "newsContent": test_df["newsContent"],
        "label": labels,
        "prediction": pred_answer,
    })

    result_path = "./prediction/"
    os.makedirs(result_path, exist_ok=True)
    output.to_csv(os.path.join(result_path, "result.csv"), index=False, encoding='utf-8-sig')
    print(f"추론 결과가 '{result_path}result.csv' 파일로 저장되었습니다.")

if __name__ == "__main__":
    main()