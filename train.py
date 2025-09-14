import torch
import pytorch_lightning as pl
from config import args
from data_utils import prepare_dataset
from model_utils import load_model_and_tokenizer, build_trainer

def train():
    """모델을 학습하고 최적의 모델을 저장"""
    pl.seed_everything(seed=args.seed, workers=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")

    model, tokenizer = load_model_and_tokenizer(args.model_name)
    model.to(device)

    train_dataset, val_dataset, _, _ = prepare_dataset(args.dataset_dir, tokenizer, args.max_len)
    
    trainer = build_trainer(model, train_dataset, val_dataset, args)

    print("--- 학습 시작 ---")
    trainer.train()
    print("--- 학습 완료 ---")

    model.save_pretrained(args.model_dir)
    print(f"최적의 모델이 '{args.model_dir}' 경로에 저장되었습니다.")


if __name__ == "__main__":
    train()