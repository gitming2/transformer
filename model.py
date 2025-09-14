import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from transformers.optimization import get_cosine_with_hard_restarts_schedule_with_warmup

def compute_metrics(pred):
    """Validation을 위한 평가지표 계산 함수"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='micro')
    return {"accuracy": acc, "f1": f1}

def load_model_and_tokenizer(model_name):
    """사전학습된 토크나이저와 모델 로드"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_config = AutoConfig.from_pretrained(model_name)
    model_config.num_labels = 2
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=model_config)
    return model, tokenizer

def build_trainer(model, train_dataset, val_dataset, args):
    """Huggingface Trainer 설정"""
    training_args = TrainingArguments(
        output_dir=args.save_path + "results",
        save_total_limit=args.save_limit,
        save_steps=args.save_step,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_dir=args.save_path + "logs",
        logging_steps=args.logging_step,
        evaluation_strategy="steps",
        eval_steps=args.eval_step,
        load_best_model_at_end=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=len(train_dataset) * args.epochs,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)],
        optimizers=(optimizer, scheduler),
    )
    return trainer