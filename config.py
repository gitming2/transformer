class args:
    """학습 및 추론에 사용되는 arguments 관리 클래스"""
    # 경로 설정
    dataset_dir = "./data/"
    save_path = "./"
    model_dir = "./best_model" # 추론 시, 저장된 모델 불러오는 경로

    # 모델 설정
    model_name = "klue/roberta-large"

    # 학습 하이퍼파라미터
    epochs = 1
    batch_size = 8
    max_len = 256
    lr = 3e-5
    weight_decay = 0.01
    warmup_steps = 300

    # Trainer 설정
    seed = 42
    save_limit = 5
    save_step = 200
    logging_step = 200
    eval_step = 100