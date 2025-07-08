from datasets import load_dataset  # Hugging Face Datasets 라이브러리에서 불러옴
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
from transformers import DataCollatorWithPadding, logging
import torch

# 데이터셋 불러오기
dataset = load_dataset('e9t/nsmc', trust_remote_code=True)

# 토크나이저
pretrained_model = 'klue/bert-base'
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

def tokenize_function(example) : # 데이터셋의 내용에 해당하는 'document'에 대해서 토큰화하도록 함수 설정
    return tokenizer(example['document'], truncation  = True, padding = True) # padding : 길이에 맞게 패딩

tokenized_dataset = dataset.map(tokenize_function, batched = True)

# 디바이스 설정 및 gpu사용여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 현재 사용 디바이스: {device}")
if device.type == 'cuda':
    print(f"🎯 GPU 이름: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ 현재 GPU를 사용하지 않고 있습니다.")

# 로그 레벨 설정
logging.set_verbosity_info()

# 모델 구축
model = AutoModelForSequenceClassification.from_pretrained(pretrained_model, num_labels = 2)
model.to(device) # 모델을 gpu로 이동

# 훈련 설정
args = TrainingArguments(
    output_dir='./results',
    learning_rate=2e-5,
    num_train_epochs=3,
    fp16=True, # gpu 반정밀도 훈련 : 모델 일부 연산을 fp16(표현력 낮음, 연산빠름, 메모리 적게)으로 수행하고, 일부는 fp32(표현력 높음, 연산 느림, 메모리 많이)로 유지해서 속도와 메모리 효율을 동시에 얻는 방식
    logging_strategy="steps", # 'epoch', 'no' / HuggingFace Traniner 훈련 중 로그를 언제 출력할지 제어하는 설정
    logging_steps=100,
    per_device_train_batch_size=32,
    weight_decay=0.01
)

# 평가함수 설정
accuracy = evaluate.load('accuracy')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis = -1)
    return accuracy.compute(predictions=predictions, references = labels)

# 훈련
data_collator = DataCollatorWithPadding(tokenizer=tokenizer) # 데이터 batch를 만들면서 패딩 길이를 자동으로 못 맞춰서 생기는 오류를 방지

trainer = Trainer( # model.train() 훈련모드를 자동으로 호출
    model = model,
    args = args,
    train_dataset=tokenized_dataset['train'],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer = tokenizer
)

trainer.train()

# 모델 저장(inference.py에서 추론함수가 호출한 모델을 사용할 수 있도록)
trainer.save_model('./model')
tokenizer.save_pretrained('./model')