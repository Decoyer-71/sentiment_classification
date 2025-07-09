from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 모델 경로
MODEL_PATH = './model'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# 모델도 같은 디바이스에 올리기
model.to(device)
model.eval()  # 평가모드 전환(vs : model.train() 훈련모드)


def predict_sentiment(text):
    # 토크나이징
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # 입력 텐서를 모델과 동일한 device로 설정

    # 추론
    with torch.no_grad():
        outputs = model(**inputs)
        pred = outputs.logits.argmax(dim=1).item()

    return '긍정' if pred == 1 else '부정'
