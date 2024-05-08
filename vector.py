from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertModel
import torch

app = Flask(__name__)

# BERT 모델 및 토크나이저 로드 (기본 768 차원)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 입력 텍스트를 BERT 임베딩으로 변환하는 함수
def text_to_vector(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = torch.mean(outputs.last_hidden_state, dim=1).squeeze()
    return embeddings.detach().numpy()

# API 엔드포인트
@app.route('/vectorize', methods=['POST'])
def vectorize_text():
    input_text = request.data.decode('utf-8')
    vector_representation = text_to_vector(input_text)
    return jsonify({'vector': vector_representation.tolist()})

if __name__ == '__main__':
    app.run(debug=True)