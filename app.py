# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from konlpy.tag import Okt
import pickle

app = Flask(__name__)

# Load resources once
model = load_model('model/LSTM.h5')
with open('model/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
twitter = Okt()

stopwords = ['질문', '문의', '관련', '그대로', '계속', '답변', '선생님', '관련문의',
            '한지', '자주', '좀', '쪽', '자꾸', '요즘', '몇개', '무조건', '하나요',
            '안해','요', '경우', '최근', '및', '몇', '달', '일반', '전날', '저번',
            '말', '일어나지', '며칠', '먹기', '지난번', '글', '때문', '너', '무',
            '오늘', '시', '잔', '뒤', '지속', '막', '것', '이건', '뭔가', '다시', '그',
            '무슨', '안', '난', '도', '기', '후', '거리', '이', '뭘', '저', '뭐', '답젼',
            '평생', '회복', '반', '감사', '의사', '보험', '학생', '제발', '살짝',
            '느낌', '제', '대해','갑자기','문제', '전','정도', '왜', '거', '가요',
            '의심', '어제', '추천', '를', '지금', '무엇', '내일', '관해', '리', '세',
             '로', '목적', '그냥', '거의', '고민', '다음', '이틀', '항상', '뭐', '때',
            '요', '가끔', '이후', '혹시']

label_to_koclass = {0:'피부과', 1:'외과', 2:'호흡기내과', 3:'소화기내과', 4:'안과',
                  5:'신경과', 6:'이비인후과', 7:'정신건강의학과', 8:'혈액종양내과', 9:'류마티스내과',
                  10:'재활의학과', 11:'신경외과', 12:'마취통증의학과', 13:'치과', 14:'성형외과',
                  15:'심장혈관흉부외과', 16:'감염내과', 17:'정형외과', 18:'응급의학과', 19:'내분비내과',
                  20:'순환기내과', 21:'한방과', 22:'산부인과', 23:'비뇨의학과', 24:'알레르기내과', 25:'신장내과'}

def preprocess_sentence(sentence):
    nouns = twitter.nouns(sentence)
    nouns = [word for word in nouns if word not in stopwords]
    input_batch = [nouns]
    embedded_batch = tokenizer.texts_to_sequences(input_batch)
    padded_batch = pad_sequences(embedded_batch, maxlen=10, padding='post', truncating='post')
    return padded_batch

def get_result(softmax):
    sp = softmax.argmax()
    return label_to_koclass[sp]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'sentence' not in data:
        return jsonify({'error': 'No sentence provided'}), 400
    
    sentence = data['sentence']
    processed = preprocess_sentence(sentence)
    prediction = model.predict(processed)
    result = get_result(prediction[0])
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
