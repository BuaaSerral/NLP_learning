#读取数据
import json
from re import VERBOSE

from tensorflow.python.keras.utils.generic_utils import validate_config

with open(r"D:\Individual\Work\Documents\Python_vscode\NLP_learning\archive\Sarcasm_Headlines_Dataset.json", 'r') as f:
    datastore = json.load(f)
#FUCK Google视屏里给的这个json文件没逗号 我傻了
sentences = [] 
labels = []
urls = []
for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

#划分测试集与训练集
training_size = 20000

training_sentences = sentences[0:training_size]
training_labels = labels[0:training_size]
testing_sentences = sentences[training_size:]
testing_labels = labels[training_size:]

#生产对应的训练向量与测试向量
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words = 1000,oov_token='<OOV>')
tokenizer.fit_on_texts(training_sentences)

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences)

#神经网络
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = tf.keras.Sequential([
    layers.Embedding(1000,16),
    layers.GlobalAveragePooling1D(),
    layers.Dense(16,activation='relu'),
    layers.Dense(1,activation='sigmoid')
])
model.compile(loss = 'binary_crossentropy',optimizer = 'adam',metrics = ['accuracy'])

num_epochs = 30
history = model.fit(training_padded,training_labels,epochs = num_epochs,
    validation_data = (testing_padded,testing_labels), VERBOSE= 2
)