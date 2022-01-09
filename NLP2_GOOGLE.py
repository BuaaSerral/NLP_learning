import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import sequence

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
]

#oov_token参数将未在语料库里出现的单词全部标记为OOV
tokenizer = Tokenizer(num_words = 100,oov_token='<OOV>')

tokenizer.fit_on_texts(sentences)

#将每个单词标号
word_index = tokenizer.word_index
#将每个句子转换成数域上的向量
sequences = tokenizer.texts_to_sequences(sentences)
#将每个向量的长度补齐，没有的标0
padded = pad_sequences(sequences)

'''
print(word_index)
print(sequences)
print(padded)
'''

test_data = [
    'i really love my dog',
    'my dog loves my manatee'
]

test_seq = tokenizer.texts_to_sequences(test_data)

test_padded = pad_sequences(test_seq)

print(test_seq)
print(test_padded)

