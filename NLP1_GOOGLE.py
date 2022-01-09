from tensorflow.keras.preprocessing.text import Tokenizer

#语料库
sentences = [
    'I love my dog',
    'I love my cat'
]
#生成大小为100的分词器
tokenizer = Tokenizer(num_words=100)

#对语料进行分词
tokenizer.fit_on_texts(sentences)

#获取单词索引
word_index = tokenizer.word_index   

print(word_index)