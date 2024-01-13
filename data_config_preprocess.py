# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 16:56:30 2019
@author: dongdongdong
仅用于科研与学术
请勿随意传播
"""
import pandas as pd
#from gensim.models.doc2vec import Doc2Vec
import numpy as np
import myProgressbar as pb
from collections import Counter
from PIL import Image
import keras

import os
import re

try:
    from keras import preprocessing
except:
    from tensorflow.keras import preprocessing
        
class ConfigWord(object):
    embedding_source_path = ''      #词嵌入路径
    sentence_len = 30       #模型文本长度

class ConfigPic(object):
    width = 224         #图片宽度
    height = 224        #图片长度
    train_image_dir_path = '' #训练集图片路径
    valid_image_dir_path = '' #验证集图片路径
    test_image_dir_path = '' #测试集图片路径

class Model(object):
    
    mask_zero = False
    multi_channel_num = 3 #通道数
    global_att_num = 3 #多头数
    dropout_rate = 0.5 #丢弃率
    spa_drop_rate = 0.1   #词丢弃

class Config(object):
    def __init__(self):
        
        if self.init is True:
            return
        else:
            self.init = True
        
        self.content = ConfigWord() # 文本设置
        self.pic = ConfigPic() #图片设置
        self.model = Model() #模型设置
        self.gpu_name='0' # GPU
        self.database_name = 'Twitter' # 数据集名称
        
        self.trainset_path = r'./twitter/twitter.txt' #训练集路径
        self.validset_path = None #验证集路径
        self.testset_path = None #测试集路径
        
        self.label_len = 2 #标签长度       
        self.use_tokenizer = False #使用tokenizer处理文本词嵌入
        self.categorical_crossentropy=False #使用多元交叉熵       
        self.stop_word_path = '' #停用词表
        
        self.learning_rate = 0.001 #学习率
        self.epoches = 50 #训练epoch
        self.early_stop_patience = 5 #早停忍耐
        self.epochs_per_cycle = 3 # 学习率周期epoch
        self.multi_gpu_num = 1 # 使用GPU个数，1：单GPU ; >1 : 多GPU, 同时gpu_name设置多个
        self.optimizers = 'adam'#优化器
        self.use_class_weights = False#使用类权重平衡
        
        self.batchsize = 16 #训练batchsize
        self.model_name = 'textual-visual' #模型名称
        self.verbose = 1 # 打印频率
        self.model_summary = False # 是否打印模型结构

        
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Config, cls).__new__(cls)
            cls.init = False
        else:
            cls.init = True
        return cls.instance
    
    
class Dataset(object):
    '''
        数据处理与生成
        # Arguments：
            config: 设置
    '''
    
    def __init__(self, config):
        self.config = config # 模型设置
        self._trainset_path = config.trainset_path # 训练集路径
        self._validset_path = config.validset_path # 验证集路径
        self._testset_path = config.testset_path # 测试集路径
        self._stop_word_path = config.stop_word_path # 停用词路径
        self._embedding_source_path = config.content.embedding_source_path # 词嵌入路径
        self._sentence_len = config.content.sentence_len # 文本长度

        self.train_content_word, self.train_label = None, None    #训练集文本序列表示， 标签    
        self.valid_content_word, self.valid_label = None, None    # 验证集文本序列表示， 标签    
        self.test_content_word, self.test_label = None, None      #测试集文本序列表示， 标签      

        self.train_results = list()     #训练集图文名称 
        self.valid_results = list()     #训练集图文名称
        self.test_results = list()      #训练集图文名称 
                
        self.word_embedding = None #词嵌入向量
        
        self.oow_list = [] #词表外词
        self.stopWordDict=[] #停用词典

        self.use_tokenizer = bool(config.use_tokenizer) #是否使用tokenizer
        self.verbose = config.verbose #打印频次
    
    def get_stop_word_dict(self, filepath):
        """
        读取停用词，生成停用词典
        # Arguments：
            filepath: 停用词路径
        # Return：
            None
        """
        
        if not os.path.exists(filepath):
            return
        with open(filepath, encoding='utf-8') as fp:
            for line in fp.readlines():
                self.stopWordDict.append(line.strip('\n'))

    # def _read_data(self, filepath):
    #     """
    #     读取数据集
    #     # Arguments：
    #         filepath: 数据集路径
    #     # Return：
    #         word_contents：分词后文本
    #         labels：文本标签
    #         results：图片名称
    #     """
    #
    #     with open(filepath, 'rb') as fp:
    #         datas = pd.read_csv(fp, sep='\t', header=None)      #读取数据集
    #
    #     results = np.asarray(datas.iloc[:,0])           #提取图片名称
    #     labels = np.asarray(datas.iloc[:,-1])           #提取标签
    #     data = np.asarray(datas.iloc[:,1])              #提取文本
    #
    #     # widget = ['\r read_data: %s  '%filepath[0], pb.NumPercentage(), ' ', pb.Bar('#'), ' ',
    #     #           pb.Percentage(), ' ', pb.Timer(), ' ', pb.ETA(), ' ',
    #     #           pb.FileTransferSpeed('line')] #进度条设置
    #     # bar = pb.MyProgressBar(maxval=len(data), widgets=widget) #进度条初始化
    #     # if self.verbose != 0:
    #     #     bar.start()
    #     word_contents = []
    #     regular = r'[]!“”"$%&\'()*…+,./:;=#@?[\\^_`{|}~-]'
    #     for i in range(0, len(data)):
    #         if isinstance(data[i], str): # 判断是否有内容
    #             jieba_res = re.sub(regular, r' \g<0> ', data[i]).strip().split() # 分词
    #             new_jieba_res = []
    #             for word in jieba_res: # 去停用词
    #                 if word in self.stopWordDict or len(word.strip()) == 0:
    #                     continue
    #                 else:
    #                     new_jieba_res.append(word.lower())
    #         else:
    #             new_jieba_res = []
    #         word_contents.append(new_jieba_res)
    #
    #     #     if self.verbose != 0:
    #     #         bar.update(i + 1)
    #     #
    #     # if self.verbose != 0:
    #     #     bar.finish()
    #
    #     return word_contents, labels, results

    def _read_data(self, filepath):
        file_data = open(filepath, 'r', encoding='utf-8')
        txt_list = []
        label_list = []
        name_id_list = []
        for lines in file_data.readlines():
            line_list = lines.strip().split()
            name_id_list.append(line_list[0])
            label_list.append(line_list[-1])
            txt_data = line_list[1:len(line_list) - 1]
            word_list = []
            for word in txt_data:
                word_list.append(word.lower())
            txt_list.append(word_list)
        labels = np.asarray(label_list)
        name_id = np.asarray(name_id_list)

        return txt_list, labels, name_id
        
    def _get_word_embedding(self, words):
        '''
        获取词嵌入
        # Arguments：
            words: 数据集使用的所有词
        # Return：
            vocab_word：词典
            np.array(word_embedding)：词嵌入，向量
        '''
        vocab, embd = self._loadGloVe(self._embedding_source_path, words) #读取向量
        vocab_word = []
        word_embedding = []
        
        embedding_size = len(embd[-1]) #词向量维度
        # 添加 "pad" 和 "UNK", 
        vocab_word.append("pad")
        vocab_word.append("UNK")
        word_embedding.append(np.zeros(embedding_size))
        word_embedding.append(np.random.randn(embedding_size))
        i = 0
        for word in words:
            try:
                vector =  embd[vocab.index(word)]                       
                vocab_word.append(word) #将词添加到词表中
                word_embedding.append(vector) #将词向量添加到向量表中
            except:
#                print(word + "不存在于词向量中")
                self.oow_list.append(word) #未发现词
                i+=1

        print("not in words", i, '/', len(words))
        return vocab_word, np.array(word_embedding)

    def _rebuild_process(self, contents, sequence_length, some_to_index):
        """
        将数据集中的每条评论用index表示
        wordToIndex中“pad”对应的index为0
        # Arguments：
            contents: 文本
            sequence_length: 文本长度
            some_to_index: 文本-index对应表
        # Return：
            contents_vec：文本的index列表

        """
        contents_vec = np.zeros((sequence_length))
        sequence_len = sequence_length
        
        # 判断当前的序列是否小于定义的固定序列长度
        if len(contents) < sequence_length:
            sequence_len = len(contents)
            
        for i in range(sequence_len):
            if contents[i] in some_to_index:
                contents_vec[i] = some_to_index[contents[i]]
            else:
                contents_vec[i] = some_to_index["UNK"]
    
        return contents_vec
    
    def _words_to_index(self, contents, labels):
        """
        遍历所有的文本，将文本中的词转换成index表示
        # Arguments：
            contents: 文本
            labels: 标签
        # Return：
            word_contents：文本的index列表
            labels：标签

        """
        
        word_contents = []
        
        if self.use_tokenizer is False: #是否使用tokenizer
            # widget = ['\r loading words  :', pb.NumPercentage(), ' ', pb.Bar('#'), ' ',
            #           pb.Percentage(), ' ', pb.Timer(), ' ', pb.ETA(), ' ',
            #           pb.FileTransferSpeed('line')]
            # if self.verbose != 0:
            #     bar = pb.MyProgressBar(maxval=len(contents), widgets=widget).start() #初始化进度条
    
            for i in range(len(contents)): #遍历所有文本
                content_vec = self._rebuild_process(contents[i], self._sentence_len, self._word_to_index) # 将文本转换未index表示
                word_contents.append(content_vec) #添加到list中
            #     if self.verbose != 0:
            #         bar.update(i)
            #
            # if self.verbose != 0:
            #     bar.finish()
        else:
            sequences = self.word_tokenizer.texts_to_sequences(contents)
            word_contents = preprocessing.sequence.pad_sequences(sequences, maxlen=self._sentence_len, padding='post')

            
        word_contents = np.asarray(word_contents, dtype="int64") #将list转换文array
        
        return word_contents, labels
  
    def _loadGloVe(self, filename, words): 
        """
        加载词向量预训练模型GloVe模型
        # Arguments：
            filename: 词向量路径
            words: 数据集使用的所有词
        # Return：
            vocab：词表
            np.asarray(embd)：词表对应的词向量
        """
        
        vocab = []
        embd = []
        is_save = True 
        if os.path.exists(f'{filename}.{self.config.database_name}_bak'): #判断是否需要保存当前数据集对应词向量，以减少下次训练的词向量读取处理时间
            filename = f'{filename}.{self.config.database_name}_bak'
            is_save = False
        
        # widget = ['\r read glove: %s  '%filename, pb.NumPercentage(), ' ', pb.Bar('#'), ' ',
        #   pb.Percentage(), ' ', pb.Timer(), ' ', pb.ETA(), ' ',
        #   pb.FileTransferSpeed('line')] # 初始化进度条
        with open(filename,'r', encoding='utf-8') as file:
            lines = file.readlines()
            # bar = pb.MyProgressBar(maxval=len(lines), widgets=widget)
            # bar.start()
            i = 0
            # print('get lines')
            rows = list()
            words_set = set(words) # 将list转换为set，提高in处理效率
            for line in lines:
                row = line.strip().split(' ')
                if row[0] in words_set:
                    rows.append(line) #有用词向量
                    vocab.append(row[0]) #将有用词添加到词表中
                    embd.append(np.array(row[1:], dtype=np.float)) #将对应词向量添加到embed中
                # bar.update(i)
                i+=1
            # bar.finish()
            print('Loaded GloVe!')
        
        if is_save:
            with open(f'{filename}.{self.config.database_name}_bak', 'w', encoding='utf-8') as fp:#保存当前数据集要使用的词向量
                fp.writelines(rows)
            print('save GloVe success!')

        return vocab, np.asarray(embd)
    
    def _gen_vocabulary(self, train_contents_word, eval_contents_word, test_contents_word):
        """
        生成词向量和词汇-索引映射字典

        # Arguments：
            train_contents_word: 训练集词
            eval_contents_word: 验证集词
            test_contents_word：测试集词
        # Return：
            None
        """       
        
        #读取所有word
        all_words = []
        if train_contents_word is not None:
            all_words += [word for content in train_contents_word for word in content]
        if eval_contents_word is not None:
            all_words += [word for content in eval_contents_word for word in content]
        if test_contents_word is not None:
            all_words += [word for content in test_contents_word for word in content]
            
        
        if self.use_tokenizer is True:
            self.data_process_tokenizer(all_words)
        else:                
            word_count = Counter(all_words)  # 统计词频
            sort_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True) #根据词频排序
            
            words = [item[0] for item in sort_word_count if item[1] >= 1]   # 去除低频词
            print("all word:", len(words), " first 10 words: ", words[:10]) 
            vocab_word, word_embedding = self._get_word_embedding(words) #成词向量和词汇-索引映射字典
            self.word_embedding = word_embedding 
            
            self._word_to_index = dict(zip(vocab_word, list(range(len(vocab_word))))) #词汇-索引映射字典

    def data_process_tokenizer(self, words):
        """
        使用tokenize获取词向量和词汇-索引映射字典
        # Arguments：
            words: 所有词
        # Return：
            None

        """
        
        voc_size = len(set(words))
        
        tokenizer = preprocessing.text.Tokenizer(nb_words=None, lower=False)

        tokenizer.fit_on_texts(words)
        embedding_matrix = dict(zip(self._loadGloVe(self._embedding_source_path)))
        embedding_size = len(embedding_matrix.values[-1])
        embed_train_matrix = np.zeros((voc_size+1,embedding_size))
        unk_vector = np.random.randn(embedding_size)
        for w,i in tokenizer.word_index.items():
            try:
                embedding_vector=embedding_matrix[w]
                embed_train_matrix[i] = embedding_vector
            except:
                embed_train_matrix[i] = unk_vector
        self.word_embedding  =  embed_train_matrix
        self.word_tokenizer = tokenizer
      
    def data_gen(self):
        
        self.get_stop_word_dict(self._stop_word_path)
        if self._trainset_path is not None and os.path.exists(self._trainset_path):
            train_content, train_label, self.train_results = self._read_data(self._trainset_path) # 读取训练集数据
        else:
            print("no train path",  self._trainset_path)
            train_content, train_label = None, None
    
        if self._validset_path is not None and os.path.exists(self._validset_path):
            valid_content, valid_label, self.valid_results = self._read_data(self._validset_path)# 读取验证集数据
        else:
            valid_content, valid_label = None, None

        if self._testset_path is not None and os.path.exists(self._testset_path):
            test_content, test_label, self.test_results = self._read_data(self._testset_path)# 读取测试集数据
        else:
            test_content, test_label = None, None


        self._gen_vocabulary(train_content, valid_content, test_content) #通过所有的数据，获取改数据集需要的词向量和词汇-索引映射字典
            
        # 初始化训练集和测试集
        if train_content is not None:
            self.train_content_word, self.train_label = self._words_to_index(train_content, train_label)# 转换训练集数据
            self.train_token = train_content
        if valid_content is not None:
            self.valid_content_word, self.valid_label = self._words_to_index(valid_content, valid_label)# 转换验证集数据
            self.valid_token = valid_content
        if test_content is not None:
            self.test_content_word, self.test_label = self._words_to_index(test_content, test_label)# 转换测试集数据
            self.test_token = test_content


class LMDataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(len(self.indices)/self.batch_size))

    def __init__(self, config, corpus_id, texts, image_path, image_size, lables=None, batch_size=32, shuffle=True, catorage='train'):
        """Compiles a Language Model RNN based on the given parameters
        :param image_path: dirname of image
        :param image_size: reshape size of images 
        :param config: config 
        :param corpus_id: array of id
        :param lables: array of label
        :param batch_size: number of steps at each batch
        :param shuffle: True if shuffle at the end of each epoch
        :param catorage: return label if catorage='train' at the end of each batcb
        :return: Nothing
        """
        self.config = config
        self.corpus_id = corpus_id
        self.texts = texts
        self.image_path = image_path
        self.image_size = image_size
        self.lables = lables
        self.batch_size = batch_size
        self.shuffle = shuffle  
        self.catorage = catorage
        self.indices = np.arange(len(self.corpus_id))

            
    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        # Read sentence (sample)
        token_ids = self.get_token_indices(sent_id=batch_indices)


        return token_ids

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)

    def get_token_indices(self, sent_id):
        """get batach data"""
        
        image_list = self.corpus_id[sent_id]
        res = []
        
        if 'textual' in self.config.model_name:
            texts = self.texts[sent_id]
            res.append(texts)
        if 'visual' in self.config.model_name:
            images = self.image_processer(image_list)
            res.append(images)

        if self.catorage == 'train' or self.catorage == 'valid':
            y_train = self.lables[sent_id]
            token_ids = [res, y_train]
        else:
            token_ids = res
        return token_ids

    def image_processer(self, image_list):
        """get image data"""
        
        image_data = list()
        for imageId in image_list:
            img = Image.open(os.path.join(self.image_path, str(imageId)+'.jpg'))
            trans_img = img.resize(self.image_size).convert('RGB')
            r,g,b = trans_img.split()
            rd, gd, bd = np.asarray(r)/255.0, np.asarray(g)/255.0, np.asarray(b)/255.0
            image_data.append(np.stack([rd,gd,bd], axis=-1))
        
        return np.asarray(image_data)
