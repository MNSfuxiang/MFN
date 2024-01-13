仅用于科研与学术
请勿随意传播
"""
import tensorflow as tf
import keras
import sklearn.metrics as sm
import numpy as np
from data_config_preprocess import Config
import attention_method as am
from keras.applications.resnet50 import ResNet50
# from doc2vec import AMGN, VSCN

def ContentNet(config, word_embedding):
    input_content = keras.layers.Input(batch_shape=(None, config.content.sentence_len,),  dtype='int64', name="input_content")
    with tf.name_scope('word_embbeding'):
        embedded_words = keras.layers.Embedding(input_dim=len(word_embedding),output_dim=len(word_embedding[-1]),
                                                        input_length=config.content.sentence_len, weights=[word_embedding],
                                                        mask_zero=False, trainable=False, name='word_embbeding')(input_content)

        if config.model.spa_drop_rate != 0.0:
            embedded_words = keras.layers.SpatialDropout1D(rate=config.model.spa_drop_rate)(embedded_words)
    

        with tf.name_scope('word_cnn_bilstm_att'):

            shap = embedded_words.get_shape().as_list()
            output_list=[embedded_words] 
            for i in range(1, config.model.multi_channel_num+1):
                output_1 = keras.layers.Conv1D(filters=shap[-1],kernel_size=3, strides=1, padding='same',use_bias=True,
                                                kernel_initializer='he_normal', bias_initializer ='zeros',
                                                activation='relu',dilation_rate=i, name='conv_rate_%d'%i)(embedded_words)
                output_list.append(output_1)
            
            for i in range(len(output_list)):
                output_1 = output_list[i]
                # output_1 = keras.layers.Bidirectional(keras.layers.CuDNNLSTM(units=output_1.get_shape().as_list()[-1], return_sequences=True),
                #                       merge_mode='concat')(output_1)
                output_1 = keras.layers.Bidirectional(
                    keras.layers.LSTM(units=output_1.get_shape().as_list()[-1], return_sequences=True),
                    merge_mode='concat')(output_1)
                output_list[i] = am.SelfAtt(output_1.get_shape().as_list()[-1])(output_1)

            output_ = keras.layers.Add(name='lstm_content_Add')(output_list)


        with tf.name_scope("Attention"):
            outp_list = []
            shap = output_.get_shape().as_list()
            for i in range(config.model.global_att_num):
                output_1 =  am.Scaled_Dot_Product_Att(shap[-1]//config.model.global_att_num, name='multi_att_%d'%i)(output_)
                outp_list.append(output_1)
                
            output_ = keras.layers.Concatenate(axis=-1)(outp_list)
            output_ = keras.layers.Dense(units=shap[-1])(output_)
            output_ = keras.layers.Flatten()(output_)
            output_ = keras.layers.Dense(units=1024, activation=None)(output_)
            model = keras.Model(inputs=[input_content], outputs=[output_], name="ContentNet")
#            model.summary()
            return model

class Textual_VISUAL(object):
    def __init__(self, config, word_embedding):
        self.input_pic = keras.layers.Input(batch_shape=(None, config.pic.width, config.pic.height, 3), name="input_pic")
        self.input_content = keras.layers.Input(batch_shape=(None, config.content.sentence_len,),  dtype='int64', name="input_content")


        with tf.name_scope('picNet'):
            pic_model = ResNet50(include_top=False, pooling='avg', weights=r'./pre_training_model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', input_shape=(config.pic.width, config.pic.height, 3))
            output_pic_tmp = pic_model(self.input_pic)
            output_pic_tmp = keras.layers.Dense(units=1024, activation=None)(output_pic_tmp) 
            
        with tf.name_scope('contentNet'):
            text_model = ContentNet(config, word_embedding)
            output_content_tmp = text_model(self.input_content)    
        with tf.name_scope('interrupt_layers'):
            output_content_list = []
            output_pic_list = []
            hidden_size_list = [32, 64, 128, 256]#[32][32,64][32,64, 128][32,64,128,256][32,64,128,256, 512]-->效果最好的通道数后再使每个相同比如[32,32,32,32,32]
            for hidden_size in hidden_size_list:
                output_content_t, output_pic_t = am.InteractAtt(attention_hidden=hidden_size)([output_content_tmp, output_pic_tmp])
                output_content_list.append(output_content_t)
                output_pic_list.append(output_pic_t)
            output_content = keras.layers.Concatenate()(output_content_list)
#            output_content = keras.layers.Dropout(config.model.dropout_rate)(output_content)
            output_content = keras.layers.Dense(units=512, use_bias=False, activation=None)(output_content)
#            output_content = keras.layers.BatchNormalization()(output_content)
#            output_content = keras.layers.Activation('tanh')(output_content)
#            
            output_pic = keras.layers.Concatenate()(output_pic_list)
#            output_pic = keras.layers.Dropout(config.model.dropout_rate)(output_pic)
            output_pic = keras.layers.Dense(units=512, use_bias=False, activation=None)(output_pic)
#            output_pic = keras.layers.BatchNormalization()(output_pic)
#            output_pic = keras.layers.Activation('tanh')(output_pic)
            
            output_ = keras.layers.Concatenate()([output_content, output_pic])
            output_ = keras.layers.Dropout(config.model.dropout_rate)(output_)
            output_ = keras.layers.Dense(units=256, activation='tanh')(output_) 
            output_ = keras.layers.Dropout(config.model.dropout_rate)(output_)
            
            if config.categorical_crossentropy is False:
                output_ = keras.layers.Dense(units=1, activation='sigmoid', use_bias=False,  kernel_regularizer=keras.regularizers.l2(l=1e-7))(output_)#l=1e-7
            else:
                output_ = keras.layers.Dense(units=config.label_len, activation='softmax', use_bias=True, kernel_regularizer=keras.regularizers.l2(l=1e-7))(output_)
    
        self.model = keras.Model(inputs=[self.input_content, self.input_pic], outputs=[output_])


class Textual_VISUAL_CONCAT(object):
    def __init__(self, config, word_embedding):
        self.input_pic = keras.layers.Input(batch_shape=(None, config.pic.width, config.pic.height, 3), name="input_pic")
        self.input_content = keras.layers.Input(batch_shape=(None, config.content.sentence_len,),  dtype='int64', name="input_content")


        with tf.name_scope('picNet'):           
            pic_model = ResNet50(include_top=False, pooling='avg', weights=r'./pre_training_model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', input_shape=(config.pic.width, config.pic.height, 3))
            output_pic_tmp = pic_model(self.input_pic)
            output_pic_tmp = keras.layers.Dense(units=1024, activation=None)(output_pic_tmp) 
            
        with tf.name_scope('contentNet'):
            output_content_tmp = ContentNet(config, word_embedding)(self.input_content)
            
        with tf.name_scope('concat_layers'):

            output_ = keras.layers.Concatenate()([output_pic_tmp, output_content_tmp])
            output_ = keras.layers.Dropout(config.model.dropout_rate)(output_)
            output_ = keras.layers.Dense(units=256, activation='tanh')(output_) 
            output_ = keras.layers.Dropout(config.model.dropout_rate)(output_)
            if config.categorical_crossentropy is False:
                output_ = keras.layers.Dense(units=1, use_bias=False,activation='sigmoid')(output_)
            else:
                output_ = keras.layers.Dense(units=config.label_len, use_bias=True,activation='softmax')(output_)
            
            
        self.model = keras.Model(inputs=[self.input_content, self.input_pic], outputs=[output_])
        
  
class Textual_VISUAL_AVG(object):
    def __init__(self, config, word_embedding):
        self.input_pic = keras.layers.Input(batch_shape=(None, config.pic.width, config.pic.height, 3), name="input_pic")
        self.input_content = keras.layers.Input(batch_shape=(None, config.content.sentence_len,),  dtype='int64', name="input_content")


        with tf.name_scope('picNet'):           
            pic_model = ResNet50(include_top=False, pooling='avg', weights=r'./pre_training_model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', input_shape=(config.pic.width, config.pic.height, 3))
            output_pic_tmp = pic_model(self.input_pic)
            output_pic_tmp = keras.layers.Dense(units=1024, activation=None)(output_pic_tmp) 
            output_pic_tmp = keras.layers.Dropout(config.model.dropout_rate)(output_pic_tmp)
            output_pic_tmp = keras.layers.Dense(units=256, activation='tanh')(output_pic_tmp) 
            output_pic_tmp = keras.layers.Dropout(config.model.dropout_rate)(output_pic_tmp)
            if config.categorical_crossentropy is False:
                output_pic_tmp = keras.layers.Dense(units=1, activation='sigmoid')(output_pic_tmp)
            else:
                output_pic_tmp = keras.layers.Dense(units=config.label_len, activation='softmax')(output_pic_tmp)
            
        with tf.name_scope('contentNet'):
            output_content_tmp = ContentNet(config, word_embedding)(self.input_content)
            output_content_tmp = keras.layers.Dropout(config.model.dropout_rate)(output_content_tmp)
            output_content_tmp = keras.layers.Dense(units=256, activation='tanh')(output_content_tmp) 
            output_content_tmp = keras.layers.Dropout(config.model.dropout_rate)(output_content_tmp)
            if config.categorical_crossentropy is False:
                output_content_tmp = keras.layers.Dense(units=1, use_bias=False,activation='sigmoid')(output_content_tmp)
            else:
                output_content_tmp = keras.layers.Dense(units=config.label_len, use_bias=True,activation='softmax')(output_content_tmp)
        with tf.name_scope('concat_layers'):
            output_ = keras.layers.Average()([output_pic_tmp, output_content_tmp])                
            
        self.model = keras.Model(inputs=[self.input_content, self.input_pic], outputs=[output_])
  
class Textual(object):
    def __init__(self, config, word_embedding):
        self.input_content = keras.layers.Input(batch_shape=(None, config.content.sentence_len,),  dtype='int64', name="input_content")

        with tf.name_scope('contentNet'):
            output_ = ContentNet(config, word_embedding)(self.input_content)
            output_ = keras.layers.Dropout(config.model.dropout_rate)(output_)
            output_ = keras.layers.Dense(units=256, activation='tanh')(output_)
            output_ = keras.layers.Dropout(config.model.dropout_rate)(output_)
            if config.categorical_crossentropy is False:
                output_ = keras.layers.Dense(units=1, use_bias=False, activation='sigmoid')(output_)
            else:
                output_ = keras.layers.Dense(units=config.label_len, use_bias=False, activation='softmax')(output_)
            
        self.model = keras.Model(inputs=[self.input_content], outputs=[output_])


class VISUAL(object):
    def __init__(self, config):
        self.input_pic = keras.layers.Input(batch_shape=(None, config.pic.width, config.pic.height, 3), name="input_pic")


        with tf.name_scope('picNet'):
            pic_model = ResNet50(include_top=False, pooling='avg', weights=r'./pre_training_model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', input_shape=(config.pic.width, config.pic.height, 3))
#            self.input_pic = pic_model.input
            output_ = pic_model(self.input_pic)
            output_ = keras.layers.Dense(units=1024, activation=None)(output_) 
            output_ = keras.layers.Dropout(config.model.dropout_rate)(output_)
            output_ = keras.layers.Dense(units=256, activation='tanh')(output_)
            output_ = keras.layers.Dropout(config.model.dropout_rate)(output_)
            if config.categorical_crossentropy is False:
                output_ = keras.layers.Dense(units=1, use_bias=False, activation='sigmoid')(output_)
            else:
                output_ = keras.layers.Dense(units=config.label_len, use_bias=False, activation='softmax')(output_)               
                
        self.model = keras.Model(inputs=[self.input_pic], outputs=[output_])

class TextClassifier(object):
    
    def __init__(self, config, word_embedding=None, summary_enabled = True):
        
        self.config = config
        
        if config.model_name == 'textual-visual':
            self.model = Textual_VISUAL(config, word_embedding).model
        elif config.model_name == 'textual':
            self.model = Textual(config, word_embedding).model
        elif config.model_name == 'visual':
            self.model = VISUAL(config).model
        elif config.model_name == 'textual-visual-concat':
            self.model = Textual_VISUAL_CONCAT(config, word_embedding).model
        elif config.model_name == 'textual-visual-avg':
            self.model = Textual_VISUAL_AVG(config, word_embedding).model
        # elif config.model_name == 'textual-visual-amgn':
        #     self.model = AMGN(config, word_embedding).model
        # elif config.model_name == 'textual-visual-vscn':
        #     self.model = VSCN(config, word_embedding).model
        else:
            return
        
        if summary_enabled is True:
            self.model.summary()

#     def load_weight(self, model):
#         name = self.config.model_name
#         if name == 'textual-visual':
#             layer = model.get_layer('resnet50')
#             layer.load_weights(r'./pre_training_model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
#         elif name == 'visual':
#             layer = model.get_layer('resnet50')
#             layer.load_weights(r'./pre_training_model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
#         elif name == 'textual-visual-concat':
#             layer = model.get_layer('resnet50')
#             layer.load_weights(r'./pre_training_model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
#         elif name == 'textual-visual-avg':
#             layer = model.get_layer('resnet50')
#             layer.load_weights(r'./pre_training_model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
#         elif name == 'textual-visual-amgn':
# #            layer = model.get_layer('vgg16')
#             model.load_weights(r'./pre_training_model/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', by_name=True)
#         elif name == 'textual-visual-vscn':
# #            layer = model.get_layer('resnet101')
#             model.load_weights(r'./pre_training_model/resnet101_weights_tf_dim_ordering_tf_kernels_notop.h5', by_name=True)
#         else:
#             return
#
#
#         if self.config.database_name.lower() == 'vso':
#             model.load_weights(r'./model.pth', skip_mismatch=True, by_name=True)
#         print('load weight success')


    def gen_metrics(self, true_y, pred_y):
        """
        生成acc和auc值
        """
        if self.config.categorical_crossentropy is False:
            true_y = [[0,1] if i > 0.5 else [1,0] for i in true_y]
            pred_y = [[0,1] if i > 0.5 else [1,0] for i in pred_y]
        
        label_size = max(len(np.unique(np.argmax(pred_y, axis=-1))), len(np.unique(np.argmax(true_y, axis=-1))))
        if label_size < 3:
            average_type='binary'
        else:
            average_type='macro'
            
        print('use %s average'%average_type)

        accuracy = sm.accuracy_score(np.argmax(true_y,axis=-1), np.argmax(pred_y, axis=-1))
        precision = sm.precision_score(np.argmax(true_y,axis=-1), np.argmax(pred_y,axis=-1),average=average_type)
        recall = sm.recall_score(np.argmax(true_y,axis=-1), np.argmax(pred_y,axis=-1),average=average_type)
        f1_score = sm.f1_score(np.argmax(true_y,axis=-1), np.argmax(pred_y,axis=-1),average=average_type)
        return  round(precision, 4), round(recall, 4), round(f1_score, 4) ,round(accuracy, 4) #

    def get_confusion(self, true_y, pred_y, labels=None):
        if self.config.categorical_crossentropy is False:
            true_y = [[0,1] if i > 0.5 else [1,0] for i in true_y]
            pred_y = [[0,1] if i > 0.5 else [1,0] for i in pred_y]
        
        confusion = sm.confusion_matrix(np.argmax(true_y,axis=-1), np.argmax(pred_y,axis=-1), labels=labels)
        print(confusion)
 

if __name__ == "__main__":
    config = Config() 
    word_embedding = np.zeros([100,200])
    model = Textual_VISUAL(config, word_embedding).model
    model.summary()      
