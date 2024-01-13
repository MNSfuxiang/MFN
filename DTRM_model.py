import tensorflow as tf
import keras
import sklearn.metrics as sm
import numpy as np
from keras.applications.resnet50 import ResNet50
from utils import SelfAtt, Scaled_Dot_Product_Att, PDG
from keras_layer_normalization import LayerNormalization

def TextNet(params, word_embedding):
    txt_input = keras.layers.Input(batch_shape=(None, params.txt_len, ), dtype='float32', name='in_txt')
    with tf.name_scope('txt_net'):
        embedding = keras.layers.Embedding(input_dim=len(word_embedding), output_dim=len(word_embedding[-1]),
                                                    weights=[word_embedding], mask_zero=False, trainable=False,
                                                    name='embedding')(txt_input)
        if params.spa_drop_rate != 0.0:
            embedding = keras.layers.SpatialDropout1D(rate = params.spa_drop_rate)(embedding)

        shape = embedding.get_shape().as_list()
        output_list = [embedding]
        for i in range(1, params.channel_num+1):
            out_CNN = keras.layers.Conv1D(filters=shape[-1], kernel_size=3, strides=1, padding='same',use_bias=True,
                                          kernel_initializer='he_normal', bias_initializer ='zeros',
                                          activation='relu',dilation_rate=i, name='conv_rate_'+str(i))(embedding)
            output_list.append(out_CNN)

        for i in range(len(output_list)):
            output_CNN = output_list[i]
            out_BL = keras.layers.Bidirectional(keras.layers.LSTM(units=output_CNN.get_shape().as_list()[-1],
                                                                       return_sequences=True),
                                                merge_mode='concat')(output_CNN)
            # out_BL = keras.layers.Bidirectional(keras.layers.LSTM(units=output_CNN.get_shape().as_list()[-1],
            #                                                            return_sequences=True),
            #                                     merge_mode='concat')(output_CNN)
            output_list[i] = SelfAtt(out_BL.get_shape().as_list()[-1])(out_BL)

        out_txt = keras.layers.Add(name='LSTM_txt_add')(output_list)

    with tf.name_scope('Attention'):
        out_list = []
        shape = out_txt.get_shape().as_list()
        for i in range(params.att_num):
            out_att = Scaled_Dot_Product_Att(shape[-1]//params.att_num, name='multi_att_'+str(i))(out_txt)
            out_list.append(out_att)

        att_txt = keras.layers.Concatenate(axis=-1)(out_list)
        att_txt = keras.layers.Dense(units=shape[-1])(att_txt)
        att_txt = keras.layers.Flatten()(att_txt)
        att_txt = keras.layers.Dense(units=1024, activation=None)(att_txt)
        model = keras.Model(inputs=[txt_input], outputs=[att_txt], name="TextNet")
        return model


class DTRM(object):
    def __init__(self, params, word_embedding):
        self.input_txt = keras.layers.Input(batch_shape=(None, params.txt_len, ), dtype='float32', name='in_txt')
        self.input_img = keras.layers.Input(batch_shape=(None, params.img_size, params.img_size, 3, ), name='in_img')

        with tf.name_scope('TextNet'):
            txt_model = TextNet(params, word_embedding)
            txt_feature = txt_model(self.input_txt)

        with tf.name_scope('ImgNet'):
            img_model = ResNet50(include_top=False, pooling='avg', weights = params.resNet_path,
                                 input_shape=(params.img_size, params.img_size, 3))
            img_feature = img_model(self.input_img)
            img_feature = keras.layers.Dense(units=1024, activation=None)(img_feature)

        with tf.name_scope('TRM'):
            M_out = PDG(dim=params.att_dim)([txt_feature, img_feature])
            multiadd_LN = LayerNormalization()(M_out)
            multihidden = keras.layers.Dense(units=params.att_dim, use_bias=True, bias_initializer='he_normal',
                                        activation='elu', name='img_fc')(multiadd_LN)
            hidden_add = keras.layers.Add()([multiadd_LN, multihidden])
            trans_out = LayerNormalization()(hidden_add)

        hidden_layer = keras.layers.Dense(units=params.hidden_dim, activation='relu', use_bias=True,
                                          bias_initializer='he_normal')(trans_out)
        predict_layer = keras.layers.Dense(units=params.out_dim, activation='softmax', use_bias=True,
                                           bias_initializer='he_normal')(hidden_layer)
        self.model = keras.Model(inputs=[self.input_txt, self.input_img], outputs=[predict_layer])


class CLASSIFER(object):
    def __init__(self, params, embedding = None, summery_enabled = True):
        self.model = DTRM(params=params, word_embedding=embedding).model
        if summery_enabled is True:
            self.model.summary()

    def get_metrics(self, true_label, predict_label):
        true_labels = []
        predict_labels = []
        for i in true_label:
            true_labels.append(i)
        for i in predict_label:
            predict_labels.append(i)
        label_size = max(len(np.unique(np.argmax(predict_label, axis=-1))),
                         len(np.unique(np.argmax(true_label, axis=-1))))
        if label_size < 3:
            average_type = 'binary'
        else:
            average_type = 'macro'
        print('use %s average' % average_type)
        accuracy = sm.accuracy_score(np.argmax(true_label, axis=-1), np.argmax(predict_label, axis=-1))
        prediction = sm.precision_score(np.argmax(true_label, axis=-1),
                                        np.argmax(predict_label, axis=-1), average=average_type)
        recall = sm.recall_score(np.argmax(true_label, axis=-1),
                                 np.argmax(predict_label, axis=-1), average=average_type)
        f1_score = sm.f1_score(np.argmax(true_label, axis=-1),
                               np.argmax(predict_label, axis=-1), average=average_type)
        return round(accuracy, 4), round(prediction, 4), round(recall, 4), round(f1_score, 4)
