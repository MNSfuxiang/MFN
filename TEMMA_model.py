import keras
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
import numpy as np
import sklearn.metrics as sm
from .utils import POSITION_GET, IMG_ADD, MMA_ATT, TMA_ATT, MULTI_HEAD
from keras_layer_normalization import LayerNormalization

class TEMMA(object):
    def __init__(self, params, embedding):
        self.input_txt = keras.layers.Input(batch_shape=(None, params.txt_len, ), dtype='float32', name='in_txt')
        self.input_img = keras.layers.Input(batch_shape=(None, params.img_size, params.img_size, 3, ), name='in_img')

        with tf.name_scope('txt_Net'):
            self.embedding = keras.layers.Embedding(input_dim=len(embedding), output_dim=len(embedding[-1]),
                                                    weights=[embedding], mask_zero=False, trainable=True,
                                                    name='embedding')(self.input_txt)
            txt_temporal = keras.layers.Conv1D(filters=params.temp_filters, kernel_size=1, strides=1, padding='causal',
                                               use_bias=True, activation='elu', kernel_initializer='he_normal',
                                               bias_initializer='he_normal',name='txt_temporal_conv')(self.embedding)
            txt_add = POSITION_GET(name='txt_position')(txt_temporal)
            # txt_add = txt_temporal

        with tf.name_scope('img_Net'):
            resNet = ResNet50(include_top=False, pooling='avg', weights = params.resNet_path,
                              input_shape=(params.img_size, params.img_size, 3))
            get_img_vec = resNet(self.input_img)
            img_shape = get_img_vec.get_shape().as_list()
            img_vec = keras.layers.Reshape((1, img_shape[1]), name='img_reshape')(get_img_vec)
            img_vec = IMG_ADD(add_step=params.txt_len, name='img_add_step')(img_vec)
            img_temporal = keras.layers.Conv1D(filters=params.temp_filters, kernel_size=1, strides=1, padding='causal',
                                               use_bias=True, activation='elu', kernel_initializer='he_normal',
                                               bias_initializer='he_normal', name='img_temporal_conv')(img_vec)
            img_add = POSITION_GET(name='img_position')(img_temporal)
            # img_add = img_temporal

        with tf.name_scope('MMA'):
            M_heads= []
            for i in range(params.heads):
                head_att_M = MMA_ATT(out_dim=params.out_dim, name='MMA_head_'+str(i))([txt_add, img_add])
                M_heads.append(head_att_M)
            heads_att_M = keras.layers.Concatenate(axis=-1)(M_heads)
            mma_att = MULTI_HEAD(out_dim=params.temp_filters, name='multi_heads')(heads_att_M)
            txt_M_add = keras.layers.Add(name='MMA_txt_add')([txt_add, mma_att])
            img_M_add = keras.layers.Add(name='MMA_img_add')([img_add, mma_att])
            print(txt_M_add.get_shape().as_list())
            txt_MMA = LayerNormalization()(txt_M_add)
            img_MMA = LayerNormalization()(img_M_add)
            # txt_MMA = txt_M_add
            # img_MMA = img_M_add

        with tf.name_scope('TMA_txt'):
            T_heads_txt = []
            for i in range(params.heads):
                head_txt_T = TMA_ATT(out_dim=params.out_dim, name='TMA_T_head_'+str(i))(txt_MMA)
                T_heads_txt.append(head_txt_T)
            heads_att_t = keras.layers.Concatenate(axis=-1)(T_heads_txt)
            tma_att_t = MULTI_HEAD(out_dim=params.temp_filters, name='temporal_heads_txt')(heads_att_t)
            txt_T_add = keras.layers.Add(name='TMA_txt_add')([txt_MMA, tma_att_t])
            txt_TMA = LayerNormalization(name='txt_TMA_LN')(txt_T_add)
            # txt_TMA = txt_T_add

        with tf.name_scope('TMA_img'):
            T_heads_img = []
            for i in range(params.heads):
                head_img_T = TMA_ATT(out_dim=params.out_dim, name='TMA_I_head_'+str(i))(img_MMA)
                T_heads_img.append(head_img_T)
            heads_att_i = keras.layers.Concatenate(axis=-1)(T_heads_img)
            tma_att_i = MULTI_HEAD(out_dim=params.temp_filters, name='temporal_heads_img')(heads_att_i)
            img_T_add = keras.layers.Add(name='TMA_img_add')([img_MMA, tma_att_i])
            img_TMA = LayerNormalization(name='img_TMA_LN')(img_T_add)
            # img_TMA = img_T_add

        with tf.name_scope('hidden_fc'):
            txt_fc = keras.layers.Dense(units=params.fc, use_bias=True, bias_initializer='he_normal',
                                        activation='elu', name='txt_fc')(txt_TMA)
            txt_fc_add = keras.layers.Add()([txt_TMA, txt_fc])
            txt_fc_LN = LayerNormalization()(txt_fc_add)
            img_fc = keras.layers.Dense(units=params.fc, use_bias=True, bias_initializer='he_normal',
                                        activation='elu', name='img_fc')(img_TMA)
            img_fc_add = keras.layers.Add()([img_TMA, img_fc])
            img_fc_LN = LayerNormalization()(img_fc_add)

        fusion = keras.layers.Concatenate(axis=-1)([txt_fc_LN, img_fc_LN])
        flatten = keras.layers.Flatten()(fusion)
        hidden_layer = keras.layers.Dense(units=params.hidden, use_bias=True, bias_initializer='he_normal',
                                          activation='elu', name='hidden_dense')(flatten)
        out_predict = keras.layers.Dense(units=params.out, use_bias=True,
                                         activation='softmax', name='out_layer')(hidden_layer)

        self.model = keras.Model(inputs = [self.input_txt, self.input_img], outputs = [out_predict])


class CLASSIFER(object):
    def __init__(self, params, embedding = None, summery_enabled = True):
        self.model = TEMMA(params=params, embedding=embedding).model
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
