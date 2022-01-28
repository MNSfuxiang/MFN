import tensorflow as tf
import keras
import sklearn.metrics as sm
import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
import self_defined as sd


# <---------------图像情感分析模型----------------->
class Single_visual_model(object):
    def __init__(self):
        self.img_inputs = keras.layers.Input(batch_shape=(None, 224, 224, 3,), dtype='float32', name='input_img')

        with tf.name_scope('Only_image'):
            pre_model = ResNet50(include_top = False, pooling='avg',
                                 weights=r'./pre_train_model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                 input_shape = (224,224,3))
            out_feature = pre_model(self.img_inputs)
            out_predict = keras.layers.Dense(units=2, activation='softmax', use_bias=False,
                                             name='out_predict_layer')(out_feature)

        self.model = keras.Model(inputs=[self.img_inputs], outputs=[out_predict])


class Single_txtual_model(object):
    def __init__(self, embedding):
        self.txt_inputs = keras.layers.Input(batch_shape=(None, 36, ), dtype='float32', name='input_txt')

        with tf.name_scope('Only_txt'):
            self.in_embedding = keras.layers.Embedding(input_dim=len(embedding), output_dim=len(embedding[-1]),
                                                       weights=[embedding], mask_zero=False, trainable=False,
                                                       name='embedding')(self.txt_inputs)
            txt_flatten = keras.layers.Flatten(name='txt_flatten_layer')(self.in_embedding)
            out_predict = keras.layers.Dense(units=2, use_bias=False, activation='softmax',
                                             name='out_predict_layer')(txt_flatten)
        self.model = keras.Model(inputs=[self.txt_inputs], outputs=[out_predict])


class Early_fusion(object):
    def __init__(self, embedding):
        self.txt_inputs = keras.layers.Input(batch_shape=(None, 36, ), dtype='float32', name='input_txt')
        self.img_inputs = keras.layers.Input(batch_shape=(None, 224, 224, 3,), dtype='float32', name='input_img')
        with tf.name_scope('img_feature'):
            pre_model = ResNet50(include_top=False, pooling='avg',
                                 weights=r'./pre_train_model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                 input_shape=(224, 224, 3))
            out_feature = pre_model(self.img_inputs)
        with tf.name_scope('txt_feature'):
            self.in_embedding = keras.layers.Embedding(input_dim=len(embedding), output_dim=len(embedding[-1]),
                                                       weights=[embedding], mask_zero=False, trainable=False,
                                                       name='embedding')(self.txt_inputs)
            txt_flatten = keras.layers.Flatten(name='txt_flatten_layer')(self.in_embedding)
        with tf.name_scope('feature_concat'):
            img_txt_concat = keras.layers.concatenate(inputs=[out_feature, txt_flatten], axis=-1, name = 'img_txt_concat')
            out_predict = keras.layers.Dense(units=2, use_bias=False, activation='softmax',
                                             name='out_predict_layer')(img_txt_concat)
        self.model = keras.Model(inputs=[self.txt_inputs, self.img_inputs], outputs=[out_predict])


class Late_fusion(object):
    def __init__(self, embedding):
        self.txt_inputs = keras.layers.Input(batch_shape=(None, 36,), dtype='float32', name='input_txt')
        self.img_inputs = keras.layers.Input(batch_shape=(None, 224, 224, 3,), dtype='float32', name='input_img')

        with tf.name_scope('txt_score'):
            self.in_embedding = keras.layers.Embedding(input_dim=len(embedding), output_dim=len(embedding[-1]),
                                                       weights=[embedding], mask_zero=False, trainable=False,
                                                       name='embedding')(self.txt_inputs)
            txt_flatten = keras.layers.Flatten(name='txt_flatten_layer')(self.in_embedding)
            txt_out_predict = keras.layers.Dense(units=2, use_bias=False, activation='softmax',
                                             name='txt_out_predict_layer')(txt_flatten)
        with tf.name_scope('img_score'):
            pre_model = ResNet50(include_top=False, pooling='avg',
                                 weights=r'./pre_train_model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                 input_shape=(224, 224, 3))
            out_feature = pre_model(self.img_inputs)
            img_out_predict = keras.layers.Dense(units=2, activation='softmax', use_bias=False,
                                             name='img_out_predict_layer')(out_feature)
        with tf.name_scope('final_score'):
            final_predict = keras.layers.Average()([txt_out_predict, img_out_predict])

        self.model = keras.Model(inputs=[self.txt_inputs, self.img_inputs], outputs=[final_predict])
        
class DMAF(object):
    def __init__(self, embedding):
        self.txt_inputs = keras.layers.Input(batch_shape=(None, 36,), dtype='float32', name='input_txt')
        self.img_inputs = keras.layers.Input(batch_shape=(None, 224, 224, 3,), dtype='float32', name='input_img')
        with tf.name_scope('txt_classifer'):
            self.in_embedding = keras.layers.Embedding(input_dim=len(embedding), output_dim=len(embedding[-1]),
                                                       weights=[embedding], mask_zero=False, trainable=False,
                                                       name='embedding')(self.txt_inputs)

            txt_LSTM = keras.layers.LSTM(units=256, return_sequences=True, name='txt_LSTM_layer')(self.in_embedding)
            txt_seman_att = sd.DMAF_semantic_att(name='semantic_att')(txt_LSTM)
            txt_flatten = keras.layers.Flatten(name='txt_flatten_layer')(txt_seman_att)
            txt_FC_1 = keras.layers.Dense(units=1024, use_bias=False, bias_initializer='glorot_normal',
                                          activation='tanh', name='fc_layer_1')(txt_flatten)
            txt_FC_1 = keras.layers.Dropout(rate=0.5)(txt_FC_1)
            txt_FC_2 = keras.layers.Dense(units=512, use_bias=True, bias_initializer='glorot_normal',
                                          activation='tanh', name='fc_layer_2')(txt_FC_1)
            txt_FC_2 = keras.layers.Dropout(rate=0.5)(txt_FC_2)
            txt_FC_3 = keras.layers.Dense(units=256, use_bias=True, bias_initializer='glorot_normal',
                                          activation='tanh', name='fc_layer_3')(txt_FC_2)
            txt_classifer = keras.layers.Dense(units=2, use_bias=False, activation='softmax', name='txt_classifer')(txt_FC_3)

        with tf.name_scope('img_classifer'):
            pre_img = VGG19(include_top = False,
                            weights = r'./pre_train_model/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
                            input_shape = (224,224,3))
            pre_img_feature = pre_img(self.img_inputs)
            img_shape = pre_img_feature.get_shape().as_list()
            img_reshape = keras.layers.Reshape((img_shape[1] * img_shape[2], img_shape[-1]),
                                                 name='img_reshape')(pre_img_feature)
            img_visual_att = sd.DMAF_visual_att(name='img_visual_att')(img_reshape)
            img_flatten = keras.layers.Flatten(name='img_flatten_layer')(img_visual_att)
            img_FC_1 = keras.layers.Dense(units=1024, use_bias=True, bias_initializer='glorot_normal',
                                          activation='tanh', name='img_FC_1')(img_flatten)
            img_FC_1 = keras.layers.Dropout(rate=0.5)(img_FC_1)
            img_FC_2 = keras.layers.Dense(units=512, use_bias=True, bias_initializer='glorot_normal',
                                          activation='tanh', name='img_FC_2')(img_FC_1)
            img_FC_2 = keras.layers.Dropout(rate=0.5)(img_FC_2)
            img_FC_3 = keras.layers.Dense(units=256, use_bias=True, bias_initializer='glorot_normal',
                                          activation='tanh', name='img_FC_3')(img_FC_2)
            img_classifer = keras.layers.Dense(units=2, use_bias=False, activation='softmax', name='img_classifer')(img_FC_3)

        with tf.name_scope('multimodel_classifer'):
            M_IT_concat = keras.layers.concatenate(inputs=[txt_flatten, img_flatten], axis=-1, name = 'concat')
            # M_IT_concat = keras.layers.BatchNormalization()(M_IT_concat)
            M_FC_1 = keras.layers.Dense(units=1024, use_bias=True, bias_initializer='glorot_normal',
                                          activation='tanh', name='M_FC_1')(M_IT_concat)
            M_FC_1 = keras.layers.Dropout(rate=0.5)(M_FC_1)
            M_FC_2 = keras.layers.Dense(units=512, use_bias=True, bias_initializer='glorot_normal',
                                        activation='tanh', name='M_FC_2')(M_FC_1)
            M_FC_2 = keras.layers.Dropout(rate=0.5)(M_FC_2)
            M_FC_3 = keras.layers.Dense(units=256, use_bias=True, bias_initializer='glorot_normal',
                                        activation='tanh', name='M_FC_3')(M_FC_2)
            M_classifer = keras.layers.Dense(units=2, use_bias=False, activation='softmax', name='M_classifer')(M_FC_3)

        with tf.name_scope('late_fusion'):
            out_score = sd.DMAF_late_fusion(name='late_fusion_layer')([M_classifer, txt_classifer, img_classifer])
        self.model = keras.Model(inputs=[self.txt_inputs, self.img_inputs], outputs=[out_score])

class AMGN(object):
    def __init__(self, embedding):
        self.txt_inputs = keras.layers.Input(batch_shape=(None, 36,), dtype='float32', name='input_txt')
        self.img_inputs = keras.layers.Input(batch_shape=(None, 224, 224, 3,), dtype='float32', name='input_img')

        with tf.name_scope('txt_pre'):
            self.in_embedding = keras.layers.Embedding(input_dim=len(embedding), output_dim=len(embedding[-1]),
                                                       weights=[embedding], mask_zero=False, trainable=False,
                                                       name='embedding')(self.txt_inputs)
        with tf.name_scope('img_pre'):
            pre_img = VGG19(include_top=False,
                            weights=r'./pre_train_model/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
                            input_shape=(224, 224, 3))
            pre_img_feature = pre_img(self.img_inputs)
            pre_img_shape = pre_img_feature.get_shape().as_list()
            img_feature_reshape = keras.layers.Reshape((pre_img_shape[1]*pre_img_shape[2],pre_img_shape[-1]),
                                                       name='img_feature_reshape')(pre_img_feature)
            semantic_img_f = sd.AMGN_SV_att(name='senmantic_visual_att')([img_feature_reshape, self.in_embedding])
            img_resahpe = keras.layers.Conv1D(filters=128, kernel_size=1, strides=1, padding='SAME', use_bias=True,
                                                kernel_initializer='he_normal', bias_initializer='zeros',
                                                activation='relu', name='img_resahpe_w_C')(semantic_img_f)
        with tf.name_scope('gated_LSTM'):
            img_txt_in = keras.layers.concatenate([self.in_embedding, img_resahpe], axis=-1)
            cell = sd.AMGN_Cell(units=512)
            multi_LSTM = keras.layers.RNN(cell=cell, trainable=True, return_sequences=True,
                                          name='modelity_gated_LSTM')(img_txt_in)
            multi_sel_att = sd.AMGN_self_att(name='multi_feature_att')(multi_LSTM)
            feature_flatten = keras.layers.Flatten(name='feature_flatten')(multi_sel_att)
            feature_flatten = keras.layers.BatchNormalization(name='feature_flatten_BN')(feature_flatten)
            feature_Fc_1 = keras.layers.Dense(units=1024, use_bias=True,
                                            activation='tanh', name='feature_Fc_1')(feature_flatten)
            feature_Fc_2 = keras.layers.Dense(units=512, use_bias=True,
                                              activation='tanh', name='feature_Fc_2')(feature_Fc_1)
            feature_Fc_3 = keras.layers.Dense(units=256, use_bias=True,
                                              activation='tanh', name='feature_Fc_3')(feature_Fc_2)
            out_predict = keras.layers.Dense(units=2, use_bias=False,
                                             activation='softmax', name='out_predict')(feature_Fc_3)

        self.model = keras.Model(inputs=[self.txt_inputs, self.img_inputs], outputs=[out_predict])





        
