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








        