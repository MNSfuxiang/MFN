import keras
import keras.backend as K
import numpy as np
import tensorflow as tf


class POSITION_GET(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(POSITION_GET, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        shape = inputs.get_shape().as_list()
        position_vec = []
        for i in range(shape[1]):
            i_postion = np.arange(0, 1, shape[1] + 1)
            # i_postion = i_postion / max(i_postion)
            i_postion = np.asarray(i_postion)
            position_vec.append(i_postion)
        position_vec = np.asarray(position_vec)
        position_tensor = tf.constant(position_vec, dtype=tf.float32)
        out_tensor = inputs+position_tensor
        return out_tensor


class IMG_ADD(keras.layers.Layer):
    def __init__(self, add_step, **kwargs):
        self.add_step = add_step
        super(IMG_ADD, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W_add = self.add_weight(shape=(input_shape[1], self.add_step), initializer='he_normal',
                                     trainable=True, name='img_add_step_W')
        super(IMG_ADD, self).build(input_shape)

    def call(self, inputs, **kwargs):
        img_T = K.permute_dimensions(inputs, [0, 2, 1])
        img_add = K.dot(img_T, self.W_add)
        img_out = K.permute_dimensions(img_add, [0, 2, 1])
        img_out = K.relu(img_out)
        return img_out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.add_step, input_shape[-1])


class MMA_ATT(keras.layers.Layer):
    def __init__(self, out_dim, **kwargs):
        self.out_dim = out_dim
        super(MMA_ATT, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W_Q_t = self.add_weight(shape=(input_shape[0][-1], self.out_dim), initializer='he_normal',
                                     trainable=True, name='weight_Q_txt')
        self.W_Q_i = self.add_weight(shape=(input_shape[1][-1], self.out_dim), initializer='he_normal',
                                     trainable=True, name='weight_Q_img')
        self.W_K_t = self.add_weight(shape=(input_shape[0][-1], self.out_dim), initializer='he_normal',
                                     trainable=True, name='weight_K_txt')
        self.W_K_i = self.add_weight(shape=(input_shape[1][-1], self.out_dim), initializer='he_normal',
                                     trainable=True, name='weight_K_img')
        self.W_V_t = self.add_weight(shape=(input_shape[0][-1], self.out_dim), initializer='he_normal',
                                     trainable=True, name='weight_V_txt')
        self.W_V_i = self.add_weight(shape=(input_shape[1][-1], self.out_dim), initializer='he_normal',
                                     trainable=True, name='weight_V_img')
        super(MMA_ATT, self).build(input_shape)

    def call(self, inputs, **kwargs):
        Q_t = K.dot(inputs[0], self.W_Q_t)
        Q_i = K.dot(inputs[1], self.W_Q_i)
        K_t = K.dot(inputs[0], self.W_K_t)
        K_i = K.dot(inputs[1], self.W_K_i)
        V_t = K.dot(inputs[0], self.W_V_t)
        V_i = K.dot(inputs[1], self.W_V_i)
        Q_C = K.concatenate([Q_t, Q_i], axis=-1)
        K_C = K.concatenate([K_t, K_i], axis=-1)
        V_C = K.concatenate([V_t, V_i], axis=-1)
        K_C_T = K.permute_dimensions(K_C, [0, 2, 1])
        d_k = self.out_dim**0.5
        K_Q = K.batch_dot(Q_C, K_C_T)
        A_C = K.softmax(K_Q/d_k)
        head_att = K.batch_dot(A_C, V_C)
        return head_att

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim*2)


class TMA_ATT(keras.layers.Layer):
    def __init__(self, out_dim, **kwargs):
        self.out_dim = out_dim
        super(TMA_ATT, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W_Q = self.add_weight(shape=(input_shape[-1], self.out_dim), initializer='he_normal',
                                   trainable=True, name='TMA_Q_weight')
        self.W_K = self.add_weight(shape=(input_shape[-1], self.out_dim), initializer='he_normal',
                                   trainable=True, name='TMA_K_weight')
        self.W_V = self.add_weight(shape=(input_shape[-1], self.out_dim), initializer='he_normal',
                                   trainable=True, name='TMA_V_weight')
        super(TMA_ATT, self).build(input_shape)

    def call(self, inputs, **kwargs):
        Q_C = K.dot(inputs, self.W_Q)
        K_C = K.dot(inputs, self.W_K)
        V_C = K.dot(inputs, self.W_V)
        K_C_T = K.permute_dimensions(K_C, [0, 2, 1])
        d_k = self.out_dim**0.5
        KQ = K.batch_dot(Q_C, K_C_T)
        A_C = K.softmax(KQ/d_k)
        head_att = K.batch_dot(A_C, V_C)
        return head_att

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.out_dim)


class MULTI_HEAD(keras.layers.Layer):
    def __init__(self, out_dim, **kwargs):
        self.out_dim = out_dim
        super(MULTI_HEAD, self).__init__(**kwargs)

    def build(self, input_shape):
        self.M_W = self.add_weight(shape=(input_shape[-1], self.out_dim), initializer='he_normal',
                                   trainable=True, name='multi_head_weight')
        super(MULTI_HEAD, self).build(input_shape)

    def call(self, inputs, **kwargs):
        multi_att = K.dot(inputs, self.M_W)
        return multi_att

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.out_dim)
