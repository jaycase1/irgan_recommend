from keras.layers import Input,Embedding,Layer,Lambda
from keras.losses import binary_crossentropy
from keras import backend as K
import tensorflow as tf
from keras.activations import sigmoid
from keras.models import Model
from keras.optimizers import Adam
from keras import regularizers
import numpy as np
'''
DIS  功能满足
'''


class inner_Dis(Layer):
    def __init__(self,emb_dim,user_num,item_num):
        super(inner_Dis,self).__init__()
        self.emb_dim = emb_dim
        self.user_num = user_num
        self.item_num = item_num
        self.u_embedding = Embedding(user_num, emb_dim)
        self.i_embedding = Embedding(item_num, emb_dim)
        self.d_item_bias = self.add_weight(shape=(self.item_num,),initializer="zeros",regularizer=regularizers.l2(0.1))


    def compute_output_shape(self, input_shape):
        return input_shape[0]


    def call(self, inputs, **kwargs):
        assert len(inputs) == 2
        input_user, input_item = inputs
        u_emb = self.u_embedding(input_user)
        i_emb = self.i_embedding(input_item)
        score = K.sum(tf.multiply(u_emb, i_emb), 1) + K.gather(self.d_item_bias,input_item)
        return score





class DIS():
    def __init__(self,emb_dim,user_num,item_num,learning_rate=0.001,decay_rate=0.01):
        self.emb_dim = emb_dim
        self.user_num = user_num
        self.item_num = item_num
        self.lr = learning_rate
        self.decay_rate = decay_rate
        # self.input_user, self.input_item, self.label 均为(1,batchsize)
        self.input_user = Input(shape=(1,),dtype="int32",name="input_0")
        self.input_item = Input(shape=(1,),dtype="int32",name="input_1")
        self.inner_ = inner_Dis(self.emb_dim,self.user_num,self.item_num)
        self.model = Model([self.input_user,self.input_item],self.inner_([self.input_user,self.input_item]))
        self.model.compile(loss="binary_crossentropy",optimizer=Adam(self.lr))

    def train(self,pred_data,pred_data_label):
        # pred_data_label 这里传入的是self.label
        self.model.train_on_batch(pred_data,pred_data_label)

    def get_reward(self,input_user,input_item):
        input_item = list(input_item)
        G_emb = self.inner_.u_embedding.weights[0][input_user].numpy()
        I_emb = self.inner_.i_embedding.weights[0].numpy()[input_item]
        I_bias = self.inner_.d_item_bias.numpy()[input_item]
        reward = np.sum(np.multiply(G_emb,I_emb),1) + I_bias
        return 2 * sigmoid(reward) - 1








