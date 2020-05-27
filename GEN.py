from keras.layers import Dense,Input,Layer,Embedding, Lambda
from keras.models import Model
import tensorflow as tf
from keras import backend as K
from keras.activations import softmax
from keras.optimizers import Adam
import numpy as np
class Inner_(Layer):
    def __init__(self,emb_dim,user_num,item_num,param_0=None,param_1=None,**kwargs):
        super(Inner_,self).__init__(**kwargs)
        self.u_emb = Embedding(user_num,emb_dim,weights=param_0)
        #self.i_emb = Embedding(item_num,emb_dim,weights=param_1)
        self.i_emb = self.add_weight(shape=(item_num,emb_dim),initializer="uniform",trainable=True)
        self.d_item_bias = self.add_weight(shape=(item_num,), initializer="zeros",trainable=True)

    def compute_output_shape(self, input_shape):
        return input_shape + (self.u_emb.output_shape[-1],)


    def call(self,inputs,**kwargs):
        user_index = inputs
        u_emb = self.u_emb(user_index)
        u_emb = K.reshape(u_emb,(1,-1))
        i_emb = self.i_emb
        i_bias = self.d_item_bias
        score =  K.sum(tf.multiply(u_emb,i_emb),1) + i_bias
        return score



class Get_index(Layer):
    def compute_output_shape(self, input_shape):
        return input_shape[-1]
    def call(self, inputs, **kwargs):
        assert len(inputs) == 2
        output, sample = inputs
        output = softmax(K.reshape(output,(1,-1)),-1)
        output = K.flatten(output)
        pred = K.gather(output,sample)
        return pred





class GEN():
    def __init__(self,emb_dim,user_num,item_num,learning_rate=0.0001,param_0 = None,param_1 = None,**kwargs):
        super(GEN,self).__init__(**kwargs)
        self.emb_dim = emb_dim
        self.user_num = user_num
        self.item_num = item_num
        # 单用户
        self.input_user = Input(batch_shape=(1,1),dtype="int32",name="input_0")
        self.input_sample = Input(batch_shape=(1,None),dtype="int32",name="input_1")
        self.inner_ = Inner_(emb_dim,user_num,item_num,param_0=param_0,param_1=param_1)
        self.output_ = self.inner_(self.input_user)
        self.output =  Get_index()([self.output_,self.input_sample])
        self.model = Model([self.input_user,self.input_sample],self.output)
        self.model.compile(loss=self.loss(),optimizer=Adam(learning_rate=learning_rate),metrics=['accuracy'])


    def loss(self):
        def _loss(y_true,y_pred):
            """
            :param y_true:  reward
            :param y_pred:  计算的结果
            :return:
            """
            return - K.mean(K.log(y_pred)*y_true)
        return _loss


    def all_logits(self,user_id):
        functor = K.function([self.model.layers[0].input]+[K.learning_phase()],self.model.layers[1].output)
        logits = functor([user_id,0])
        return logits

    def all_rating(self,user_id):
        # 仅支持一个user_id传入  这里可以更改
        G_emb = self.inner_.u_emb.weights[0][user_id].numpy()
        I_emb = self.inner_.i_emb.numpy()
        I_bias = self.inner_.d_item_bias.numpy()
        G_emb = np.reshape(G_emb,(-1,self.emb_dim))
        I_bias = np.reshape(I_bias,(-1,self.item_num))
        rating =np.matmul(G_emb,np.transpose(I_emb)) + I_bias
        return rating


    def train(self,pre_data,pre_labe):
        self.model.train_on_batch(pre_data,pre_labe)



