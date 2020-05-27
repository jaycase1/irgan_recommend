import os
import numpy as np
import pickle as pkl
from GEN import GEN
from DIS import DIS
import utils as ut
import sys

EMB_DIM = 5
USER_NUM =943
ITEM_NUM = 1693
batch_size = 32
INIT_DELTA =0.05

all_items = set(range(ITEM_NUM))
workdir = 'ml-100k/'
DIS_TRAIN_FILE = workdir + 'dis-train.txt'

user_pos_train = {}
with open(workdir + 'movielens-100k-train.txt')as fin:
    # 仅对用户添加好评的商品 train文件
    for line in fin:
        line = line.split()
        uid = int(line[0])
        iid = int(line[1])
        r = float(line[2])
        if r > 3.99:
            if uid in user_pos_train:
                user_pos_train[uid].append(iid)
            else:
                user_pos_train[uid] = [iid]

user_pos_test = {}
with open(workdir + 'movielens-100k-test.txt')as fin:
    for line in fin:
        line = line.split()
        uid = int(line[0])
        iid = int(line[1])
        r = float(line[2])
        if r > 3.99:
            if uid in user_pos_test:
                user_pos_test[uid].append(iid)
            else:
                user_pos_test[uid] = [iid]

# test数据集  仅取测试的user范围是0-461  train数据集 取train的user范围是0-942


generator = GEN(EMB_DIM,USER_NUM,ITEM_NUM)
discriminator = DIS(EMB_DIM,USER_NUM,ITEM_NUM)


lamda = 0.1 / batch_size

def generate_for_d(filename):
    data = []
    for u in user_pos_train:
        pos = user_pos_train[u]

        # rating (1,1693)
        rating = generator.all_rating(u)
        rating = rating * 5
        exp_rating = np.exp(rating)
        prob = exp_rating / np.sum(exp_rating)

        neg = np.random.choice(np.arange(ITEM_NUM),size=len(pos))
        for i in range(len(pos)):
            data.append(str(u) + '\t' + str(pos[i]) + '\t' + str(neg[i]))

    with open(filename,"w") as fout:
        fout.write('\n'.join(data))


def all_rating_in_batch(model,user_batchs):
    user_batch_rating = []
    for user_id in user_batchs:
        rating = model.all_rating(user_id)
        user_batch_rating.append(rating)
    return np.array(user_batch_rating)

def get_all_content(users):
    test_item_set = set()
    for user in users:
        # 这里user是test集里面某一个user
        # 这里是将批测试用户 在训练集中出现好评的商品剔除掉
        test_item_set.union(set(user_pos_train[user]))
    return test_item_set

def get_all_rating_oneUser(ratings,index):
    rating_oneUser = []
    for rating in ratings:
        rating_oneUser.append(rating[:,index].tolist())
    return rating_oneUser


def simple_test_one_user(x):
    '''
    only used in test
    '''
    x = list(x)
    rating = [i[0] for i in x]
    u = [i[1] for i in x]
    test_items = list(all_items - get_all_content(u))
    item_score = []
    for i in test_items:
        # 这里取的整个user_batch 对这个item取值的概率
        item_score.append(get_all_rating_oneUser(rating,i))
    # item_score 里面存储的是1693 个测试物品  每个测试物品包括batch_size各user的概率
    item_sort = [x[0] for x in item_score]
    item_score = np.array(item_score).reshape((len(x),-1))
    r = []
    for user in u:
        # 测试标签集
        u_r = []
        for i in item_sort:
            if i in user_pos_test[user]:
                u_r.append(1)
            else:
                u_r.append(0)
        r.append(u_r)
    r = np.array(r)
    # item_score 和 r 为 test_batch 的 y_pred 和 y_true  如何确定test的验证  后续补充







def simple_test(model,batch_size):
    '''
    :param model:   generative model
    :param batch_size:  test batch size
    :return:
    '''
    result = np.array([0.]*6)
    test_users = list(user_pos_test.keys())
    test_user_num = len(test_users)
    index = 0
    while True:
        if index >= test_user_num:
            break
        # never overstep the bound of arrays
        # 列表如果越界的话会取到最后一个元素
        user_batch  = test_users[index:index+batch_size]
        index += batch_size
        # 这里取到的是users 对 电影的喜爱概率
        user_batch_rating = all_rating_in_batch(model,user_batch)
        user_batch_rating_uid = zip(user_batch_rating,user_batch)
        batch_result = simple_test_one_user(user_batch_rating_uid)



def main():
    dis_log = open(workdir + 'dis_log.txt', 'w')
    gen_log = open(workdir + 'gen_log.txt', 'w')
    print("starting adversarial training")
    for epoch in range(100):
        print("Training D")
        for d_epoch in range(30):
            if d_epoch % 5 == 0:
                generate_for_d(DIS_TRAIN_FILE)
                train_size = ut.file_len(DIS_TRAIN_FILE)
            index = 1
            while True:
                if index > train_size:
                    break
                if index + batch_size <= train_size + 1:
                    input_user, input_item, input_label = ut.get_batch_data(DIS_TRAIN_FILE, index, batch_size)
                else:
                    input_user, input_item, input_label = ut.get_batch_data(DIS_TRAIN_FILE, index,
                                                                            train_size - index + 1)
                index += batch_size
                discriminator.train([input_user, input_item], input_label)
            #print("\r[D Epoch %d/%d] [loss: %f]" % (d_epoch, 100, discriminator.model.loss))
        for g_epoch in range(50):
            print("Training G")
            for u in user_pos_train:
                sample_lambda = 0.2
                pos = user_pos_train[u]
                rating = generator.all_logits(u)
                # print(rating,type(rating),rating.shape)
                exp_rating = np.exp(rating)
                prob = exp_rating / np.sum(exp_rating)
                pn = (1 - sample_lambda) * prob
                pn[pos] += sample_lambda * 1.0 / len(pos)
                sample = np.random.choice(np.arange(ITEM_NUM),2*len(pos),p=pn)
                reward = discriminator.get_reward(u,sample)
                reward = reward  * prob[sample] / pn[sample]
                sample = sample[np.newaxis,:]
                reward = reward[np.newaxis,:]
                generator.train([np.array(u)[np.newaxis][np.newaxis,:],sample],reward)
            #print("\r[G Epoch %d/%d] [loss: %f]" % (g_epoch, 50, generator.model.loss))



if __name__ == '__main__':
    main()

