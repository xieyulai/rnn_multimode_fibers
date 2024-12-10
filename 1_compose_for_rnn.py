import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import pickle
import cv2
import os



DATA_DIR='ori_data/data_10'
RNN_DIR='rnn_data/rnn_10_480'

DATA_DIR='../data/data_100_480'
RNN_DIR='rnn_data/rnn_100_480'

DATA_DIR='../data/data_4000_480'
RNN_DIR='rnn_data/rnn_4000_480'

if not os.path.exists(RNN_DIR):
    os.makedirs(RNN_DIR)

cof_path = f'{DATA_DIR}/cof.pkl'
par_path = f'{DATA_DIR}/par.pkl'

with open(cof_path, 'rb') as fo:
    cof = pickle.load(fo, encoding='bytes')
print(cof)

with open(par_path, 'rb') as fo:
    par = pickle.load(fo, encoding='bytes')
print('parameter length',len(par))

N=cof['N']

START = 0
END = 479


#SHORTEN = 4
SHORTEN = 1

BASE_LEN = END + 1
LEN = BASE_LEN // SHORTEN


S_LIST = []
T_LIST = []


for i in range(N):
    
    File=f'{DATA_DIR}/data_s{i+1}.mat'
    data=sio.loadmat(File)
    data_s = data['data_s']

    data_s = data_s[::SHORTEN]

    data_s_rnn_down = data_s 
    data_s_rnn_down = data_s_rnn_down.transpose(1,2,0)
    data_s_rnn_down = cv2.resize(data_s_rnn_down,(32,32))
    data_s_rnn_down = data_s_rnn_down.transpose(2,0,1)

    data_s_rnn = data_s_rnn_down.reshape((data_s_rnn_down.shape[0],-1))


    S_LIST.append(data_s_rnn)

    File1=f'{DATA_DIR}/data_t{i+1}.mat'
    data=sio.loadmat(File1)
    data_t = data['data_t']    
    data_t_rnn = data_t
    T_LIST.append(data_t_rnn)

    print(f'{i}/{N} s {data_s.shape} -> {data_s_rnn.shape} t: {data_t.shape} -> {data_t_rnn.shape}')

# 修改保存逻辑,分批保存数据
def save_batch_data(data, save_dir, prefix, batch_size=100):
    """分批保存数据到小文件
    Args:
        data: 要保存的数据数组
        save_dir: 保存目录
        prefix: 文件名前缀 (data_s 或 data_t)
        batch_size: 每个批次的大小
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    num_samples = len(data)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        batch_data = data[start_idx:end_idx]
        
        # 保存为mat文件
        filename = f'{save_dir}/{prefix}_{i}.mat'
        sio.savemat(filename, {prefix: batch_data})
        print(f'Saved {filename}, shape: {batch_data.shape}')

# 在主代码中使用
if __name__ == '__main__':
    # ... 前面的代码保持不变 ...
    
    # 替换原来的保存逻辑
    stack_s = np.stack(S_LIST)
    stack_s = stack_s.transpose(0,2,1)
    save_batch_data(stack_s, RNN_DIR, 'data_s', batch_size=100)
    
    stack_t = np.stack(T_LIST)
    stack_t = stack_t.transpose(0,2,1)
    save_batch_data(stack_t, RNN_DIR, 'data_t', batch_size=100)



