import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import pickle

DIR = 'ori_data/data_4_L960'
cof_path = os.path.join(DIR, 'cof.pkl')
par_path = os.path.join(DIR, 'par.pkl')

# 读取第一个数据文件获取数据长度
first_file = os.path.join(DIR, 'data_s1.mat')
first_data = sio.loadmat(first_file)
data_length = first_data['data_s'].shape[0]
print(f'数据长度: {data_length}')

IMG_DIR = 'img/'
os.makedirs(IMG_DIR, exist_ok=True)

with open(cof_path, 'rb') as fo:
    cof = pickle.load(fo, encoding='bytes')
print(cof)

with open(par_path, 'rb') as fo:
    par = pickle.load(fo, encoding='bytes')
print('parameter length',len(par))

N = cof['N']
START = 0
END = data_length - 1
STEP = 100  # 每隔100个时间点绘制一次图

for i in range(N):
    File = os.path.join(DIR, f'data_s{i+1}.mat')
    data = sio.loadmat(File)
    data_s = data['data_s']
    
    # 按照步长绘制data_s的图
    for t in range(START, END + 1, STEP):
        plt.imshow(np.squeeze(data_s[t,]))
        plt.colorbar()
        plt.title(f's{t}')
        plt.savefig(f'{IMG_DIR}/data_s{i}_{t}.jpg')
        plt.close()

    File1 = os.path.join(DIR, f'data_t{i+1}.mat')
    data = sio.loadmat(File1)
    data_t = data['data_t']
    
    # 按照步长绘制data_t的图
    for t in range(START, END + 1, STEP):
        plt.plot(data_t[t,])
        plt.title(f't{t}')
        plt.savefig(f'{IMG_DIR}/data_t{i}_{t}.jpg')
        plt.close()

    print(f'{i}/{N}', 's', data_s.shape, 't', data_t.shape)
