import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import tqdm
import os



def load_batch_data(data_dir, prefix, start_idx=0, end_idx=None):
    """加载分批保存的数据
    Args:
        data_dir: 数据目录
        prefix: 文件前缀 (data_s 或 data_t)
        start_idx: 起始样本索引
        end_idx: 结束样本索引
    """
    data_list = []
    batch_idx = 0
    total_samples = 0
    
    while True:
        filename = f'{data_dir}/{prefix}_{batch_idx}.mat'
        if not os.path.exists(filename):
            break
            
        batch_data = sio.loadmat(filename)[prefix]
        data_list.append(batch_data)
        total_samples += len(batch_data)
        batch_idx += 1
    
    # 合并所有批次的数据
    all_data = np.concatenate(data_list, axis=0)
    
    # 如果指定了索引范围,则只返回该范围的数据
    if end_idx is not None:
        all_data = all_data[start_idx:end_idx]
    
    return all_data

def norm(train_dir, test_dir, ST,normalization='none'):
    # 加载训练数据
    data_train = load_batch_data(train_dir, f'data_{ST}')
    print("data_train loaded...", data_train.shape)
    
    # 加载测试数据
    data_test = load_batch_data(test_dir, f'data_{ST}')
    print("data_test loaded...", data_test.shape)
    
    # 归一化逻辑保持不变
    if normalization == 'max':
        m_max = np.max(np.fabs(data_train))
        print('max:', m_max)
        data_train = data_train/m_max
        data_test = data_test/m_max
    # ... 其他归一化选项保持不变 ...
    
    return data_train, data_test, m_max

import random
random.seed(78)
#count = 0
def generate_data(data_all, num_evo, steps, window_size, batchsize=1):
    while 1:
        count = random.randint(0, num_evo-1)
        if count > data_all.shape[0]: 
            count = data_all.shape[0]
            
        data = data_all[count]
        data = np.expand_dims(data, axis=0)
        
        evo_size = steps - window_size  # 修改这里
        num_samples = np.round(batchsize*evo_size).astype(int)
        X_data_series = np.zeros((num_samples, window_size, i_x))
        Y_data_series = np.zeros((num_samples, i_x))
        
        for evo in range(batchsize):
            evo_data = np.transpose(data[evo, :, :])
            for step in range(evo_size):
                input_data = evo_data[step:step + window_size, :]
                output_data = evo_data[step + window_size, :]
                series_idx = evo*evo_size + step
                X_data_series[series_idx, :, :] = input_data
                Y_data_series[series_idx, :] = output_data
                
        yield (X_data_series, Y_data_series)

def load_data(data, num_evo, steps, window_size):
    # Make the time series
    evo_size = steps - window_size  # 修改这里
    num_samples = np.round(num_evo*evo_size).astype(int)
    X_data_series = np.zeros((num_samples, window_size, i_x))
    Y_data_series = np.zeros((num_samples, i_x))

    for evo in range(num_evo):
        evo_data = np.transpose(data[evo, :, :])
        for step in range(evo_size):
            input_data = evo_data[step:step + window_size, :]
            output_data = evo_data[step + window_size, :]
            series_idx = evo*evo_size + step
            X_data_series[series_idx, :, :] = input_data
            Y_data_series[series_idx, :] = output_data

    return (X_data_series, Y_data_series)

def load_data_no_tile_1st(data, num_evo, steps, window_size):
    # Make the time series
    evo_size = steps - window_size  # 修改这里：可用步数需要减去窗口大小
    num_samples = np.round(num_evo*evo_size).astype(int)
    X_data_series = np.zeros((num_samples, window_size, i_x))
    Y_data_series = np.zeros((num_samples, i_x))

    # loop in each sample(evo)
    for evo in range(num_evo):
        evo_data = np.transpose(data[evo, :, :])  # 480,1024

        for step in range(evo_size):  # 现在step最大到 465 (当steps=480, window_size=15时)
            input_data = evo_data[step:step + window_size, :]  # 取15个时间步
            output_data = evo_data[step + window_size, :]      # 取第16个时间步作为输出
            series_idx = evo*evo_size + step
            X_data_series[series_idx, :, :] = input_data
            Y_data_series[series_idx, :] = output_data

    return (X_data_series, Y_data_series)

def pred_evo_new(model, X_test, test_evo, steps, window_size, i_x):
    """
    从第15步开始预测,使用0-14步的真实数据作为初始输入
    """
    evo_size = steps - 1
    Y_submit = np.zeros((test_evo, evo_size, i_x))
    
    # 对每个演化序列进行处理
    for evo in range(test_evo):
        # 获取当前演化序列的起始索引
        start_idx = evo * evo_size
        
        # 获取前15步的真实数据
        initial_data = X_test[start_idx:start_idx+window_size]  # 取出前15个窗口
        current_input = initial_data[-1:]  # 取最后一个窗口作为初始输入
        
        # 前14步直接使用真实值
        for step in range(window_size-1):
            Y_submit[evo, step, :] = X_test[start_idx+step, -1, :]  # 使用每个窗口的最后一个时间步
        
        # 从第15步开始预测
        for step in range(window_size-1, evo_size):
            pred = model.predict(current_input, verbose=0)
            Y_submit[evo, step, :] = pred
            
            # 更新输入窗口
            current_input = np.roll(current_input, shift=-1, axis=1)
            current_input[0, -1, :] = pred

    # reshape到原始维度
    Y_submit = np.reshape(Y_submit, (evo_size*test_evo, i_x))

    return Y_submit

def pred_evo(model, X_test, test_evo, steps, window_size, i_x):
    # 修改可用步数
    evo_size = steps - window_size  # 而不是 steps - 1
    Y_submit = np.zeros((test_evo, evo_size, i_x))
    test_data = X_test[::evo_size,:,:]
    
    for step in tqdm.tqdm(range(evo_size)):
        test_result = model.predict(test_data,verbose=0)
        Y_submit[:,step,:] = test_result
        test_result = np.expand_dims(test_result, axis=1)
        test_data = np.concatenate((test_data,test_result), axis=1)
        test_data = test_data[:, 1:, :]

    # reshape to the original dimensions
    Y_submit = np.reshape(Y_submit,(evo_size*test_evo, i_x))
    return Y_submit


from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, LSTM
# from keras.callbacks import History
from keras import optimizers


def make_RNN_model(window_size, i_x, added_params=0,lr=1e-5):
    """
    Create RNN model

    Parameters
    ----------
    window_size : RNN window size as integer
    i_x : number of grid points as integer
    added_params : number of additional parameters as integer (optional)

    Returns
    -------
    model : keras model
    """

    # Define model architecture
    model = Sequential()

    a = 'relu'
    input_shape = (window_size, i_x+added_params)

    model.add(LSTM(1000, activation=a, input_shape=input_shape))
    model.add(Dense(1000, activation=a))
    model.add(Dense(1000, activation=a))
    model.add(Dense(i_x, activation='sigmoid'))

    # Compile model
    optimizer = optimizers.RMSprop(learning_rate=lr, rho=0.9)
    #loss = 'mean_squared_error'
    loss = 'mean_absolute_error'
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['mse', 'mae'])
    return model

def plt_y(target,pred_step,pred_comp,title):


    #fig,ax = plt.subplots(nrows=5,ncols=1)
    #fig.tight_layout()
    #fig.subplots_adjust(hspace=1.5, wspace=0.5)
    plt.subplot(3,2,1)
    plt.imshow(target)
    plt.colorbar()
    plt.clim(0, 1)
    #plt.xlabel('xy')
    plt.ylabel('z')
    plt.title('label')

    err_step = abs(target - pred_step)

    plt.subplot(3,2,3)
    plt.imshow(pred_step)
    plt.colorbar()
    plt.clim(0, 1)
    #plt.xlabel('xy')
    plt.ylabel('z')
    plt.title('pred_step')


    plt.subplot(3,2,4)
    plt.imshow(err_step,cmap='hot')
    plt.colorbar()
    plt.clim(0, 0.2)
    #plt.xlabel('xy')
    plt.ylabel('z')
    plt.title('err_step')

    err_comp = abs(target - pred_comp)


    plt.subplot(3,2,5)
    plt.imshow(pred_comp)
    plt.colorbar()
    plt.clim(0, 1)
    #plt.xlabel('xy')
    plt.ylabel('z')
    plt.title('pred_comp')


    plt.subplot(3,2,6)
    plt.imshow(err_comp,cmap='hot')
    plt.colorbar()
    plt.clim(0, 0.2)
    plt.xlabel('xy')
    plt.ylabel('z')
    plt.title('err_comp')

    #plt.suptitle(f'{title}')
    plt.savefig(f'{SAVE_DIR}/{title}.png')
    plt.close()

def s_plt(tar,pred_step,pred_comp,title):
    plt.subplot(2,1,1)
    plt.plot(tar,label='label')
    plt.plot(pred_step,label='pred_step')
    plt.plot(pred_comp,label='pred_comp')
    plt.legend()
    plt.xlabel('xy')

    plt.subplot(2,3,4)
    plt.imshow(tar.reshape((32,32)))
    plt.colorbar()
    plt.clim(0, 1)
    plt.title('label')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.subplot(2,3,5)
    plt.imshow(pred_step.reshape((32,32)))
    plt.colorbar()
    plt.clim(0, 1)
    plt.title('pred_step')
    plt.xlabel('x')
    plt.ylabel('y')


    plt.subplot(2,3,6)
    plt.imshow(pred_comp.reshape((32,32)))
    plt.colorbar()
    plt.clim(0, 1)
    plt.title('pred_comp')
    plt.xlabel('x')
    plt.ylabel('y')


    plt.savefig(f'{SAVE_DIR}/{title}.png')
    plt.close()


if __name__=='__main__':

    window_size = 15 # RNN window size
    num_epoch = 100 # number of epochs

    added_params = 0

    ST = 't'
    ST = 's'

    RNN_DIR='rnn_data/'
    SAVE_DIR=f'results/rnn_4000+100_480_{ST}_e{num_epoch}'
    DATA_TRAIN=f'rnn_4000_480'
    DATA_TEST=f'rnn_100_480'
    train_evo, test_evo, steps = 4000, 100, 480
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)


    filename_train = f'{RNN_DIR}/{DATA_TRAIN}'
    filename_test = f'{RNN_DIR}/{DATA_TEST}'

    data_train,data_test,m_max = norm(filename_train,filename_test,ST,normalization='max')


    i_x = data_train.shape[1]

    #(X_test, Y_test) = load_data(data_test, test_evo, steps,window_size) # max/dBm
    (X_test, Y_test) = load_data_no_tile_1st(data_test, test_evo, steps,window_size) # max/dBm
    print('test_data',X_test.shape,Y_test.shape,'i_x',i_x)


    lr=1e-5
    model = make_RNN_model(window_size, i_x,lr=lr)
    #model.summary()

    IS_TRAIN = 1

    if IS_TRAIN:
        ### Fit model on training_data

        if 1:
            batch_size = steps - window_size  # 而不是 steps - 1
            steps_per_epoch = train_evo * (steps - window_size) // batch_size
            history = model.fit(generate_data(data_train, train_evo, steps,window_size),batch_size=batch_size,steps_per_epoch=train_evo,epochs=num_epoch,validation_data=(X_test,Y_test),verbose=1)
        else:
            history = model.fit(X_train, Y_train,epochs=num_epoch,validation_split=0.1,verbose=2)
        plt.figure()
        plt.xlabel('Epoch')
        plt.plot(history.epoch, np.array(history.history['loss']),label='Train Loss')
        plt.plot(history.epoch, np.array(history.history['val_loss']),label = 'Val loss')
        plt.ylim(0,0.02)
        plt.title(f'lr{lr} ep{history.epoch}')
        plt.legend()
        plt.show()
        plt.savefig(f'{SAVE_DIR}/train_loss_{ST}.png')
        plt.close()
        model.save(f'{SAVE_DIR}/evo_wave_{ST}.h5')
    else:
        import keras
        #model = keras.models.load_model(f'rnn_100+10_480/evo_wave.h5')
        model = keras.models.load_model(f'{SAVE_DIR}/evo_wave_{ST}.h5')
        print('load weights')


    #directly calculting all samples (1190,15,1024) -> (1190,1024)
    Y_submit_stepwise = model.predict(X_test)
    print("TESTING STEP-WISE...",Y_submit_stepwise.shape)
    sio.savemat(f'{SAVE_DIR}/test_results_wave_{ST}.mat', {'Y_submit':Y_submit_stepwise, 'Y_test':Y_test, 'steps':steps,'window_size':window_size})

    Y_submit_complete = pred_evo(model, X_test, test_evo, steps, window_size, i_x)
    print("TESTING COMPLETE...",Y_submit_complete.shape)
    sio.savemat(f'{SAVE_DIR}/full_test_results_wave_{ST}.mat', {'Y_submit':Y_submit_complete, 'Y_test':Y_test, 'steps':steps,'window_size':window_size})

    pred_len = steps - 1

    IS_DRAW_SAMPLE = 0
    if IS_DRAW_SAMPLE:
        e = 0
        st = pred_len * e
        ed = pred_len * (e+1)

        title = f'e{e}_err'

        tar = Y_test[st:ed,:]
        pred_step = Y_submit_stepwise[st:ed,:]
        pred_comp = Y_submit_complete[st:ed,:]

        plt_y(tar,pred_step,pred_comp,title)

        target_segment = Y_test.reshape((test_evo, steps-window_size, i_x))
        predict_segment_stepwise = Y_submit_stepwise.reshape((test_evo, steps-window_size, i_x))
        predict_segment_complete = Y_submit_complete.reshape((test_evo, steps-window_size, i_x))
        #10,119,1024
        err_step = abs(target_segment - predict_segment_stepwise)
        err_step = np.mean(err_step,axis=2)
        err_step = np.mean(err_step,axis=0)
        zero = np.array([0])
        err_step = np.append(zero,err_step)
        np.save(f'{SAVE_DIR}/err_step_{ST}.npy',err_step)


        err_comp = abs(target_segment - predict_segment_complete)
        err_comp = np.mean(err_comp,axis=2)
        err_comp = np.mean(err_comp,axis=0)
        zero = np.array([0])
        err_comp = np.append(zero,err_comp)
        np.save(f'{SAVE_DIR}/err_comp_{ST}.npy',err_comp)

        #fig,ax = plt.figure()
        plt.plot(err_step,label='stepwise')
        plt.plot(err_comp,label='complete')
        plt.legend()
        plt.ylim(0,0.02)
        plt.xlabel('z')
        plt.title(f'MAE on z {lr}')
        plt.xticks(np.arange(0,steps-1,steps//10))
        #plt.grid(True,axis='x')
        plt.grid(True)
        plt.savefig(f'{SAVE_DIR}/err_comparsion_{ST}.png')
        plt.close()


    IS_DRAW_SPACE = 0
    if IS_DRAW_SPACE:
        #evo
        e = 0
        #step
        n = pred_len

        for s in range(n):
            all_step = e * pred_len + s
            tar = Y_test[all_step,:]
            pred_step = Y_submit_stepwise[all_step,:]
            pred_comp = Y_submit_complete[all_step,:]
            s_plt(tar,pred_step,pred_comp,f's_e{e}_s{s}_{ST}')

