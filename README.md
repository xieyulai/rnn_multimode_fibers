# Numerical Simulation and RNN Prediction Project

## 0. Acknowledgement and Purpose
This project is based on https://github.com/ugurtegin/MMF_RNN_Reuse. 

- We use the numerical simulation code from the repository to generate the data. 
- We use the RNN code from the repository to train the RNN models. 
- We reproduce the RNN's results for comparing them with our method (https://github.com/xieyulai/query-based-prediction-for-multimode-fibers).

## 1. Procedure
1. **Data Generation (0)**
2. **Data Processing for RNN (1)**
3. **RNN Model Training (2)**

## 2. Environment
- CuPy
- Tensorflow 2.4.0
- Python 3

64 GB memory is recommended for generating N=4000 samples and RNN training.

## 3. Data Generation

### Usage

Run the data generation script to create the dataset:
```bash
python 0_generate.py
```

Manually specify `N` (number of samples) and `L` (length of each sample) in the script.

We select `N = 4000` for our experiments.

The generated result will be saved in `ori_data/data_{N}`.

## 5. Data Processing for RNN

Once the data is generated, it needs to be processed into a format suitable for RNN training. This involves organizing the data into sequences with a specified window size.



### Usage

Process the data for RNN using the following command:

```bash
python 1_compose_for_rnn.py
```

The processed data will be saved in `rnn_data/rnn_{N}_{L}`.

## 6. RNN Model Training

The final step is to train the RNN model using the processed data. The model can be configured to train on either spectral ('s') or temporal ('t') data.


### Usage

Train the RNN model with the following command:

```bash
python 2_RNN_Modes_3W.py
```

Manually specify `ST = 's'` or `ST = 't'` for training spatial or temporal networks.
Manually specify `IS_TRAIN = 1` for training or `IS_TRAIN = 0` for loading the model.

Configure the training like the following:

```python
    window_size = 15 # RNN window size
    num_epoch = 100 # number of epochs

    added_params = 0

    ST = 's'
    ST = 't'

    RNN_DIR='rnn_data/'
    SAVE_DIR=f'results/rnn_4000+100_480_{ST}_e{num_epoch}'
    DATA_TRAIN=f'rnn_4000_480'
    DATA_TEST=f'rnn_100_480'
    train_evo, test_evo, steps = 4000, 100, 480
```
The results will be saved in `results/` like `results/rnn_4000+100_480_{ST}_e{num_epoch}`.
