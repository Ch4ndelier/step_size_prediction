# step_size_prediction

This is project to predict step size using the data from accelerometer and gyroscope. 


## To get processed data

Link: https://mega.nz/folder/c4oGRCRZ#0PcHlPWv8eFrxJzI_DFyvg

## To start

We suggest you run 

`pip install -r requirements.txt`

`config.py` contains the parameters used in training.You should write your own config.py and put it in the root directory.

Here is an example of config.py:
```
LR = 0.0003
DATA_DIR = "./data/step_cord_run/"
BATCH_SIZE = 5
NUM_EPOCHS = 30
HIDDEN_SIZE = 12
NUM_LAYERS = 2
```

You may write your own config.py to try different hyperparameters or use different data.
## Scripts

* `data_preprocess.py`: Generates the data used in LSTM training.

* `train_lstm.py`: Train the LSTM network.