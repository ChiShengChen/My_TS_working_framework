import os
import sys
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../../..'))

from basicts.metrics import masked_mae, masked_mse, masked_rmse, masked_mape
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings

from .arch import HybridTransformer

############################## Hot Parameters ##############################
# Dataset & Metrics configuration
DATA_NAME = 'Electricity'  # Dataset name
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings['INPUT_LEN']  # Length of input sequence
OUTPUT_LEN = regular_settings['OUTPUT_LEN']  # Length of output sequence
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL']
RESCALE = regular_settings['RESCALE']
NULL_VAL = regular_settings['NULL_VAL']
# Model architecture and parameters
MODEL_ARCH = HybridTransformer
NUM_NODES = 321 # Number of nodes in Electricity dataset

MODEL_PARAM = {
    # Architecture parameters from HybridTransformer __init__
    "seq_len": INPUT_LEN,
    "pred_len": OUTPUT_LEN,
    "enc_in": NUM_NODES,
    "c_out": NUM_NODES,
    "hidden_dim": 128,                    # Example: common choice
    "num_total_encoder_layers": 3,        # Example: 2 TF layers + 1 Quantum layer
    "n_heads_tf": 8,                      # Example: For standard TransformerEncoderLayer
    "dim_feedforward_tf_factor": 4,       # Example: For standard TransformerEncoderLayer
    "n_heads_qel": 4,                     # Example: For QuantumEncoderLayer's attention
    "ff_factor_qel": 4,                   # Example: For QuantumEncoderLayer's FFN
    "dropout_rate": 0.1,
    "n_qubits_q": 4,                      # Example: Number of operational qubits in QuantumLayer
    "n_layers_q_circuit": 2,              # Example: Number of layers in the quantum circuit
    "use_quantum_middle_layer_only": False # Places quantum layer at the end by default
}
NUM_EPOCHS = 50 # Adjusted for potentially longer training time of Q-models

############################## General Configuration ##############################
CFG = EasyDict()
CFG.DESCRIPTION = 'QASA model configuration for Electricity dataset'
CFG.GPU_NUM = 1
CFG.RUNNER = SimpleTimeSeriesForecastingRunner

############################## Dataset Configuration ##############################
CFG.DATASET = EasyDict()
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = TimeSeriesForecastingDataset
CFG.DATASET.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_val_test_ratio': TRAIN_VAL_TEST_RATIO,
    'input_len': INPUT_LEN,
    'output_len': OUTPUT_LEN,
})

############################## Scaler Configuration ##############################
CFG.SCALER = EasyDict()
CFG.SCALER.TYPE = ZScoreScaler
CFG.SCALER.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_ratio': TRAIN_VAL_TEST_RATIO[0],
    'norm_each_channel': NORM_EACH_CHANNEL,
    'rescale': RESCALE,
})

############################## Model Configuration ##############################
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
# BasicTS framework expects these features.
# Adjust if your model or data uses covariates.
# history_data (batch, L, N, C) -> C=1 for default
# future_data (batch, S, N, C)
# x_enc = history_data[:, :, :, 0] # (B, L, N)
# y = future_data[:, :, :, 0] # (B, S, N)
CFG.MODEL.FORWARD_FEATURES = [0] # Index of history_data in batch output by TimeSeriesForecastingDataset
CFG.MODEL.TARGET_FEATURES = [0]  # Index of future_data in batch output by TimeSeriesForecastingDataset

############################## Metrics Configuration ##############################
CFG.METRICS = EasyDict()
CFG.METRICS.FUNCS = EasyDict({
    'MAE': masked_mae,
    'MSE': masked_mse,
    'RMSE': masked_rmse,
    'MAPE': masked_mape
})
CFG.METRICS.TARGET = 'MAE'
CFG.METRICS.NULL_VAL = NULL_VAL

############################## Training Configuration ##############################
CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    MODEL_ARCH.__name__,
    '_'.join([DATA_NAME, str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)
CFG.TRAIN.LOSS = masked_mse # Or masked_mae, consistent with metrics target
# Optimizer: AdamW was used in original QASA
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "AdamW"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 1e-4, # Original QASA: 5e-5, adjust as needed
    "weight_decay": 1e-4,
}
# LR Scheduler: CosineAnnealingLR was used in original QASA
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "CosineAnnealingLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "T_max": NUM_EPOCHS,
    "eta_min": 1e-7 # Example: very small lr at the end
}
CFG.TRAIN.CLIP_GRAD_PARAM = {
    'max_norm': 5.0
}
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 32 # Adjusted, Q-models can be memory intensive
CFG.TRAIN.DATA.SHUFFLE = True

############################## Validation Configuration ##############################
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 32

############################## Test Configuration ##############################
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 32

############################## Evaluation Configuration ##############################
CFG.EVAL = EasyDict()
CFG.EVAL.USE_GPU = True 