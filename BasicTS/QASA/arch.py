import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import pennylane as qml
print(f"PennyLane version used in arch.py: {qml.__version__}")
try:
    import pennylane.qnn.torch as qnn_torch_module
    print(f"Path of pennylane.qnn.torch: {qnn_torch_module.__file__}")
except ImportError:
    print("Could not import pennylane.qnn.torch to check path.")

# Helper function for initialization
def init_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv1d):
        init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)

# Parameterized Quantum Circuit
def quantum_circuit_fn(inputs, weights, n_qubits, n_layers_q_circuit):
    for i in range(n_qubits):
        qml.RX(inputs[i], wires=i)
        qml.RZ(inputs[i], wires=i)
    for i in range(n_qubits):
        qml.RX(weights[0, i], wires=i)
        qml.RZ(weights[1, i], wires=i)
    for l in range(1, n_layers_q_circuit):
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])
            qml.RY(weights[l, i], wires=i)
            qml.RZ(weights[l, i], wires=i)
        # Ancilla qubit interaction for more complex entanglement, if desired
        qml.CNOT(wires=[n_qubits - 1, n_qubits]) # Connect last operational qubit to ancilla
        qml.RY(weights[l, n_qubits], wires=n_qubits) # Rotate ancilla
    # Return a list of expval MeasurementProcess objects
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QuantumLayer(nn.Module):
    def __init__(self, input_dim, output_dim, n_qubits_q, n_layers_q_circuit, dev_name="default.qubit"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_qubits_q = n_qubits_q
        self.n_layers_q_circuit = n_layers_q_circuit

        self.dev = qml.device(dev_name, wires=self.n_qubits_q + 1)

        def _circuit_for_qnode(inputs, weights):
            # inputs from TorchLayer (when processing sample_input of shape (1, n_qubits_q))
            # will have shape (1, n_qubits_q).
            # quantum_circuit_fn expects inputs of shape (n_qubits_q,).
            # print(f"Inside _circuit_for_qnode, original inputs.shape: {inputs.shape}, n_qubits_q: {self.n_qubits_q}")
            if inputs.ndim == 2 and inputs.shape[0] == 1:
                processed_inputs = inputs.squeeze(0) # Remove the batch dim of 1
            else:
                # This case should ideally not be hit if QuantumLayer.forward loop is correct
                processed_inputs = inputs 
            # print(f"Inside _circuit_for_qnode, processed_inputs.shape: {processed_inputs.shape}")
            return quantum_circuit_fn(processed_inputs, weights, self.n_qubits_q, self.n_layers_q_circuit)

        self.qnode_instance = qml.QNode(_circuit_for_qnode, self.dev, interface="torch", diff_method="parameter-shift")

        self.weight_shapes = {"weights": (self.n_layers_q_circuit, self.n_qubits_q + 1)}
        self.qlayer = qml.qnn.TorchLayer(self.qnode_instance, self.weight_shapes)

        # Projects from input_dim (e.g. hidden_dim) to n_qubits_q for quantum circuit input
        self.input_proj = nn.Linear(input_dim, self.n_qubits_q) 
        self.norm = nn.LayerNorm(self.n_qubits_q)
        # Projects from n_qubits_q (quantum output) back to output_dim (e.g. hidden_dim)
        self.output_proj = nn.Linear(self.n_qubits_q, output_dim)

        self.input_proj.apply(init_weights)
        self.output_proj.apply(init_weights)

    def forward(self, x): # x shape: (effective_batch_size, input_dim (hidden_dim))
        x_orig_device = x.device
        
        # Project to n_qubits_q for quantum circuit
        x_projected = self.input_proj(x) # Shape: (effective_batch_size, n_qubits_q)
        x_proj_tanh = torch.tanh(x_projected) 
        x_norm = self.norm(x_proj_tanh) # Shape: (effective_batch_size, n_qubits_q)
        
        # Prepare input for CPU-based qlayer
        x_norm_cpu = x_norm.detach().cpu() # Shape: (effective_batch_size, n_qubits_q)
        
        outputs_list_cpu = []
        # Loop through each sample in the batch
        for i in range(x_norm_cpu.shape[0]):
            sample_input = x_norm_cpu[i].unsqueeze(0) # Shape: (1, n_qubits_q)
            
            # self.qlayer expects (batch, features) and returns tuple for multiple measurements
            # For a single sample input (1, n_qubits_q), output should be tuple of 3 tensors, each (1,)
            quantum_output_per_sample_tuple_cpu = self.qlayer(sample_input)
            
            # Stack the tuple/list of tensors into a single tensor for this sample
            if isinstance(quantum_output_per_sample_tuple_cpu, tuple):
                stacked_output_per_sample_cpu = torch.stack(quantum_output_per_sample_tuple_cpu, dim=-1) # Shape: (1, n_qubits_q)
            elif isinstance(quantum_output_per_sample_tuple_cpu, list):
                stacked_output_per_sample_cpu = torch.stack(quantum_output_per_sample_tuple_cpu, dim=-1) # Shape: (1, n_qubits_q)
            else: # Should be a single tensor if qnode returned one value (not the case here)
                stacked_output_per_sample_cpu = quantum_output_per_sample_tuple_cpu
            
            outputs_list_cpu.append(stacked_output_per_sample_cpu)
        
        # Concatenate all sample outputs along the batch dimension
        quantum_output_batch_stacked_cpu = torch.cat(outputs_list_cpu, dim=0) # Shape: (effective_batch_size, n_qubits_q)
        
        # Move result back to original device
        quantum_output_final_gpu = quantum_output_batch_stacked_cpu.to(x_orig_device)
        
        # Project quantum output back to output_dim (hidden_dim)
        out_projected_gpu = self.output_proj(quantum_output_final_gpu) # Shape: (effective_batch_size, output_dim)

        # Skip connection: x is (eff_batch, input_dim), out_projected_gpu is (eff_batch, output_dim)
        # input_dim and output_dim are the same (hidden_dim) when QuantumLayer is used in QuantumEncoderLayer
        if self.input_dim == self.output_dim:
            return x + out_projected_gpu 
        else:
            return out_projected_gpu

class QuantumEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads_qel, dropout_rate, n_qubits_q, n_layers_q_circuit, ff_factor_qel=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads_qel, dropout=dropout_rate, batch_first=True)
        self.v_quantum = QuantumLayer(hidden_dim, hidden_dim, n_qubits_q, n_layers_q_circuit)
        
        dim_feedforward_qel = hidden_dim * ff_factor_qel
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward_qel),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_feedforward_qel, hidden_dim),
            nn.Dropout(dropout_rate)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn.apply(init_weights)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        # x shape: (batch_size, seq_len, hidden_dim)
        attn_out, _ = self.attn(x, x, x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        x = x + F.dropout(attn_out, p=self.attn.dropout, training=self.training) # Explicit dropout application
        x = self.norm1(x)
        
        # Reshape for QuantumLayer: (batch_size * seq_len, hidden_dim)
        batch_size, seq_len, features = x.shape
        x_flat = x.reshape(batch_size * seq_len, features)
        
        q_out_flat = self.v_quantum(x_flat) # Quantum processing
        
        q_out = q_out_flat.view(batch_size, seq_len, features) # Reshape back
        
        # FFN part (applied to q_out, as in original QASA)
        ffn_out = self.ffn(q_out)
        # Original QASA: x = self.norm2(q_out + self.ffn(q_out))
        # Let's follow common transformer: Add & Norm after FFN
        x = q_out + ffn_out # Or x = x + ffn_out if quantum layer is an alternative path
        x = self.norm2(x) # Apply norm after FFN and skip connection
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # shape (1, max_len, d_model)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]

class HybridTransformer(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, c_out,
                 hidden_dim=256, num_total_encoder_layers=4,
                 n_heads_tf=8, dim_feedforward_tf_factor=4,
                 n_heads_qel=4, ff_factor_qel=4,
                 dropout_rate=0.1,
                 n_qubits_q=4, n_layers_q_circuit=2,
                 use_quantum_middle_layer_only=False # If true, only one quantum layer in the middle
                 ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.c_out = c_out
        self.hidden_dim = hidden_dim

        self.embedding = nn.Linear(enc_in, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=seq_len) # max_len should be context_window or similar
        self.dropout = nn.Dropout(dropout_rate)

        if num_total_encoder_layers <= 0:
            raise ValueError("num_total_encoder_layers must be positive.")

        encoder_layers = []
        dim_feedforward_tf = hidden_dim * dim_feedforward_tf_factor

        if use_quantum_middle_layer_only:
            if num_total_encoder_layers < 1:
                raise ValueError("Need at least 1 layer for a single quantum layer.")
            
            num_tf_layers_before = (num_total_encoder_layers -1) // 2
            num_tf_layers_after = num_total_encoder_layers - 1 - num_tf_layers_before

            for _ in range(num_tf_layers_before):
                encoder_layers.append(
                    nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads_tf,
                                               dim_feedforward=dim_feedforward_tf,
                                               dropout=dropout_rate, batch_first=True, norm_first=True)
                )
            encoder_layers.append(
                QuantumEncoderLayer(hidden_dim, n_heads_qel, dropout_rate,
                                    n_qubits_q, n_layers_q_circuit, ff_factor_qel=ff_factor_qel)
            )
            for _ in range(num_tf_layers_after):
                encoder_layers.append(
                    nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads_tf,
                                               dim_feedforward=dim_feedforward_tf,
                                               dropout=dropout_rate, batch_first=True, norm_first=True)
                )

        else: # Original QASA: TF layers first, then one QuantumEncoderLayer
            num_tf_layers = num_total_encoder_layers - 1
            if num_tf_layers < 0: # Only quantum layer
                 encoder_layers.append(
                    QuantumEncoderLayer(hidden_dim, n_heads_qel, dropout_rate,
                                        n_qubits_q, n_layers_q_circuit, ff_factor_qel=ff_factor_qel)
                )
            else:
                for _ in range(num_tf_layers):
                    encoder_layers.append(
                        nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads_tf,
                                                   dim_feedforward=dim_feedforward_tf,
                                                   dropout=dropout_rate, batch_first=True, norm_first=True)
                    )
                encoder_layers.append(
                    QuantumEncoderLayer(hidden_dim, n_heads_qel, dropout_rate,
                                        n_qubits_q, n_layers_q_circuit, ff_factor_qel=ff_factor_qel)
                )
        
        self.encoder = nn.ModuleList(encoder_layers)
        
        # Output layer: projects the representation of the last time step to the prediction length
        self.output_layer = nn.Linear(hidden_dim, pred_len * c_out)

        self.embedding.apply(init_weights)
        self.output_layer.apply(init_weights)

    def forward(self, history_data, future_data=None, x_mark_enc=None, x_mark_dec=None, batch_seen=None, epoch=None, train=None, mask=None, **unused_kwargs):
        # history_data (from runner) corresponds to x_enc for this model.
        x_enc = history_data

        if x_enc.ndim == 4 and x_enc.shape[-1] == 1:
            x_enc = x_enc.squeeze(-1)
        
        embedded_x = self.embedding(x_enc)
        pos_encoded_x = self.pos_encoding(embedded_x)
        x = self.dropout(pos_encoded_x)

        for layer in self.encoder:
            if isinstance(layer, QuantumEncoderLayer):
                x = layer(x)
            elif isinstance(layer, nn.TransformerEncoderLayer):
                x = layer(x, src_mask=None, src_key_padding_mask=None) 
            else:
                x = layer(x)

        last_time_step_representation = x[:, -1, :]
        output_flat = self.output_layer(last_time_step_representation)
        
        current_batch_size = output_flat.size(0)
        output = output_flat.view(current_batch_size, self.pred_len, self.c_out)
        
        print(f"HybridTransformer output shape: {output.shape}, pred_len: {self.pred_len}, c_out: {self.c_out}") # Debugging output shape
        return output

# To ensure the file is created even if empty initially by the tool
# pass # Commenting out original pass if it exists

import torch
import numpy as np
# It's better to import from the exact location if basicts.metrics.mae is the canonical one
from basicts.metrics import masked_mae as original_masked_mae
from basicts.metrics import masked_mse as original_masked_mse
from basicts.metrics import masked_rmse as original_masked_rmse # Added for RMSE wrapper
from basicts.metrics import masked_mape as original_masked_mape # Added for MAPE wrapper

def debugging_masked_mae_wrapper(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan, **kwargs):
    print(f"DEBUG WRAPPER (before squeeze): prediction.shape={prediction.shape}, target.shape={target.shape}")
    target_for_loss = target
    if prediction.ndim == 3 and target.ndim == 4 and target.shape[-1] == 1:
        if prediction.shape == target.shape[:-1]:
            print(f"DEBUG WRAPPER: Target is 4D with last dim 1, and its first {prediction.ndim} dims match prediction. Squeezing target\'s last dimension.")
            target_for_loss = target.squeeze(-1)
            print(f"DEBUG WRAPPER (after squeeze): New target_for_loss.shape={target_for_loss.shape}")
        else:
            print(f"DEBUG WRAPPER: Target is 4D with last dim 1, but its first {prediction.ndim} dims ({target.shape[:-1]}) do not match prediction dims ({prediction.shape}). Not squeezing automatically.")
    return original_masked_mae(prediction=prediction, target=target_for_loss, null_val=null_val)

# New wrapper for MSE
def debugging_masked_mse_wrapper(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan, **kwargs):
    print(f"DEBUG MSE WRAPPER (before squeeze): prediction.shape={prediction.shape}, target.shape={target.shape}")
    target_for_loss = target
    if prediction.ndim == 3 and target.ndim == 4 and target.shape[-1] == 1:
        if prediction.shape == target.shape[:-1]:
            print(f"DEBUG MSE WRAPPER: Squeezing target\'s last dimension.")
            target_for_loss = target.squeeze(-1)
            print(f"DEBUG MSE WRAPPER (after squeeze): New target_for_loss.shape={target_for_loss.shape}")
    return original_masked_mse(prediction=prediction, target=target_for_loss, null_val=null_val)

# New wrapper for RMSE
def debugging_masked_rmse_wrapper(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan, **kwargs):
    print(f"DEBUG RMSE WRAPPER (before squeeze): prediction.shape={prediction.shape}, target.shape={target.shape}")
    target_for_loss = target
    if prediction.ndim == 3 and target.ndim == 4 and target.shape[-1] == 1:
        if prediction.shape == target.shape[:-1]:
            print(f"DEBUG RMSE WRAPPER: Squeezing target\'s last dimension.")
            target_for_loss = target.squeeze(-1)
            print(f"DEBUG RMSE WRAPPER (after squeeze): New target_for_loss.shape={target_for_loss.shape}")
    return original_masked_rmse(prediction=prediction, target=target_for_loss, null_val=null_val)

# New wrapper for MAPE
def debugging_masked_mape_wrapper(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan, **kwargs):
    print(f"DEBUG MAPE WRAPPER (before squeeze): prediction.shape={prediction.shape}, target.shape={target.shape}")
    target_for_loss = target
    if prediction.ndim == 3 and target.ndim == 4 and target.shape[-1] == 1:
        if prediction.shape == target.shape[:-1]:
            print(f"DEBUG MAPE WRAPPER: Squeezing target\'s last dimension.")
            target_for_loss = target.squeeze(-1)
            print(f"DEBUG MAPE WRAPPER (after squeeze): New target_for_loss.shape={target_for_loss.shape}")
    return original_masked_mape(prediction=prediction, target=target_for_loss, null_val=null_val) 