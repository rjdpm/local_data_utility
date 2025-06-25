import torch
import torch.nn as nn
from transformers import RobertaPreTrainedModel
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Codes.data_utils_local.models_module import *

__all__ = [
    'network_pyfile'
    'Identity',
    'Flatten',
    'View_',
    'ConvStandard',
    'Conv',
    'AllCNN_',
    'LeNet32_',
    'ResidualBlock_',
    'ResNet9_',
    'MobileNet',
    'ResNet18',
    'TimeDistributed',
    'GLU',
    'GatedResidualNetwork',
    'PositionalEncoder',
    'VariableSelectionNetwork',
    'TFT',
    'LSTMnetwork',
    'SMILESLinearNet',
    'GCN',
    'GCN_Connected',
    'MRGCN',
    'ChemBERTaRegressorroberta',
    'ChemBERTaRegressor',
    'SimpleAttention',
]

class Identity(nn.Module):
    """
    Identity layer used to pass input without modification.

    Task:
        Acts as a placeholder in models when conditional operations like dropout 
        or batch norm are toggled off.

    Input:
        Tensor of any shape.

    Output:
        Same tensor, unchanged.
    """
    def __init__(self):
        ...
    
    def forward(self, x):
        ...

class Flatten(nn.Module):
    """
    Flattens all dimensions of a tensor except the batch dimension.

    Task:
        Prepares convolutional outputs for linear layers.

    Input:
        Tensor of shape (N, C, H, W)

    Output:
        Tensor of shape (N, C*H*W)
    """
    def __init__(self):
        ...
    
    def forward(self, x):
        ...


class View_(nn.Module):
    """
    Reshapes tensor to a specified shape using `.view`.

    Task:
        Useful in pipelines like nn.Sequential where dynamic reshaping is needed.

    Args:
        size (tuple): Target shape (can include -1 for inferred dimensions)

    Input:
        Tensor of any shape.

    Output:
        Tensor reshaped to specified dimensions.
    """
    def __init__(self, size):
        ...
    
    def forward(self, tensor):
        ...


class ConvStandard(nn.Conv2d):
    """
    2D convolutional layer with custom Gaussian weight initialization.

    Task:
        Provides more control over initialization for experimentation or 
        stable training in small-data regimes.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int or tuple): Convolutional kernel size.
        stride (int): Stride of convolution.
        padding (int): Zero-padding size.
        w_sig (float): Std deviation factor for weight init.

    Input:
        Tensor of shape (N, C_in, H, W)

    Output:
        Tensor of shape (N, C_out, H_out, W_out)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, output_padding=0, w_sig=1.0):
        ...
    
    def reset_parameters(self):
        ...
    
    def forward(self, input):
        ...
        
class Conv(nn.Sequential):
    """
    Convolutional block with optional batch normalization and activation.

    Task:
        Modular convolution layer with support for transposed convolutions,
        batch normalization, and configurable activation.

    Args:
        in_channels, out_channels, kernel_size, stride, padding (int or tuple)
        activation_fn (nn.Module): e.g., nn.ReLU
        batch_norm (bool): Whether to apply BatchNorm2d
        transpose (bool): Whether to use ConvTranspose2d

    Input:
        Tensor of shape (N, C_in, H, W)

    Output:
        Tensor of shape determined by layer parameters.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, output_padding=0, activation_fn=nn.ReLU, batch_norm=True, transpose=False):
        ...


class AllCNN_(nn.Module):
    """
    All Convolutional Network for classification (variant of All-CNN).

    Task:
        Performs image classification using stacked convolutional layers without
        max pooling, using stride for downsampling.

    Args:
        filters_percentage (float): Scaling for channel width.
        n_channels (int): Input channels (3 for RGB).
        num_classes (int): Output class count.
        dropout (bool): Whether to apply dropout.
        batch_norm (bool): Whether to use batch norm.

    Input:
        Tensor of shape (N, C, H, W)

    Output:
        Tensor of shape (N, num_classes)
    """
    def __init__(self, filters_percentage=1., n_channels=3, num_classes=10, dropout=False, batch_norm=True, padding=0):
        ...
    
    def forward(self, x):
        ...

class LeNet32_(nn.Module):
    """
    LeNet-like architecture adapted for 32x32 images.

    Task:
        A compact CNN for image classification, suitable for CIFAR-10 or similar datasets.

    Args:
        n_classes (int): Number of output classes.
        num_input_channels (int): Input image channels.
        padding (int): Padding for conv layers.

    Input:
        Tensor of shape (N, C, 32, 32)

    Output:
        Tensor of shape (N, n_classes), after softmax.
    """
    def __init__(self, n_classes=10, num_input_channels=3, padding=0):
        ...
    
    def forward(self, x):
        ...


class ResidualBlock_(nn.Module):
    """
    Residual block with optional downsampling (as in ResNet).

    Task:
        Applies two conv layers and adds input via skip connection. Downsamples 
        residual if stride ≠ 1.

    Args:
        in_channels (int): Input feature maps.
        out_channels (int): Output feature maps.
        kernel_size (int): Convolutional kernel size.
        stride (int): Stride for first conv.
        padding (int): Padding to maintain size.

    Input:
        Tensor of shape (N, C_in, H, W)

    Output:
        Tensor of shape (N, C_out, H_out, W_out)
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        ...
    
    def forward(self, x):
        ...


class ResNet9_(nn.Module):
    """
    ResNet-9: Small ResNet variant for low-resolution image classification.

    Task:
        Employs residual blocks and downsampling for efficient classification.

    Args:
        n_classes (int): Output classes.
        num_input_channels (int): Input channels (1 for grayscale, 3 for RGB).
        padding (int): Optional padding for initial conv layer.

    Input:
        Tensor of shape (N, C, H, W)

    Output:
        Tensor of shape (N, n_classes)
    """
    def __init__(self, n_classes=10, num_input_channels=3, padding=0):
        ...
    
    def forward(self, x):
        ...


class MobileNet(nn.Module):
    """
    Custom MobileNetV3 Large model with activation trace outputs.

    Task:
        Feature extraction and classification using MobileNetV3 backbone.
        Returns intermediate activations for interpretability or auxiliary training.

    Input:
        x (Tensor): Image tensor of shape (N, C, H, W)

    Output:
        Tuple of:
            - Output logits (N, 1)
            - Intermediate activations from each major layer (actvn1 to actvn17)
    """
    def __init__(self):
        ...

    def forward(self, x):
        ...


class ResNet18(nn.Module):
    """
    Customized ResNet-18 architecture returning intermediate feature maps.

    Task:
        Image classification with feature extraction at various ResNet stages.

    Input:
        x (Tensor): Image tensor of shape (N, C, H, W)

    Output:
        Output logits (N, n_classes)
    """
    def __init__(self, n_classes=10, num_input_channels=3):
        ...

    def forward(self, x):
        ...


class TimeDistributed(nn.Module):
    """
    Applies a module over each time step of an input sequence.

    Task:
        Mimics Keras' TimeDistributed wrapper by reshaping input tensors.

    Input:
        x (Tensor): Shape (batch_size, time_steps, features)

    Output:
        Tensor with the same time structure, post-module application.
    """
    def __init__(self, module, batch_first=False):
        ...

    def forward(self, x):
        ...


class GLU(nn.Module):
    """
    Gated Linear Unit (GLU) for element-wise feature gating.

    Task:
        Applies a learned gate to modulate input features.

    Input:
        x (Tensor): Feature tensor

    Output:
        Tensor gated elementwise.
    """
    def __init__(self, input_size):
        ...

    def forward(self, x):
        ...


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) with context support and normalization.

    Task:
        Enhances feature transformations using gating and residual connections.

    Input:
        x (Tensor): Input feature sequence
        context (Tensor): Optional contextual input

    Output:
        Transformed and gated tensor with residual and normalization.
    """
    def __init__(self, input_size, hidden_state_size, output_size, dropout, hidden_context_size=None, batch_first=False):
        ...

    def forward(self, x, context=None):
        ...


class PositionalEncoder(nn.Module):
    """
    Fixed sinusoidal positional encoder for transformer inputs.

    Task:
        Injects position information into sequence embeddings.

    Input:
        x (Tensor): Sequence tensor (seq_len, batch, d_model)

    Output:
        Positionally encoded tensor.
    """
    def __init__(self, d_model, max_seq_len=160):
        ...

    def forward(self, x):
        ...

class VariableSelectionNetwork(nn.Module):
    """
    Task:
        Selects relevant input variables using gated residual networks and attention-based weighting.
    
    Inputs:
        - embedding (Tensor): Input tensor of shape [batch_size, time_steps, num_inputs * input_size].
        - context (Tensor, optional): Additional context tensor for conditioning, shape [batch_size, time_steps, context_size].
    
    Outputs:
        - outputs (Tensor): Weighted combination of transformed inputs, shape [batch_size, time_steps, hidden_size].
        - sparse_weights (Tensor): Attention weights for each input variable, shape [batch_size, num_inputs, 1].
    """

    def __init__(self, input_size, num_inputs, hidden_size, dropout, context=None):
        """Initializes submodules for input selection."""
        super(VariableSelectionNetwork, self).__init__()
        ...

    def forward(self, embedding, context=None):
        """Computes sparse weights and transforms each input variable."""
        ...


class TFT(nn.Module):
    """
    Task:
        Implements the Temporal Fusion Transformer (TFT) for interpretable multi-horizon forecasting.

    Inputs:
        - x (Dict[str, Tensor]):
            - 'identifier': Static categorical inputs of shape [batch_size, time_steps, static_variables]
            - 'inputs': Dynamic inputs (real + categorical) of shape [batch_size, time_steps, input_dim]

    Outputs:
        - output (Tensor): Forecasted quantile values, shape [batch_size, decode_length, num_quantiles].
        - encoder_output (Tensor): LSTM encoder output.
        - decoder_output (Tensor): LSTM decoder output.
        - attn_output (Tensor): Output from attention mechanism.
        - attn_output_weights (Tensor): Weights from multi-head attention.
        - encoder_sparse_weights (Tensor): Variable importance weights for encoder inputs.
        - decoder_sparse_weights (Tensor): Variable importance weights for decoder inputs.
    """

    def __init__(self, config):
        """Initializes the full TFT model with embeddings, LSTMs, attention, and feedforward layers."""
        super(TFT, self).__init__()
        ...

    def init_hidden(self):
        """Initializes LSTM hidden state."""
        ...

    def apply_embedding(self, x, static_embedding, apply_masking):
        """
        Applies linear and categorical embeddings to time-varying inputs.

        Inputs:
            - x (Tensor): Input tensor, shape [batch_size, time_steps, input_features].
            - static_embedding (Tensor): Static embeddings to broadcast, shape [batch_size, embed_dim * static_variables].
            - apply_masking (bool): Whether to mask inputs (used in decoder).

        Outputs:
            - embeddings (Tensor): Embedded and concatenated inputs, reshaped for sequence modeling.
        """
        ...

    def encode(self, x, hidden=None):
        """
        Encodes the input sequence using LSTM.

        Inputs:
            - x (Tensor): Input embeddings.
            - hidden (Tensor): Optional hidden state.

        Outputs:
            - output (Tensor): Encoded sequence.
            - hidden (Tensor): Final hidden state.
        """
        ...

    def decode(self, x, hidden=None):
        """
        Decodes the output sequence using LSTM.

        Inputs:
            - x (Tensor): Decoder input embeddings.
            - hidden (Tensor): Initial hidden state from encoder.

        Outputs:
            - output (Tensor): Decoded sequence.
            - hidden (Tensor): Final decoder state.
        """
        ...

    def forward(self, x):
        """
        Performs full forward pass through embedding, variable selection, LSTM, attention, and output head.

        Inputs:
            - x (dict): Dictionary containing static and dynamic input sequences.

        Outputs:
            - See class docstring.
        """
        ...


class LSTMnetwork(nn.Module):
    """
    Task:
        Compares two sequences using LSTM for similarity or regression scoring.

    Inputs:
        - sent1 (Tensor): First input sequence, shape [batch_size, seq_len, embed_dim].
        - sent2 (Tensor): Second input sequence, shape [batch_size, seq_len, embed_dim].

    Outputs:
        - output (Tensor): Regression or similarity score.
        - actv1, actv2, actv3, actv4 (Tensor): Intermediate activations for analysis or interpretability.
    """

    def __init__(self, text_embedding_dimension):
        """Initializes the LSTM and feedforward layers."""
        super().__init__()
        ...

    def forward(self, sent1, sent2):
        """Processes two input sequences and outputs a similarity score."""
        ...


class ViT(nn.Module):
    """
    Task:
        Performs image classification using pretrained Vision Transformer.

    Inputs:
        - pixel_values (Tensor): Input image batch, shape [batch_size, channels, height, width].

    Outputs:
        - logits (Tensor): Class logits, shape [batch_size, num_classes].
    """

    def __init__(self, num_classes=20):
        """Initializes pretrained ViT and a final classification head."""
        super(ViT, self).__init__()
        ...

    def forward(self, pixel_values):
        """Processes image data and returns classification logits."""
        ...

class SMILESLinearNet(nn.Module):
    """
    Task:
        Feedforward network for regression or classification on SMILES-based vector inputs.

    Inputs:
        - list_dims (List[int]): List of dimensions for each linear layer.
        - dropout (float): Dropout probability applied after each layer except the last.

    Outputs:
        - Tensor of shape [batch_size, output_dim] after passing through linear layers.
    """
    def __init__(self, list_dims, dropout):
        ...
        
    def forward(self, x):
        """
        Task:
            Forward pass through MLP with ReLU, BatchNorm, and Dropout.

        Inputs:
            - x (Tensor): Input tensor of shape [batch_size, list_dims[0]]

        Outputs:
            - Tensor: Output tensor after final linear layer.
        """
        ...
class GCNLayer(nn.Module):
    """
    Task:
        Implements a single Graph Convolutional Layer with matrix multiplication.

    Inputs:
        - in_features (int): Number of input features per node.
        - out_features (int): Number of output features per node.

    Outputs:
        - Updated node features after applying graph convolution.
    """
    def __init__(self, in_features, out_features):
        ...
        
    def forward(self, x, adjacency_matrix):
        """
        Task:
            Apply graph convolution using the adjacency matrix.

        Inputs:
            - x (Tensor): Node features [batch_size, num_nodes, in_features]
            - adjacency_matrix (Tensor): Normalized adjacency matrix [batch_size, num_nodes, num_nodes]

        Outputs:
            - Tensor: Updated features [batch_size, num_nodes, out_features]
        """
        ...
class GCN(nn.Module):
    """
    Task:
        Graph Convolutional Network with dense layers for molecular representation learning.

    Inputs:
        - list_dims_gcn (List[int]): Dimensions for GCN layers.
        - list_dims_fc (List[int]): Dimensions for FC layers.
        - dropout (float): Dropout probability.
        - max_num_atom (int): Maximum number of atoms per graph for weighted pooling.

    Outputs:
        - Tensor: Graph-level representation [batch_size, output_dim]
    """
    def __init__(self, list_dims_gcn, list_dims_fc, dropout, max_num_atom):
        ...
        
    def get_features(self, x, adjacency_matrices, degree_matrices):
        """
        Task:
            Extract GCN-based molecular features using adjacency and degree info.

        Inputs:
            - x (Tensor): Node features [batch_size, num_nodes, in_features]
            - adjacency_matrices (Tensor): [batch_size, num_nodes, num_nodes]
            - degree_matrices (Tensor): [batch_size, num_nodes, num_nodes]

        Outputs:
            - Tensor: Pooled graph-level features [batch_size, gcn_out_dim]
        """
        ...
    
    def input2repr(self, x, layers, layer_norms, dropout):
        """
        Task:
            Apply a series of linear layers with normalization and dropout.

        Inputs:
            - x (Tensor): Input to FC stack.
            - layers, layer_norms, dropout: FC architecture components.

        Outputs:
            - Tensor: Final representation.
        """
        ...
    
    def forward(self, x, adjacency_matrices, degree_matrices):
        """
        Task:
            Forward pass through GCN and FC layers.

        Inputs:
            - x: Node features
            - adjacency_matrices, degree_matrices

        Outputs:
            - Tensor: Graph-level prediction [batch_size, output_dim]
        """
        ...
        
class GCN_Connected(nn.Module):
    """
    Task:
        Augmented GCN that merges graph and auxiliary input `y` using learnable weighting.

    Inputs:
        - list_dims_gcn, list_dims_fc: GCN/FC architecture
        - dropout (float): Dropout probability
        - type_ (str): Merge type, default is 'concat'

    Outputs:
        - Tensor: Combined representation
    """
    def __init__(self, list_dims_gcn, list_dims_fc, dropout, type_='concat'):
        ...
        
    def get_features(self, x, adjacency_matrices, degree_matrices, y):
        """
        Task:
            Extract graph features and merge with auxiliary `y`.

        Inputs:
            - x: Node features
            - adjacency_matrices: Adjacency matrices
            - degree_matrices: Degree matrices
            - y: External input [batch_size, feature_dim]

        Outputs:
            - Tensor: Combined features [batch_size, fusion_dim]
        """
        ...
    
    def input2repr(self, x, layers, layer_norms, dropout):
        ...
    
    def forward(self, x, adjacency_matrix, degree_matrix, y):
        """
        Task:
            Forward pass through GCN and FC layers, integrating auxiliary input.

        Inputs:
            - x: Node features
            - adjacency_matrix, degree_matrix
            - y: External feature

        Outputs:
            - Tensor: Final output [batch_size]
        """
        ...


class RGCNConv(nn.Module):
    """
    Task:
        Relation-aware Graph Convolution Layer for multi-relational graphs.

    Inputs:
        - in_channels (int): Input node feature dim.
        - out_channels (int): Output node feature dim.
        - num_relations (int): Number of bond types or edge relations.

    Outputs:
        - Tensor: Transformed node features with all relation contributions.
    """
    def __init__(self, in_channels, out_channels, num_relations=4, bias=True):
        ...
        
    def reset_parameters(self):
        """Task: Initialize parameters using Xavier initialization."""
        ...
        
    def forward(self, x, adjacency_tensor):
        """
        Task:
            Perform multi-relational message passing.

        Inputs:
            - x (Tensor): Node features [batch_size, num_nodes, in_channels]
            - adjacency_tensor (Tensor): [batch_size, num_relations, num_nodes, num_nodes]

        Outputs:
            - Tensor: Updated node features [batch_size, num_nodes, out_channels]
        """
        ...


class MRGCN(nn.Module):
    """
    Task:
        Multi-Relational Graph Convolutional Network with relation-aware aggregation and dense prediction head.

    Inputs:
        - list_dims_gcn: Dimensions of GCN layers
        - list_dims_fc: Dimensions of FC layers
        - dropout: Dropout probability
        - max_num_atom: For weighted pooling (not used directly here)
        - num_relations: Number of edge types

    Outputs:
        - Tensor: Final prediction or representation [batch_size, output_dim]
    """
    def __init__(self, list_dims_gcn, list_dims_fc, dropout, max_num_atom, num_relations=4):
        ...
        
    def input2repr(self, x, layers, layer_norms, dropout):
        ...
        
    def get_features(self, x, adjacency_tensor, degree_tensor):
        """
        Task:
            Extract graph-level features from multi-relational input.

        Inputs:
            - x (Tensor): Node features
            - adjacency_tensor (Tensor): Edge-type-specific adjacency matrices
            - degree_tensor (Tensor): Node degree matrices

        Outputs:
            - Tensor: Aggregated graph features [batch_size, gcn_out_dim]
        """
        ...
        
    def forward(self, x, adjacency_tensor, degree_tensor):
        """
        Task:
            Forward pass from graph input to final prediction via GCN and FC layers.

        Inputs:
            - x: Node features
            - adjacency_tensor: Multi-relational adjacency
            - degree_tensor: Node degree tensor

        Outputs:
            - Tensor: Final prediction/representation
        """
        ...

class ChemBERTaRegressorroberta(RobertaPreTrainedModel):
    """
    Task:
        A regression model built on top of ChemBERTa (RobertaModel) for predicting molecular properties.
    """

    def __init__(self, config, list_dims, dropout=0.2):
        """
        Task:
            Initializes the ChemBERTaRegressorroberta model with a Roberta backbone and MLP head.

        Input:
            config (PretrainedConfig): Configuration object for RobertaModel.
            list_dims (list[int]): Dimensions for the MLP regressor head.
            dropout (float): Dropout probability for regularization.

        Output:
            None
        """
        ...
        
        
    def make_mlp(self, list_dims, dropout):
        """
        Task:
            Constructs a sequential MLP regressor.

        Input:
            list_dims (list[int]): Dimensions of the layers in the MLP.
            dropout (float): Dropout probability between layers.

        Output:
            nn.Sequential: An MLP with linear, normalization, activation, and dropout layers.
        """
        ...
        

    def embeddings(self, input_ids, attention_mask=None):
        """
        Task:
            Extracts the CLS embedding from Roberta's output.

        Input:
            input_ids (Tensor): Tokenized input IDs of shape [batch_size, seq_len].
            attention_mask (Tensor, optional): Attention mask indicating valid tokens.

        Output:
            Tensor: CLS token embedding of shape [batch_size, hidden_size].
        """
        ...
        

    def forward(self, input_ids, attention_mask=None):
        """
        Task:
            Performs a forward pass through the model to predict the regression value.

        Input:
            input_ids (Tensor): Input token IDs.
            attention_mask (Tensor, optional): Attention mask.

        Output:
            Tensor: Predicted output of shape [batch_size, output_dim].
        """
        ...
        

class ChemBERTaRegressor(nn.Module):
    """
    Task:
        ChemBERTa-based regressor for molecular property prediction using AutoModel.
    """
    def __init__(self, model_name: str, list_dims: list[int], dropout: float = 0.2):
        """
        Task:
            Initialize model with a pretrained transformer and a regression MLP.

        Input:
            model_name (str): Huggingface model identifier (e.g., 'seyonec/ChemBERTa').
            list_dims (list[int]): Dimensions for the MLP head.
            dropout (float): Dropout probability.

        Output:
            None
        """
        ...
        

    def make_mlp(self, list_dims, dropout):
        """
        Task:
            Build a feedforward regressor with optional dropout and activation.

        Input:
            list_dims (list[int]): List of layer dimensions.
            dropout (float): Dropout rate.

        Output:
            nn.Sequential: Multi-layer perceptron for regression.
        """
        ...
        

    def embeddings(self, input_ids, attention_mask):
        """
        Task:
            Compute mean-pooled embedding over valid token positions.

        Input:
            input_ids (Tensor): Input token IDs of shape [B, T].
            attention_mask (Tensor): Attention mask of shape [B, T].

        Output:
            Tensor: Mean pooled embedding of shape [B, H].
        """
        ...
        

    def forward(self, input_ids, attention_mask):
        """
        Task:
            Perform forward pass for regression prediction.

        Input:
            input_ids (Tensor): Tokenized inputs.
            attention_mask (Tensor): Valid attention positions.

        Output:
            Tensor: Regression prediction.
        """
        ...
        

class SimpleAttention(nn.Module):
    """
    Task:
        Implements a simple additive attention mechanism for sequence summarization.
    """

    def __init__(self, input_dim, hidden_dim):
        """
        Task:
            Initialize layers for additive attention.

        Input:
            input_dim (int): Dimensionality of input vectors.
            hidden_dim (int): Dimensionality of attention space.

        Output:
            None
        """
        ...


    def forward(self, x):
        """
        Task:
            Compute weighted context vector using additive attention.

        Input:
            x (Tensor): Input sequence tensor of shape [B, S, D].

        Output:
            context (Tensor): Weighted context vector [B, H].
            attn_weights (Tensor): Attention weights [B, S, 1].
        """
        ...
