# Segment Detection and Transformer-LSTM Model for Workers Trajectory Prediction

This project implements Segment Detection for input trajectory segments, aTransformer-LSTM model for workers trajectory prediction, combining the strengths of the Transformer encoder and LSTM networks. The model is trained to predict a single output value based on input features.

## Model Description

The model consists of a Transformer encoder followed by an LSTM network and a linear decoder. The Transformer encoder captures long-range dependencies in the input sequence, while the LSTM network processes the encoded features to produce a final prediction.

## Data Structure

The input data structure for training the model is as follows:
- `x`: Input features of shape `(sequence_length, feature_size)`
- `y`: Target output of shape `(1,)`
- `edge_index`: Edge indices for graph-based data (if applicable)

## Training

To train the model, the following steps are performed:
1. Load and preprocess the data.
2. Initialize the model, loss function, and optimizer.
3. Train the model over a specified number of epochs, computing the loss and updating the model parameters.

## Usage

To train the model, run the `train.py` script with the appropriate dataset. Ensure you have installed the necessary dependencies as specified in `requirements.txt`.

## Requirements

- PyTorch
- Torch Geometric

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
