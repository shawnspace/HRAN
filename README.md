# Hierarchical Recurrent Attention Network (HRAN) for Chatbots

This project implements a Hierarchical Recurrent Attention Network (HRAN) for building a chatbot. The model is built using TensorFlow and is designed to handle multi-turn dialogues.

## Features

*   **Hierarchical Architecture:** The model uses a hierarchical structure to process conversations, with one RNN for encoding individual utterances and another for encoding the context of the conversation.
*   **Attention Mechanism:** It employs an attention mechanism at both the utterance and context levels to focus on the most relevant parts of the conversation.
*   **Beam Search:** For prediction, the model can use beam search to generate more fluent and coherent responses.
*   **Pre-trained Embeddings:** The model can be initialized with pre-trained word embeddings like GloVe.

## Requirements

*   TensorFlow
*   Python 3

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file is not provided in the project. You will need to install the necessary libraries, primarily TensorFlow.)*

3.  **Download pre-trained word embeddings:**
    Download the GloVe embeddings and place them in the `data` directory.

4.  **Prepare the data:**
    The project expects the data to be in the `.tfrecords` format. The `prepare_context_RG_data.py` script can be used to convert raw data into this format.

## Usage

### Training

To train the model, run the `train.py` script:

```bash
python train.py --data_dir ./data/multi --model_dir ./model/persona_chat1
```

You can customize the training process by modifying the hyperparameters in `hparam.py`.

### Evaluation

To evaluate the model, run the `evaluate.py` script:

```bash
python evaluate.py --data_dir ./data/multi --model_dir ./model/persona_chat1
```

### Prediction

To generate responses using a trained model, use the `online_predict.py` script:

```bash
python online_predict.py --model_dir ./model/persona_chat1
```

## Hyperparameters

The hyperparameters for the model and training process are defined in `hparam.py`. You can modify this file to experiment with different settings. Key hyperparameters include:

*   `word_dim`: Dimensionality of word embeddings.
*   `word_rnn_num_units`: Number of units in the utterance-level RNN.
*   `context_rnn_num_units`: Number of units in the context-level RNN.
*   `learning_rate`: The learning rate for the optimizer.
*   `batch_size`: The batch size for training.
*   `beam_width`: The beam width for beam search decoding.

## Files

*   `HRAN.py`: Contains the implementation of the HRAN model.
*   `train.py`: The script for training the model.
*   `evaluate.py`: The script for evaluating the model.
*   `online_predict.py`: The script for running predictions.
*   `hparam.py`: Contains the hyperparameters for the model.
*   `helper.py`: Contains helper functions for data processing.
*   `modekeys.py`: Defines the different modes (train, eval, predict).
*   `prepare_context_RG_data.py`: A script to prepare the data for the model.
