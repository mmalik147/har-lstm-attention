# har-lstm-attention
Human Activity Recognition with custom PyTorch BiLSTM and Attention
# Human Activity Recognition Using LSTM and Attention

## Project Members

M.Malik         22F-3487
Shoaib Mansoor  23F-6037

## 1. Project Introduction

Human Activity Recognition (HAR) is the process of identifying human physical activities using sensor data. It is an important application of machine learning and deep learning, particularly in healthcare monitoring, fitness tracking, and human behavior analysis.

In this project, we developed a deep learning model using Long Short-Term Memory (LSTM) networks and an Attention mechanism to classify different human activities.

The main objective is to learn patterns from sequential sensor signals and accurately recognize the activity being performed.

## 2. Dataset Description

The project uses sensor-based time-series data for human activity recognition.

The sensor signals are organized into windows and used as input to the deep learning model. Each window represents a sequence of sensor measurements associated with a human activity.

The model is trained to recognize the activity classes represented in the dataset.

The dataset is prepared and processed before training to ensure that the input has the required format for the model.

## 3. Model Architecture

The project uses an LSTM-based deep learning architecture with Bidirectional LSTM and Attention.

The main components include:

- **LSTM:** Learns temporal patterns and relationships in sequential sensor data.
- **Bidirectional LSTM (BiLSTM):** Processes sequences in both forward and backward directions to capture information from both directions.
- **Attention Mechanism:** Assigns importance to different time steps, allowing the model to focus on more relevant parts of the input sequence.
- **Classification Layer:** Uses the learned features to predict the final human activity class.

The model contains **46,982 trainable parameters**.

## 4. Experimental Results

The model achieved the following results:

| Evaluation Metric | Result |
|---|---:|
| Test Accuracy | 91.11% |
| Best Epoch | 18 |
| LAYING Class Accuracy | 100% |
| Total Model Parameters | 46,982 |

The results indicate that the model performs well in recognizing human activities.

The model achieved 100% accuracy for the LAYING activity. Some confusion was observed between SITTING and STANDING, as these activities can have similar sensor patterns.

## 5. Front-End Application

The project also includes a front-end application that provides an interface for interacting with the trained model.

The application allows users to access the activity recognition system through a graphical interface.


## 6. Conclusion

This project demonstrates the use of LSTM, Bidirectional LSTM, and Attention mechanisms for human activity recognition using sequential sensor data.

The model achieved a test accuracy of 91.11%, demonstrating its ability to identify different human activities. The results also highlight the difficulty of distinguishing between activities with similar sensor patterns, such as sitting and standing.

The project provided practical experience in data preparation, deep learning model development, model evaluation, and front-end integration.
