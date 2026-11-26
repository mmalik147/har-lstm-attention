
# HARNet Assignment 4 — Results Analysis

## 1. Model Performance

The custom HARNet model achieved 91.11% test accuracy.

The best validation checkpoint was obtained at epoch 18.

- Best validation loss: 0.1102
- Best validation accuracy: 96.51%
- Test accuracy: 91.11%

## 2. Per-Class Performance

| Activity | Accuracy |
|---|---:|
| Walking | 89.11% |
| Upstairs | 98.09% |
| Downstairs | 97.14% |
| Sitting | 79.63% |
| Standing | 83.65% |
| Laying | 100.00% |

Laying achieved perfect classification. Sitting and standing were the most challenging activities.

## 3. Confusion Matrix Observations

The largest classification errors occurred between sitting and standing.

- 98 sitting windows were classified as standing.
- 87 standing windows were classified as sitting.

These activities have similar motion characteristics, which makes them difficult to distinguish.

## 4. Attention Analysis

Mean maximum attention weight per activity:

| Activity | Mean Maximum Attention |
|---|---:|
| Walking | 0.326123 |
| Upstairs | 0.225232 |
| Downstairs | 0.234500 |
| Sitting | 0.105216 |
| Standing | 0.035428 |
| Laying | 0.017623 |

Moving activities generally showed more concentrated attention, while standing and laying showed more distributed attention.

These values describe attention concentration, not classification accuracy.

The average attention plots and individual examples provide additional temporal information.


## 5. Conceptual Analysis — Task 6

### 6.1 Why split validation data by person?

The dataset windows overlap by 50%. Random splitting could place highly similar
neighbouring windows from the same person in both training and validation sets.
This can cause data leakage and overestimate model performance. Splitting by
person evaluates generalization to unseen volunteers.

### 6.2 Which activities are confused most often?

Sitting and standing are confused most often. The confusion matrix shows
98 sitting windows predicted as standing and 87 standing windows predicted
as sitting. Their motion patterns are similar when measured using a waist-worn
phone, making them difficult to distinguish.

### 6.3 Why is LAYING recognized accurately?

The total acceleration X, Y and Z signals include gravity. When a person lies
down, the phone's orientation relative to gravity becomes distinctive and
relatively stable. This helps the model identify the laying activity.

### 6.4 Is attention more peaked for moving or still activities?

Attention is generally more concentrated for moving activities in our results.
Walking has a mean maximum attention of 0.326123, compared with 0.225232 for
upstairs and 0.234500 for downstairs. Still activities generally have lower
values, although sitting is an exception compared with standing and laying.
This suggests that informative temporal patterns may be more concentrated
during movement.

### 6.5 Why is BiLSTM allowed here?

The model receives a complete 2.56-second window, so it can process the
sequence in both forward and backward directions. This is suitable for
offline activity recognition. A bidirectional model would not be suitable
for strictly real-time prediction when future sensor samples are unavailable.

### 6.6 Why does LSTM retain information better than a plain RNN?

The LSTM cell state is updated using:

c_t = f_t * c_(t-1) + i_t * g_t

The forget gate controls how much previous information is retained, while
the input gate controls how much new information is added. This gated,
additive update helps preserve information across long sequences.

Gradient clipping limits the gradient norm to 1.0 and helps prevent
exploding gradients during training.
