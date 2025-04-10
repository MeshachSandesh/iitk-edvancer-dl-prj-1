### Project 1 version 1.0

There are 3 files 1) `Questions.csv` 2) `Answers.csv` 3) `Tags.csv`

- I will not be using Answers, because we can label questions directly,
- The Tags dataset is structured with multiple rows per question ID, since each question can have multiple associated tags.

## ✅ Steps I am following:

| **Stage**                                            | **Description**                                         | **Note**                                                                                  |
| ---------------------------------------------------- | ------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| **1. Data Cleaning & Preprocessing**                 | Clean, tokenize, pad sequences, encode labels           | Feature preparation                                                                       |
| **2. Baseline Model (TF-IDF + Logistic Regression)** | Train and evaluate a baseline model using classical ML  | This is optional, Just making sure the piple line is good                                 |
| **3. Model Training (Initial - Deep Learning)**      | Train LSTM/NN model on train/val split                  | Training the model on a single split                                                      |
| **4. Threshold Tuning**                              | Find best threshold per class to binarize probabilities | Here I get the best threshold for each class, I will be using it later on the final model |
| **5. Cross-Validation**                              | Independently verify model stability across folds       | This is optional: doing it to get an ideal on the fitting                                 |
| **6. Final Model Training**                          | Train model on**full data** (after cleaning)            | Training the data on the complete set                                                     |
| **7. Final Evaluation**                              | Apply`best_thresholds` from Stage 4 on predictions      | Evaluating the data with the best threshold for each class                                |

NOTE:

- Stage 1,3, and 4 are connected.
  - Stage 3 trains a model on a single train test split
  - Stage 4 uses the model in stage 3 to arrive at the best threshold for each class
- Stage 2 is an optional ML model for sanity check
- Stage 5 (Cross-Validation) (sanity check) is only for my confidence that the model is working fine ( this is optional and no learning from the above stages are used here neither is the learning from here is passed into the next stages)
- Stage 6 (Final model training) uses the stage 1 (data cleaning) to train on the complete data
- Stage 7 (Final Evaluation) is post-processing where we classify the model output to positive or negative using the Best Threshold we come to in Stage 4

# Stage 1: Data Cleaning & Pre processing

---

## DataPreperation-1-merge.ipynb

In this file, I am removing all the unwanted columns and merging the tags and questions together and creating a single data. I'm downloading this data into a file named `final_merged_data.csv`

## DataPreparation-2-clean.ipynb

In this file, I am importing `final_merged_data.csv` and cleaning the database

- converting to lowercase
- Removing special character keeping some that is important for codes
- Removing html tags
- Saving the file as `"final_data_cleaned.csv"`

## DataPreparation_3_text_procession.ipynb

In this I am

- Splitting the sentence into words
- Removing stopwords
- Lemmatizing remaining words
- Joining back to string
- Saving the final cleaned data as `"final_preprocessed.csv"`

## Exploratory_Data_Analysis_EDA.ipynb

In this I am:

- Getting Frequency of Top Tags
- Distributing Number of Tags Per Question
- Creating a Wordcloud of Tags
- Taking a look at Sample Questions per Tag

## DataPreparation_4_filter_top_tags.ipynb

Create a csv with top 10 frequent tags only

- The Tags are in json format - will convert them to python list
- Getting the top 10 frequent tags
- Deleting data of all other tags and only keep the top 10
- Saving the data into as `top_10_filtered_data.csv`

## BinarizationVectorization.ipynb

In this I am using "Multi-label binarization".

_Note for me: One-hot encoding assigns one class per sample, while multi-label binarization allows multiple classes per sample using a binary vector_

- importing `top_10_filtered_data.csv`
- Converting ["Tags"] which is string like JSON to python list
- Converting ["Tags"] to Multi-label binarization (one hot encoding with multiple values in one row)
- Removing the original ["Tags"]
- Saving the data as `top_10_tags_encoded.csv `

  The output of the file is now like:

  | clean_text                                    | android | c#  | c++ | html | ios | java | javascript | jquery | php | python |
  | --------------------------------------------- | ------- | --- | --- | ---- | --- | ---- | ---------- | ------ | --- | ------ |
  | adding scripting functionality ... language . | 0       | 1   | 0   | 0    | 0   | 0    | 0          | 0      | 0   | 0      |
  | use nested class case ... think issue .       | 0       | 0   | 1   | 0    | 0   | 0    | 0          | 0      | 0   | 0      |

# Stage 2: Baseline Model (sanity check)

---

## model_1_tfidf_logreg.py

I will be trying TFIDF, and Logistic Regression only to make sure the data pipeline is good for training, I will be using a LSTM model for the actual training later.

- Importing `top_10_tags_encoded.csv`
- TFIDF encoding "clean_text" (which is question and title mixed)
- Training test split
- Logistic regression with MultiOutput
- Prediction
- Evaluation

Output

| Label            | Precision | Recall | F1-score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| android          | 0.98      | 0.89   | 0.93     | 18069   |
| c#               | 0.92      | 0.77   | 0.84     | 20236   |
| c++              | 0.93      | 0.75   | 0.83     | 9346    |
| html             | 0.71      | 0.49   | 0.58     | 11722   |
| ios              | 0.96      | 0.85   | 0.90     | 9397    |
| java             | 0.91      | 0.75   | 0.82     | 22921   |
| javascript       | 0.84      | 0.70   | 0.76     | 24903   |
| jquery           | 0.87      | 0.72   | 0.79     | 15625   |
| php              | 0.95      | 0.85   | 0.89     | 20055   |
| python           | 0.98      | 0.88   | 0.93     | 12886   |
| **micro avg**    | 0.91      | 0.77   | 0.83     | 165160  |
| **macro avg**    | 0.90      | 0.76   | 0.83     | 165160  |
| **weighted avg** | 0.90      | 0.77   | 0.83     | 165160  |
| **samples avg**  | 0.82      | 0.80   | 0.80     | 165160  |

The score is very good :) ❤️

---

# `Moving to colab:`

The script just crashed on my local machine as soon as the training began (ran out of memory)
So from this point on wards I have moved the codes to Google Colab

---

# Stage 3: Model Training (Initial - Deep Learning)

---

## ModelTraining-Initial.ipynb

In this I will use LSTM

- Importing `top_10_tags_encoded.csv`
- Tokenization & Padding
- Train test split
- Prepareing the LSDM model
- Training the model (this crashed my system - will use Colab)
- Saveing the model as `lstm_multilabel_model.h5`

Assessment of the epoch's output:

- Val Accuracy: ~84.3% → not increasing much after Epoch 2.
- Val Loss: plateaued around 0.0740.
- Training Loss: still improving, but gap is growing → suggests slight over-fitting may be starting.

Will not train more epochs yet. Instead, will move on to fine-tuning & enhancements:

# Stage 4: Threshold Tuning

---

# Threshold Tuning:

ThresholdTuning.ipynb

## Prediction of validation set into probabilities

`y_pred_probs` = Predicted probabilities on validation set

```python
[[1.3742083e-01 8.0447520e-05 4.2072592e-05 2.9579774e-04 2.2695679e-04
  9.9088919e-01 3.2090952e-04 4.5822024e-05 9.0044586e-04 4.2599757e-04]
 [2.3912480e-03 2.5329317e-03 1.7092051e-04 4.7491857e-01 2.5283507e-04
  8.7853996e-03 9.4237328e-01 1.9064820e-01 4.4764105e-02 5.5983348e-04]
 [9.9907351e-01 8.6471031e-04 1.5171616e-04 1.0077984e-03 4.2597804e-04
  7.4797377e-02 7.0538226e-04 4.0577419e-04 7.1070957e-05 9.5933952e-05]
 [4.2406041e-03 1.9286251e-04 6.4736395e-04 7.1064476e-04 2.0546702e-05
  9.9795806e-01 2.6034662e-03 8.8846442e-05 9.8128425e-05 2.9508787e-04]
 [2.5365818e-05 8.6190802e-01 1.0530833e-02 1.2335751e-03 9.8581484e-04
  1.0896993e-02 8.8235073e-02 5.2736686e-03 9.8816614e-05 2.0790767e-02]]
```

Note: The output of y_pred_probs looks like it is ranging from `1 to 10` when it is suppose to be `0 to 1`. that is because the output is written in `scientific notation` (also called exponential notation), which can look a bit cryptic at first glance.
📌 What it means:
• 1.3742083e-01 = 0.13742083
• 8.0447520e-05 = 0.000080447520
• 9.9088919e-01 = 0.99088919
• 2.5365818e-05 = 0.000025365818
So yes — these numbers are indeed between 0 and 1, just displayed using a compact format for small/large numbers.

## Best threshold

_We have y_pred_probs which is the prediction, we can apply 0.5 as the threshold and with this we can arrive at 0/1 binary predictions.
But there is something to note: Each class might perform better with its own threshold, hence a generic threshold like 0.5 across is not the best approach. So we tune the threshold individually for each class. The `best_thresholds` var here hold threshold values for each class _

`best_thresholds`: has Each class's best threshold

```python
Class 0: Best Threshold = 0.55, F1 Score = 0.9521
Class 1: Best Threshold = 0.45, F1 Score = 0.9006
Class 2: Best Threshold = 0.40, F1 Score = 0.8807
Class 3: Best Threshold = 0.35, F1 Score = 0.6546
Class 4: Best Threshold = 0.55, F1 Score = 0.9403
Class 5: Best Threshold = 0.55, F1 Score = 0.8727
Class 6: Best Threshold = 0.45, F1 Score = 0.8136
Class 7: Best Threshold = 0.50, F1 Score = 0.8208
Class 8: Best Threshold = 0.45, F1 Score = 0.9245
Class 9: Best Threshold = 0.60, F1 Score = 0.9549
```

## Binary prediction

_Now that we have the prediction and best threshold values, we use the `best_thresholds` to arrive at the final predictions from the probabilities `y_pred_probs`_

`y_pred_binary`: hold the final prediction in binary 0/1 (yes/no)

```python
array([[0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [1., 0., 0., ..., 0., 0., 0.],
       ...,
       [0., 1., 0., ..., 0., 0., 0.],
       [0., 1., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.]], dtype=float32)
```

---

# `Moving to Kaggle:`

Colab would take 8 hours and then disconnect, this happened 3 times over 3 days and hence I moved to Kaggle

---

# Stage 5: Cross-Validation

## kaggle_cross_validation.ipynb

This is not a required step, I am only doing it for sanity check
✅ Average F1 Score: 0.8530

---

# Stage 6: Final Model Training

## p1-finalmodeltraining.ipynb

- I have trained the completed data set on LSTM
- Saving the model as final_model.h5

---

# Stage 7: Final Evaluation

### `overly optimistic performance metrics`

`Since I am evaluating the final model on the same data it was trained on, I will likely get overly optimistic performance metrics.`

FinalEvaluation.ipynb

- Load the trained model
- Load the same data (top 10 classes)
- Predicting the same data with the trained model
- Producing a classification report with using the best threshold found earlier

| Label            | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| javascript       | 0.98      | 0.95   | 0.97     | 90,588  |
| java             | 0.93      | 0.95   | 0.94     | 101,077 |
| c#               | 0.96      | 0.92   | 0.94     | 47,544  |
| php              | 0.72      | 0.68   | 0.70     | 58,970  |
| android          | 0.98      | 0.95   | 0.97     | 46,976  |
| jquery           | 0.97      | 0.84   | 0.90     | 115,011 |
| python           | 0.86      | 0.82   | 0.84     | 124,071 |
| html             | 0.86      | 0.84   | 0.85     | 78,526  |
| c++              | 0.95      | 0.95   | 0.95     | 98,750  |
| ios              | 0.99      | 0.97   | 0.98     | 64,553  |
| **Micro Avg**    | 0.92      | 0.89   | 0.90     | 826,066 |
| **Macro Avg**    | 0.92      | 0.89   | 0.90     | 826,066 |
| **Weighted Avg** | 0.92      | 0.89   | 0.90     | 826,066 |
| **Samples Avg**  | 0.93      | 0.92   | 0.92     | 826,066 |

---
