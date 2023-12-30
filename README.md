# Credit-Card-Fraud-Detection

The crucial problem of locating fraudulent transactions in credit card data is what the Credit Card Fraud Detection project aims to solve. The main objective is to create a machine learning model that would enable credit card companies to better safeguard their clients from unauthorized charges by accurately differentiating between fraudulent and lawful transactions.

## Installation

To run this project, you need to install the required dependencies. Use the following command:

```bash
pip install -r requirements.txt
```
## Context

The difficulty for credit card firms is identifying fraudulent transactions so that consumers are not charged for purchases they did not make. Ensuring the security of user accounts and preserving the integrity of financial transactions depend heavily on fraud detection.

## Dataset
The project makes use of a dataset that includes September 2013 credit card transactions done by cardholders throughout Europe. Out of 284,807 transactions, 492 instances of fraudulent transactions are included in the dataset. Only 0.172% of the total transactions are fraudulent, indicating a significant imbalance in the statistics.

The dataset's features are numerical and the outcome of a modification using Principal Component Analysis (PCA). Regretfully, due to confidentiality considerations, the original features and more background are not made available. The transaction value, the amount of time since the first transaction, and the PCA-transformed components are important characteristics.

The source of the dataset was Kaggle.

You can collect dataset from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Project Workflow


1.#### Data Inspection: #### A preliminary look at the structure, numbering, and basic statistics of the dataset.

2.#### Managing Duplicate Values: #### Locating and eliminating duplicate data points from the collection.

3.#### Managing Null Values: #### Examining the dataset for any missing values and taking appropriate action.

4.#### Feature engineering : #### Feature involving involves investigating the connections between features, concentrating on 'Amount' and 'Time'.

5. #### Exploratory Data Analysis (EDA): #### Investigating patterns in time versus quantity, assessing transaction amounts, and visualizing the distribution of legitimate and fraudulent transactions.

6.#### Multicollinearity Analysis: #### Examining how independent variables are correlated.

7.#### Implementing Machine Learning Models: #### To train and assess machine learning models such as XGBoost, Local Outlier Factor (LOF), Quadratic Discriminant Analysis (QDA), and Isolation Forest, use the PyCaret package.

8.#### Evaluation Metrics: #### Prioritizing recall due to the imbalanced nature of the dataset and discussing the trade-offs between false positives and false negatives.

9.#### Final Evaluation on Test Dataset: #### Assessing the model's performance on a separate test dataset.


## Results

The study evaluates the effectiveness of several anomaly detection methods and machine learning models. A number of criteria, including accuracy, precision, recall, and F1 score, are used to assess the selected models. One technique that stands out as being especially good at finding anomalies is the XGBoost.

## Conclusion

The experiment sheds light on how well different approaches identify credit card fraud work. It highlights how crucial it is to take into account the particular needs and repercussions of false positives and false negatives in fraud detection scenarios.

