# Spambase Classification using Gaussian Naive Bayes Algorithm

## Description:
This project is based on Gaussian Naïve Bayes algorithm. Here, we try to classify the spambase data with the help of Gaussian Naïve Bayes and Logical regression model. The word “spam” concept here is diverse which refers to various advertisements for products/websites, fast money schemes, chain letters or porn etc.

The spambase data mentioned here consists of a collection of spam and non-spam mails which are identified with labels 1 and 0 respectively. The spambase dataset consists of 4601 instances in total which are then split into 50% train data and remaining 50% as test data. Both sets have 2300 instances each with 40% spam and 60% non-spam mails. We determine that the prior probability of spam as 40% and non-spam as 60%.

We try to calculate the mean and standard deviation here for both spam and non-spam data based on the 57 attributes in the train dataset. Also, we change the standard deviation is changed to 0.0001 whenever it is encountered as 0 so as to avoid division by zero error thereby assigning it a minimal standard deviation. Then, we use the Gaussian Naïve Bayes algorithm to obtain the required probabilities.


## Results:
![Output](/images/Result.JPG)

It was found that both the train and test data consisted of approximately 40% spam and 60% non-spam data. 

The accuracy obtained is 83.83%. 

With the help of confusion matrix, it was found that there were 372(317+55) mails which were classified incorrectly. 

Precision and Recall were also calculated from the confusion matrix. Although Gaussian Naïve Bayes takes less time to train the data, accuracy is not that great.
