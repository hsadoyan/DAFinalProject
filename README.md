# Predicting Song Positivity Based on Audio Metaparameters

#### by Harry Sadoyan, Jacob Henry, and Emily Bigwood

We set out to find if there is a relationship between various meta-features of a song and how happy or positive it is. We obtained the dataset from Spotify, which keeps track of a number of features of each song in their database, including but not limited to loudness, duration, speechiness, and energy. They also keep track of the "valence", or positivity, of each song, as a decimal between 0 and 1.

We tried a number of machine learning techniques including Linear Regression, kNN, Random Forests, and SVM, paired with PCA and LASSO for feature engineering.

## Results

Once we determined the optimal parameters for each of our models, we ran each of our machine learning techniques on our entire training set to see which would be the most effective. The table below shows the testing errors for each model. Support Vector Machines had the lowest training error of 0.0690, so we decided to test our vaulted data using SVM.

![](https://i.imgur.com/nFKG400.png)
