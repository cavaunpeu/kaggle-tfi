## [Tab Food Investments Restaurant Revenue Prediction](https://www.kaggle.com/c/restaurant-revenue-prediction)

#### Summary
* The contest offers a training set of 137 observations (yes), and a testing set of 100,000. All models will overfit - the goal is to not overfit too much. 
* I experimented primarily with linear models. Using Gradient Boosted Trees / Random Forests seemed cavalier; these capture more complex variable interactions, and therefore lead to more overfitting.
* I tried several variable selection techniques, with RandomizedLasso being most useful. To quote sklearn's documentation: "The limitation of L1-based sparse models is that faced with a group of very correlated features, they will select only one. To mitigate this problem, it is possible to use randomization techniques, reestimating the sparse model many times perturbing the design matrix or sub-sampling data and counting how many times a given regressor is selected."
* With such a small training set, it was possible to thoroughly understand the variable space. Univariate regression with raw, transformed, and polynomial features played heavily into the modeling process.
* To the organizers, and anyone looking to produce a better model, it's really quite simple: get more data!

#### To train models and generate a submission, run the following on the command line:

```bash
python create_submission.py <path_to_training_set> <path_to_test_set> <path_to_sample_submission> <path_to_your_submission>
```
