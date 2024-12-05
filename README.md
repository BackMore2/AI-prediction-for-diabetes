# AI-prediction-for-diabetes
Artificial intelligence final homework, using a vatiety of models to predict and comprehensive application, evaluation of various models

The following content is all translated using machine translation
1. Analysis purpose
The machine learning model can identify high-risk groups in advance based on a variety of risk factors (such as age, body mass index, BMI, family history, etc.), so as to achieve early warning of diabetes.

2.Dataset Description
*The dataset is located in data.csv under the matser branch
Data Overview: The data contains 22 features, of which target is the predicted target. The following are the preliminary classified features

![QQ20241205-171504](https://github.com/user-attachments/assets/a8131cf8-197a-431e-8bd7-5d4bf87e5a7d)

![QQ20241205-171549](https://github.com/user-attachments/assets/68f2c5e0-449e-45a9-a7ae-da3e7e05932a)


FEAT_COLS=["HighBP","HighChol","CholCheck","BMI","Smoker","Stroke","HeartDiseaseorAttack","PhysActivity","Fruits","Veggies","HvyAlcoholConsump","AnyHealthcare","NoDocbcCost","GenHlth","MentHlth","PhysHlth","DiffWalk","Sex","Age","Education","Income"]
![plot_2024-12-05 10-52-13_1](https://github.com/user-attachments/assets/42cb22b7-80fb-4441-ba5f-1ae6c58f97f0)
![plot_2024-12-05 10-52-13_2](https://github.com/user-attachments/assets/6799aeb1-5766-46ce-abb1-6309642e637d)
![plot_2024-12-05 10-52-13_3](https://github.com/user-attachments/assets/9c2aeae6-ae1f-4167-90c0-233c6eaf1525)


The above three images are histograms and visual correlation matrices obtained after EDA analysis of the data

3. Model selection and theoretical analysis
KNN: Its simplicity, flexibility and good ability to deal with nonlinear relationships are suitable for diabetes prediction
Linear regression: It is not suitable for classification problems, but can train an independent model for each category.
Logistic regression: it can well adapt to the classification problem of diabetes prediction, and also supports probability estimation
LightGBM: Fast Iteration and Large Scale Data Processing
Random Forest: It has unique advantages in handling nonlinear relationships, high-dimensional data, and preventing overfitting.

4. Experimental design
Data segmentation: using the train_test_stpile method in sklearn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=10)
Evaluation criteria: F1 score
Model configuration:
KNN: Cross validation
param_grid = {
N'neighbors': list (range (1,7)), # try different k values
'weights': ['uniform', 'distance'], # Try two weight schemes
'p': [1,2] # Try Manhattan distance (p=1) and Euclidean distance (p=2)}
Linear regression: using default parameters
Logistic regression: default parameters
LightGBM: Cross validation adjustment parameters
params = {
'boosting_type': 'gbdt',
'objective': 'binary',
'metric': 'binary_logloss',
'num_leaves': 31,
'learning_rate': 0.05,
'feature_fraction': 0.9}
Random forest uses default parameters
***Details can be found in the Python file

5. Model performance comparison
KNN model:
The optimal parameters after cross validation are as follows: {'n'neighbors': 2, 'p': 2, 'weights':'distance'}

![图片1](https://github.com/user-attachments/assets/aa53b028-0bcc-434e-8dcd-98cff1ca68c5)

The retrained model obtained F1 score of 0.716, with parameter k value of 2, using Euclidean distance and weight of distance

Linear regression: 
The model is too poor for three class classification problems, and the effectiveness of linear regression is far less than that of logistic regression. Only the method of sub parametric regression can be used to solve the 3-classification problem. For details, refer to the linear regression results in the generated dataset. Linear regression models have very low performance in dealing with datasets containing multiple nonlinear relationships

![图片2](https://github.com/user-attachments/assets/a8743071-51f0-481a-843d-3a779a9412a0)

The F1 score obtained by this method is 0.304

Logistic regression: 
Under default parameter conditions, F1 score is very low

![图片3](https://github.com/user-attachments/assets/8a551b0d-cfc7-49a2-9675-56fc9e67923c)

However, since logistic regression can reduce principal components through PCA

![图片4](https://github.com/user-attachments/assets/a20824f9-b232-4e53-86a2-0c59cc78d1f4)
![图片5](https://github.com/user-attachments/assets/377c259b-a8ba-49a9-8f8f-350e2df240df)

The F1 score obtained is 0.80, and the accuracy is also not low

LightGBM: 
The F1 score of this model is also not low, and the generated chart indicates that HighBP is the main factor
Its F1 score is 0.77

![图片6](https://github.com/user-attachments/assets/dace1342-3384-485a-9369-3f5d2da548f8)
![图片7](https://github.com/user-attachments/assets/bf142f2c-f685-4d96-b78a-5016d61fe815)

Random Forest: F1 score 0.85, accuracy as high as 0.87, belonging to the best fitting effect among these five models

![图片8](https://github.com/user-attachments/assets/2b1634f2-99c2-48a5-bf1c-9a797311e81f)
![图片9](https://github.com/user-attachments/assets/33c8bb32-4ca6-41c1-ad6d-dd2f6c176e3b)

6. Result analysis
As far as the results of the five models are concerned, it can be seen that linear regression is not suitable for such problems, while the F1 scores of random forest and logical regression are both greater than 0.8. It is not difficult to see that the fitting effect of these two models is the best, and random forest can effectively capture the nonlinear relationship in data, which is particularly important for the prediction of complex health conditions such as diabetes. Because the relationship between factors affecting diabetes (such as diet, exercise, genetic factors, etc.) and disease is often non-linear.

7. Conclusion
For the prediction of diabetes, it is clear from the results of multiple models that the accuracy of linear regression is obviously very inefficient (F1 score is only 0.3) when dealing with nonlinear diseases. For random forests and logical regression, these data without linear relationship can be better fitted.

翻译非常粗暴，实在看不懂浏览器机翻回去也看的明白（嘻嘻
