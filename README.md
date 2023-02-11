# predicting customer churn on SyriaTel company
## Bussiness understanding
SyriaTel is a Telecommunication company that provides data and telecomunication services. The company has recently been experiencing decrease in revenue returns which has mostly been due to customer churn.
customer churn(customer stop using the company services) has direct effect on the company revenue, the company is looking to identify means to predict customers churn and how to rectify it.
This project has been developed to identify the factors that are causing the high rise on customer churn and provides solutions on ways to mitigete it.

## Objective
The objective of this project is to develop the best model that can be used to predict causes of customer churn and how the company can mitigate them to avoid loosing its customers

### model used
* Decision Tree
* Baseline Logistic regression model
* SMOTE
* XGBoosting
* GridSearch model
### DATA

## EDA

## modeling
The model that performed better of the one evaluated was XGBoost which had an accuracy of 0.9763(97.63%) on training data and Accuracy of 0.944(94.4%) on testing data.
The model was able to identify the positive class (1) with a precision of 88%, recall of 72%, and f1-score of 79%. On the other hand, it was able to identify the negative class (0) with a precision of 95%, recall of 98%, and f1-score of 97%. Which was better than the other models
## conclusion
Customers who called more than four times left, this might be due to their concern not addressed or the feedback recieved was not helpful and didn't sort their problem and they opted to try other providers.

There were also some states that had highest number of customers who churned. this can be due to poor network coverage or were not being given good services compared to other telecommunication networks in the area.

Also the charges and minutes of day calls were contributing highly to the costomers churning out. this might be due to high charges imposed on the services provided.

## Recommendations
1. XGBoost model can be used for deployment as its the one that produced the highest performance compared to the other model
2. There is need for the company to check on the customers who called more and identify there problem first to avoid losing them
3. Company to develop a department that can deal with customer concerns quickly and to prioristise the areas that are more affected.
4. The model can also be further evaluated to improve its performance even better.
5. Need to train their employees to identify and provide feedback that are more helpful to the company customers to avoid losing them
6. Check on the charges and the day calls, provide affordable services that are standard in regards to the services they offer and compared to other companies
## limitations
