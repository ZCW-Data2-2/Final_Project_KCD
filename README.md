# Tweet Sentiment Analysis
## Sentiment on Mask Mandates
Capstone project for ZCW Data's course.

This group consists of:

Keerthi Balla: https://github.com/ZCW-Data2-2/Final_Project_KCD/tree/keerthiballa

Drake Dwornik: https://github.com/ZCW-Data2-2/Final_Project_KCD/tree/drake

Creasen Naicker: https://github.com/ZCW-Data2-2/Final_Project_KCD/tree/creasen

This application will run a query against Twitter using the Twitter API. We are using the search topic "Mask Mandate" for our analysis. From there it will utilize Apache Kafka and be sent to a producer. From the producer it will make the queries and send messages to the database as well as a kafka Topic. From there the tweet will be sent to a Kafka consumer where the trained model (logistic regression) will take one tweet and determine whether its positive, negative, or neutral. From there the tweet will be sent to the stored database. From the database it will be sent to AWS Quicksite for data visualization.

![Dataflow_Final_Project_2](https://user-images.githubusercontent.com/92214453/150574561-e5d69c02-de1b-42dc-a7e3-1785db5eb866.png)

([live document](https://docs.google.com/presentation/d/1pXJSsQBkr6xXI2dluPxIpyOxXZqf_O7f65USWllmJAk/edit?usp=sharing))

[Project board link](https://github.com/ZCW-Data2-2/Final_Project_KCD/projects/1)



## Algorithms Tested
1. Logistic Regression

2. Bernoulli Naive Bayes

3. Support Vector Machine

4. XGBoost


