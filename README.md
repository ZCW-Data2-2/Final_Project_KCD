# Tweet Sentiment Analysis
## Sentiment on Mask Mandates
Capstone project for ZCW Data's course.

This group consists of:

Keerthi Balla: https://github.com/ZCW-Data2-2/Final_Project_KCD/tree/keerthiballa

Drake Dwornik: https://github.com/ZCW-Data2-2/Final_Project_KCD/tree/drake

Creasen Naicker: https://github.com/ZCW-Data2-2/Final_Project_KCD/tree/creasen

This application will run a query against Twitter using the Twitter API. We are using the search topic "Mask Mandate" for our analysis. From there it will utilize Apache Kafka and be sent to a producer. From the producer it will make the queries and send messages to the database as well as a kafka Topic. From there the tweet will be sent to a Kafka consumer where the trained model (logistic regression) will take one tweet and determine whether its positive, negative, or neutral. From there the tweet will be sent to the stored database. From the database it will be sent to AWS Quicksite for data visualization.

![Dataflow_Final_Project_1](https://user-images.githubusercontent.com/92214453/150572386-a2974e51-7135-41a5-afce-064fd00d54a6.png)

([live document](https://docs.google.com/presentation/d/1pXJSsQBkr6xXI2dluPxIpyOxXZqf_O7f65USWllmJAk/edit?usp=sharing))

[Project board link](https://github.com/ZCW-Data2-2/Final_Project_KCD/projects/1)

Phases of project:
phase 1: develop alanalysis models using Machine Learning
phase 2: deploy Kafka
Phase 3.a: deployconsumer
Phase 3.b: deploy Kafka consumer
phase 4: Develop a way to present results

Phases that can overlap some:
1 and 2
3.a 3.b starting for 4

## Algorithms Tested
??? 80%

Naive Bayes 75%
