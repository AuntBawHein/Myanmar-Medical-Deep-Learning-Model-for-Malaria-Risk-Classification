 Myanmar-Medical-Deep-Learning-Model-for-Malaria-Risk-Classification
Myanmar Medical Deep Learning Model for Malaria Risk Classification for Data Science Project

Step 1: Defining Project Scope/Objectives
I'm going to create a Deep Learning model to help predict if someone might get malaria. The model will use details like a person's age, gender, lab test results, 
travel history, and doctor's diagnosis to make predictions.

 Objective:
Predict the chances of getting malaria based on:
Age: How old the person is.
Sex: Whether they are male or female.
Lab test results: If malaria is found in their blood tests.
Travel history: If they traveled to places where malaria is common.
Diagnosis: What the doctor found after checking the patient.

Tools Use:
TensorFlow: To build the deep learning model.
Pandas: clean the unwanted and duplicated data 
Seaborn & Matplotlib: To make charts and graphs.
Scikit-learn: To check how well the model works.

Step 2: Data Collection and Exploration
I’m going to load the data into a table and explore it. I will look at output results like age, gender, lab test results, travel history, and whether Doctor detects any positive or negative 
pattern.

Objective:
I will load the data using Pandas (a tool that helps organize and clean the data).
I will explore the data to see how Age, Sex, Lab results, and Travel History are spread out.
I will create charts using Seaborn and Matplotlib (tools to make graphs) to show any patterns in the data.
I will clean the data if there are any missing or wrong values.

Columns I'm Looking At:
Age: How old the person is.
Sex: Whether the person is male or female.
Lab test results: Whether the person has malaria or not.
Travel history: If the person has traveled to places where malaria is common.
Diagnosis: What the doctor says about the person’s health.

Tools I'll Use:
Pandas: To load and organize the data into a table.
Seaborn & Matplotlib: To create charts that show patterns in the data.

Why do I have to use Pandas, Seaborn, & Matplotlib for this step?
Pandas: Helps me load the data into a table and clean it so it's easy to work with.
Seaborn and Matplotlib: Help me make graphs so I can easily see any patterns in the data.

![distribution_of_ages_malaria_cases_Myanmar_1](https://github.com/user-attachments/assets/81248cc1-5779-4370-9f16-403cc81470a6)
![sex_male_female_malaria_patients_2](https://github.com/user-attachments/assets/b1dec30e-fc09-4911-b332-cd4199184b8e)
![travel_history_malaria_patients_Myanmar_3](https://github.com/user-attachments/assets/1af3537a-3402-42df-a4c9-e3e5abf4590b)
![lab_test_results_malaria_travel_history_Myanmar_4](https://github.com/user-attachments/assets/f651cf45-8b9a-4297-a786-41c9b9e584da)


Step 3: Data Preprocessing and Cleaning
In this step, I’m going to get the data ready so the computer can use it. This means fixing it up and making it easier for the computer to understand.

Objective:
Fixing missing values: I will find and fill in or remove any missing data (like if someone's age or travel history is missing).
Normalize numbers: I will make sure all the numbers, like age, are on the same scale so the computer can compare them easily.
Change words into numbers: I will turn words like "Male" and "Female" into numbers because the computer understands numbers better.
Split the data: I will divide the data into two groups: one group to train (teach) the computer and another group to test (see if the computer learned correctly).

Columns I'm Looking At:
Age (in Yr): How old the person is.
Sex: Whether the person is male or female.
Lab result: Whether the person has malaria or not.
Traveling History: If the person has traveled to places where malaria is common.
Dx: What the doctor says about the person’s condition.

Libraries I'll Use:
Pandas: To help clean and organize the data.
Scikit-learn: To split the data and make sure all the numbers are on the same scale.


Why is this step important?
Cleaning the data makes sure there are no mistakes, and turning everything into numbers helps the computer understand the data better.

Why do I have to use Pandas and Scikit-learn for this step?
Pandas: I’m going to use Pandas because it helps clean and organize messy data.
Scikit-learn: I’m going to use Scikit-learn because it helps me split the data and make sure the numbers are fair when teaching the computer.

Step 4: Feature Engineering
In this step, I’m going to create new features (pieces of data) that might help the computer learn better.

Objective:
I’m going to create new features. For example, I can combine "Traveling History" and "Lab result" to see if people who traveled are more likely to have 
malaria. This will help the computer make better predictions.

Columns:
Age (in Yr): How old the person is.
Sex: Whether the person is male or female.
Lab result: Whether the person has malaria or not.
Traveling History: If the person has traveled to places where malaria is common.

Libraries:
Pandas: I’m going to use this to create and organize the new features in a table-like structure.
Scikit-learn: I’m going to use this to turn the features into numbers that the computer can understand better.

Why is this step important?
Creating new features helps the computer learn more from the data, which can make its predictions more accurate.

Why do I have to use Pandas and Scikit-learn?
I’m going to use Pandas to organize and manage the data in tables.
I’m going to use Scikit-learn to turn the data into numbers so the computer can learn from it.

Step 5: Model Building Using TensorFlow

In this step, I’m going to teach the computer how to predict malaria using a tool called TensorFlow.

What I’m going to do:
[malaria_cases_as_csv_file.csv](https://github.com/user-attachments/files/19367352/malaria_cases_as_csv_file.csv)

I’m going to create a model: This model is like a smart system that learns from the data and tries to guess (predict) whether someone has malaria or not.
I’m going to train the model: This means the model will look at examples from the data and learn patterns like who gets malaria and who doesn't.
I’m going to test the model: After training, I will test how well the model learned by giving it new data (data it hasn’t seen before) to make predictions.

Why is this step important?
Building and training the model is important because it helps the computer make smart guesses about who might have malaria in the future.

What are Pandas, Numpy, TensorFlow, and scikit-learn?

Pandas: I’m going to use Pandas to organize and clean the data. It’s like sorting your homework to make sure everything is neat and easy to find.
Numpy: I’m going to use Numpy to help the computer do Math and make calculations faster and easier.
TensorFlow: I’m going to use TensorFlow to build and train the model. This is the tool that teaches the computer how to predict things, like whether 
someone has malaria.
Scikit-learn: I’m going to use Scikit-learn to split the data into training and testing sets. This way, the model can learn from one part of the data and be
 tested on the other to see how well it learned.



Why do I have to use these tools?

I need Pandas to organize and clean the data.
I need Numpy to help with math.
I need TensorFlow to teach the computer how to predict.
I need Scikit-learn to test the model and see how well it learned.

Step 6: Model Building Using PyTorch

In this step, I'm going to build a new model using PyTorch to predict malaria cases.

What I'm going to do:

I'm going to create a model: The model will learn from the data to predict if someone has malaria, similar to the TensorFlow model, but this time using a 
tool called PyTorch.
I'm going to train the model: The model will look at examples from the data and learn patterns about who has malaria and who doesn't.
I'm going to test the model: After training, I'll check how well the PyTorch model learned by testing it with new data it hasn't seen before.
Why use PyTorch: PyTorch is another tool that helps us build models. We will compare how well this PyTorch model performs compared to the 
TensorFlow model.

In simple words: I'll use PyTorch to build and train another model that predicts malaria cases, just like we did with TensorFlow, and then we'll see how well it
 works!

What are Pandas, PyTorch, Scikit-learn, Numpy, Matplotlib, and Seaborn?

Pandas: I use Pandas to organize and clean the data. It helps me to clean and remove unwanted data as well as duplicated information in the table.
PyTorch: I use PyTorch to build the model that will learn from the data.
Scikit-learn: I use Scikit-learn to split the data into training and testing sets, and to scale the data so the model can learn better.
Numpy: I use Numpy to help with numbers and mathematical operations, like creating random values.
Matplotlib: I use Matplotlib to create graphs and visualizations to better understand the data.
Seaborn: I use Seaborn to make heatmaps and other easy-to-read graphs.


Why do I need these tools?

I need Pandas to organize and clean the data before giving it to the model.
I need PyTorch to build and train the model so it can learn and make predictions.
I need Scikit-learn to split the data and scale it, so the model can learn from it properly.
I need Numpy to help with numbers and math calculations.
I need Matplotlib and Seaborn to create graphs that help me understand the data and the results.

Step 7: Model Evaluation with Scikit-learn

In this step, I'm going to evaluate how well my model predicts malaria cases by checking how accurate it is and how well it identifies who has malaria and 
who doesn't.

What I'm going to do:
I'm going to measure accuracy: I'll check if the model correctly predicted whether someone has malaria or not by comparing its predictions to the actual 
results.
I'm going to use Scikit-learn for evaluation: I'll also create charts, like confusion matrices and ROC (Receiver Operating Characteristic) curves, to 
visually see how good the model is at making predictions.

In simple words:
I'm going to use Scikit-learn to measure how well my model works at predicting malaria cases and create some charts to show how well it performs.

What is Scikit-learn, Seaborn, Matplotlib, Numpy, Pandas, and PyTorch?

Scikit-learn: I use this tool to evaluate how well my model predicts. It gives me numbers and scores to see if the model is good at its job.
Seaborn: I use this tool to make charts that help me better understand how the model is performing.
Matplotlib: I use this tool to plot (draw) graphs that make it easier to see if the model is making good or bad predictions.
Numpy: I use this tool to handle arrays and large datasets.
Pandas: I use this tool to organize and clean the data.
PyTorch: I use this tool to build and train the model that predicts whether someone has malaria.

Why do I have to use Scikit-learn, Seaborn, Matplotlib, Numpy, Pandas, and PyTorch for this step?

I need Scikit-learn to measure how accurate my model is at predicting.
I need Seaborn to help make charts that explain the model's performance in a visual way.
I need Matplotlib to plot graphs that clearly show how good or bad the model is at predicting malaria cases.
I need Numpy to work with numbers and large datasets efficiently.
I need Pandas to organize and analyze my data in a simple format.
I need PyTorch to create a model that learns to predict malaria based on the data.

Step 8: Hyperparameter Tuning

What I'm going to do:
I'm going to find the best settings for my model, like adjusting how fast it learns (learning rate), how much data it looks at once (batch size), and other 
important settings.
I will use a tool called GridSearchCV or RandomizedSearchCV from Scikit-learn. These tools will try different settings to see which ones make the 
model performed better.

Why I'm doing this:
I want my model to get better at predicting. By changing these settings (called hyperparameters), I can improve how well my model works and make 
more accurate predictions.

In simple words:
I'm going to change some settings in my model to help it learn better and guess more accurately. I will use special tools to test different settings and 
choose the best ones.

What is TensorFlow, PyTorch, Scikit-learn?

TensorFlow: A tool used to build and train AI models.
PyTorch: Another tool used to build and train models.
Scikit-learn: A tool that helps split data, tune settings (hyperparameters), and check how well the model is performing.

Why do I have to use TensorFlow, PyTorch, and Scikit-learn for this step?

I use TensorFlow and PyTorch to build and train the model.
I use Scikit-learn to tune the settings (hyperparameters) and evaluate how well the model is performing.

![output_for_best_hyper_9](https://github.com/user-attachments/assets/32aad5e2-d452-44e2-995f-276c61fed1a9)


Step 9: Final Model Performance Evaluation

What I'm going to do:

I'm going to compare the performance of both the TensorFlow and PyTorch models to see which one predicts malaria better.
I will check how good each model is at predicting malaria by looking at key evaluation metrics like accuracy (how often it predicts correctly), 
precision (how many times it correctly predicted malaria), recall (how well it finds actual malaria cases), F1-score (a mix of precision and recall), and the
 confusion matrix (a table showing correct and incorrect predictions).
I will also create charts (like bar charts and heatmaps) to show how well each model performs visually.

In simple words:
I’m going to check how well the TensorFlow and PyTorch models work by looking at important numbers like how often they predict correctly and how 
many mistakes they make.
I will also make charts to easily compare the results of both models and decide which one is better at predicting malaria.

This step will help us clearly see which model (TensorFlow or PyTorch) performs better at predicting malaria, and we can make a final decision based on these 
results.

What is TensorFlow, PyTorch, Scikit-learn, Matplotlib, Seaborn?
TensorFlow: A powerful tool used to build and train AI models (like the ones we’re using to predict malaria).
PyTorch: Another tool, like TensorFlow, that helps us build and train models for predicting malaria.
Scikit-learn: This tool helps us split the data, train models, and calculate important numbers to check how well the models are doing.
Matplotlib: A tool for making charts and graphs so we can see the results clearly.
Seaborn: A tool that helps us make nicer-looking charts and graphs, making it easier to understand the results.

Why do we have to use TensorFlow, PyTorch, Scikit-learn, Matplotlib, and Seaborn?
We use TensorFlow and PyTorch to build, train, and test models that predict whether someone has malaria or not.
We use Scikit-learn to split the data, train the models, and calculate important numbers (like accuracy and precision) to check how well the models 
perform.
We use Matplotlib and Seaborn to create charts and graphs to help us understand the results better by seeing them visually.

![Tensorflow_conf_matrix_Myanmar_7](https://github.com/user-attachments/assets/ea66e858-bb03-49ef-b93d-ae3ecb375108)
![Pytorch_conf_matrix_Myanmar_8](https://github.com/user-attachments/assets/e253d7f9-5c67-4078-b7bb-10f079abae47)


Step 10: Asking and Answering Questions about the Data Science Project on Myanmar Medical Deep Learning Model for Malaria Risk Classification

In this step, I’m going to ask and answer 11 real questions about my project on malaria risk classification using machine learning models. These questions 
will help me explain my project clearly and simply.

1.Why did you choose malaria prediction as your project?

I chose this project because malaria is a serious health issue, especially in Myanmar. Using AI models to predict malaria cases can help identify cases early 
and save lives.
2.What types of models did you use for this project?

I used two machine learning models: TensorFlow and PyTorch. Both are powerful tools that help build smart models to predict malaria.
3.Which model do you use the most?

I use TensorFlow more often because it is beginner-friendly and easy to work with. However, I used PyTorch in this project to compare how they both 
perform.
4.What kind of data did you use to train your models?

I used data that includes a patient's age, gender, travel history, lab results, and diagnosis information related to malaria.
5.How did you prepare the data for the models?

I cleaned the data by fixing any missing values, scaling the features (like age) so the models could understand them better, and splitting the data into 


6.Why did you compare two models instead of just one?

I wanted to see which model, TensorFlow or PyTorch, performs better at predicting malaria. Comparing models helps me choose the most effective one 
for this specific task.
7.What are the benefits of using machine learning to predict diseases like malaria?

AI can quickly analyze data and predict which patients are at risk of malaria, allowing doctors to act faster and provide treatments in areas where medical 
resources are limited.

8.Did you face any difficulties while building the models?

Yes, I faced challenges in finding the best settings (like learning rate) for the models and handling incomplete or inconsistent data.

9.Which model performed better in your project?

In my project, the TensorFlow model had better overall accuracy, but the PyTorch model did a great job in detecting more real malaria cases, even though
 it made more prediction errors.
 
10.Can the models be improved further?

Yes, the models can be improved by adding more data, using advanced techniques like fine-tuning, or adjusting the number of layers and neurons to make 
the models more accurate.

11.What did you learn from this project?

I learned how to use TensorFlow and PyTorch to build machine learning models that can predict diseases like malaria. I also learned that comparing 
different models helps identify the best tool for the task.
