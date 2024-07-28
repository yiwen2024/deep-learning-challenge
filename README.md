# deep-learning-challenge

Background

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. 

Machine learning and neural networks were applied to analyze the dataset of a nonprofit foundation Alphabet Soup that contains historic information of more than 34,000 its funded organizations. The features listed below in the provided dataset were used to create a binary classifier to predict and select the applicants with the best chance of success. 

*EIN and NAME—Identification columns

*APPLICATION_TYPE—Alphabet Soup application type

*AFFILIATION—Affiliated sector of industry

*CLASSIFICATION—Government organization classification

*USE_CASE—Use case for funding

*ORGANIZATION—Organization type

*STATUS—Active status

*INCOME_AMT—Income classification

*SPECIAL_CONSIDERATIONS—Special considerations for application

*ASK_AMT—Funding amount requested

*IS_SUCCESSFUL—Was the money used effectively

Instructions

Step 1: Preprocess the Data

Pandas and scikit-learn’s StandardScaler() were applied to preprocess the dataset: 

* Upload the starter file to Google Colab, follow the instructions and provided file to complete the preprocessing steps.

* Read in the charity_data.csv to a Pandas DataFrame, identify the target and feature variables. 

* Determine the number of unique values for each column. Combine rare categorical variables together in a new value, 'Other', and then check if the replacement was successful.

* Use pd.get_dummies() to encode categorical variables.

* Split the preprocessed data into a features array, X, and a target array, y. Use these arrays and the train_test_split function to split the data into training and testing datasets.

* Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.

Step 2: Compile, Train, and Evaluate the Model

* TensorFlow and Keras were used to build a binary classification neural network model and the model was compiled, trained, and evaluated based on the loss and accuracy.

* The moedel was optimized by adding an additional training of 'NAME' column, as well as changing nodes, hidden layers, activation function, optimizer, and epochs.

* Saved and export your results to an HDF5 file. Name the file AlphabetSoupCharity.h5.

Step 3: Optimize the Model

Optimized the model to achieve a target predictive accuracy higher than 75% by using the following methods:

* Adjusted the input data by dropping fewer clolumn 

* Adjusted the number of nodes in each hidden layer

* Added one more hidden layer

* Tried different activation functions "Tanh", "reLu", "sigmoid", "leaky reLu" for the hidden layers and optimizers "adam", "rmsprop" for the training process

* Tried adding and reducing the number of epochs to the training regimen.

Step 4: Write a Report on the Neural Network Model

Overview of the analysis:
-The purpose of this analysis was to build a binary classification model that can predict whether an applicant will be successful if funded by Alphabet Soup. The dataset provided contained 34,000 organizations that were funded by Alphabet Soup. The dataset was split into a training dataset and a testing dataset. The training dataset was used to train the model and the testing dataset was used to evaluate the model. The model was compiled, trained, and evaluated based on the loss and accuracy. The model was optimized by adding an additional training of 'NAME' column, as well as changing nodes, hidden layers, activation function, optimizer, and epochs. The model was saved and exported to an HDF5 file.

Results: 
-The target variable is IS_SUCCESSFUL.

-The features variables are: APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT.

-The columns 'EIN' and 'NAME' are not targets nor features so they should be removed from the input data.

Original Model Results:
* Accuracy: 72.85%
* Loss: 0.5511
* Hidden layer 1: nodes=20, activation='relu'
* Hidden layer 2: nodes=10, activation='relu'
* Hidden layer 3: nodes=5, activation='sigmoid'
* Output layer: units=1, activation='sigmoid'
* Optimizer: 'adam'
* Epochs: 100

![model_accuracy](https://github.com/user-attachments/assets/77e28c40-5a37-45d2-bff7-c8c77bb54b55)

Optimized Model Results:
* Accuracy: 78.92%
* Loss: 0.4528
* Hidden layer 1: nodes=20, activation='relu'
* Hidden layer 2: nodes=10, activation='tanh'
* Hidden layer 3: nodes=5, activation='sigmoid'
* Output layer: units=1, activation='sigmoid'
* Optimizer: 'adam'
* Epochs: 100
  
![model_accuracy_opt](https://github.com/user-attachments/assets/2a95adfc-2242-48de-bd63-f155df7eb990)

Summary: 
After optimizing the model, the accuracy increased from 72.85% to 78.92%. The loss decreased from 0.5511 to 0.4528. The major changes were dropping fewer columns and keeping the column "NAME", and changing the activation function for the second hidden layer. In addtion to these actual changes, different number of nodes and hidden layers were tested. Also number of epochs as well as types of optimizer had been tried. The hyperparameters that resulted in the highest accuracy were kepted. 

Step 5: Push Files Into Repository

*The file of AlphabetSoupCharity_colab is shared through the link https://colab.research.google.com/drive/1dkeTDb0zOOtKVhuiT6C3Xg3Zhs_6BydW?usp=drive_link

*The file of AlphabetSoupCharity_Optimiztion_colab is shared through the link https://colab.research.google.com/drive/1dkeTDb0zOOtKVhuiT6C3Xg3Zhs_6BydW?usp=drive_link

