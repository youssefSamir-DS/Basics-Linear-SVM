This repository implements a basic Linear Support Vector Machine (SVM) model from scratch in Python. It includes functions for data exploration, training, prediction, evaluation, and visualization.
Project Overview

    This project provides a basic understanding of how a linear SVM works.
    It demonstrates the implementation of core SVM functionalities
    It visualizes the decision boundary learned by the model.

Dependencies

This project requires the following Python libraries:

    numpy (numerical computing)
    matplotlib.pyplot (plotting)
    sklearn.model_selection (data splitting)

Note: You can install these dependencies using pip install numpy matplotlib scikit-learn
Usage

    Clone the repository:

Bash

git clone https://github.com/youssefSamir-DS/Basics-Linear-SVM.git


    Run the script:

Bash

python main.py


This will execute the code, perform training, and display results.
Code Structure

    Basic_SVM_model.py: Contains the implementation of the SVM model functions.
    main.py: Demonstrates the usage of the implemented functions.

Key Functions:

    fit(num_iterations, X, y_true): Trains the SVM model with gradient descent.
    predict(X_test, W, b): Predicts labels for new data points.
    Hingeloss(y_test, y_pred): Calculates the hinge loss for the model.
    accuracy(y_test, y_pred): Calculates the accuracy of the model predictions.
    plot_decision_boundary(X, y, W, b): Plots the decision boundary for the data.

Output

The script will generate the following outputs:

    Exploration plot of the data points
    Trained model weights and bias
    Predicted labels for the test set
    Mean Hinge Loss
    Accuracy score of the model
    Decision boundary plots for the entire data and for the test data


Contributing

We welcome contributions to this project! Please create a pull request to share your improvements.  

