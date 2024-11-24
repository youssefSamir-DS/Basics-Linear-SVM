import numpy as np
import matplotlib.pyplot as plt 

def Hingeloss(y_true,y_pred):
    """Calculate the hinge loss of given true and predicted labels.
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
    Returns:
        Hinge loss."""
    loss = np.maximum(0,1-y_true*y_pred)
    return np.mean(loss)


def gradient_descent(X,y_true,W,b,learning_rate):
    """Perform one step gradient descent for Updating weights and bias.
    Args:
        X: Frature matrix.
        y_true: True labels.
        W: Current weights.
        b: Current bias.
        learning_rate: Learning rate.
    Returns:
        Updated weights and bias"""
    n_samples, n_features = X.shape
    # Calculate Gradients
    dw = np.zeros(n_features)
    db = 0
    for i in range(n_samples):
        y_pred = np.dot(X,W) + b
        Y = y_true[i] * y_pred
        if Y.all() < 1 :
            dw += y_true[i] * X[i]
            db += y_true[i]
    # Update Parameters
    W -= learning_rate * dw
    b -= learning_rate * db
    return W,b


def predict(X,W,b):
    """Predict labels for given data points.
    Args:
        X: Features matrix.
        W: Model weights.
        b: Model bias.
    Returns:
        Predicted Labels."""
    return np.sign(np.dot(X,W) + b)


def fit(num_iterations,X,y_true,learning_rate=0.1):
    W = np.zeros(X.shape[1])
    b = 0
    for _ in range(num_iterations):
        W,b = gradient_descent(X,y_true,W,b,learning_rate)
    return W,b


def accuracy(y_true,y_pred):
    """Calculate the accuracy of the predictions.
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
    Returns:
        Accuracy score."""
    correct_predictions = np.sum(y_true[i] == y_pred[i] for i in range(len(y_true)))
    total_predictions = len(y_true)
    accuracy_score = correct_predictions / total_predictions
    return accuracy_score


def plot_decision_boundary(X,y_true,W,b):
    """Plots the decision boundary for linear SVM.
    Args:
        X: Feature matrix.
        y_true: True labels.
        W: Model weights.
        b: Model bias."""
    # Create a meshgrid to plot the decision boundary
    x1_min,x1_max = X[:,0].min() -1, X[:,0].max() +1
    x2_min,x2_max = X[:,1].min() -1, X[:,1].max() +1
    xx1, xx2 = np.meshgrid(np.arange(x1_min,x1_max,0.02),np.arange(x2_min,x2_max,0.02))
    # Calculate decision function
    z = W[0] * xx1 + W[1] * xx2 + b
    # Plot the decision boundary
    plt.contour(xx1,xx2,z, colors='k',levels=[0])
    # Plot the data points
    plt.scatter(X[:,0],X[:,1],c=y_true,cmap=plt.cm.Paired)
    # Put labels and title for plot
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Linear SVM Decision Boundary')
    # Show the plot
    plt.show()
