import Basic_SVM_model as SVM

# Create Dummy Data
import numpy as np 
X = np.array([[10,13], [5,6], [13,11], [8,7], [15,11]])
y = np.array([1,1,-1,1,-1])

# Explore Data
import matplotlib.pyplot as plt
plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.Paired)
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Explore Data")
plt.show()

# Divide Data into Training and Testing sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y, test_size=0.2,random_state=42)

# Train SVM model
W, b = SVM.fit(num_iterations=1000,X=x_train,y_true=y_train)

# Print model weights and biaas
print(f"Weights = {W}\nBias = {b}")

# Print Test Feature 
print('Test Feature Matrix = ',x_test)

# Make Predictions
y_pred = SVM.predict(x_test,W,b)
# print predictions
print("Predictions = ",y_pred)
# Print test value
print('Test value = ',y_test)

# Calculate Hinge Loss
loss = SVM.Hingeloss(y_test,y_pred)
# print mean value of Hinge Loss
print('Mean Loss = ',loss)

# Calculate accuracy of the model predictions
accuracy = SVM.accuracy(y_test,y_pred)
# print accuracy score
print('Accuracy Score = ',accuracy)

# Plot Decision Boundary for all data
SVM.plot_decision_boundary(X,y,W,b)

# Plot Decision Boundary for predicted data
SVM.plot_decision_boundary(x_test,y_pred,W,b)