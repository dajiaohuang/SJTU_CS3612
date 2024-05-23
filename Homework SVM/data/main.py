import os
import numpy as np
from dataset import get_data,get_HOG,standardize

from matplotlib import pyplot as plt

if __name__ == '__main__':
######################## Get train/test dataset ########################
    X_train,X_test,Y_train,Y_test = get_data('dataset')
########################## Get HoG featues #############################
    H_train,H_test = get_HOG(X_train), get_HOG(X_test)
######################## standardize the HoG features ####################
    H_train,H_test = standardize(H_train), standardize(H_test)
########################################################################
######################## Implement you code here #######################
########################################################################

from sklearn.svm import SVC
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--kernel', type=str, default='linear', help='linear,rbf,poly')
parser.add_argument('--C', type=float, default=0.1, help='[0.001,1000]')
args = parser.parse_args()
print(args)

svc = SVC(kernel=args.kernel, C=args.C)
svc.fit(H_train, Y_train)


predicted = svc.predict(H_test)
acc=np.sum(predicted == Y_test) / len(predicted)

print(f'Acc: {acc}')



# class SVM:
#     def __init__(self, kernel='linear', C=1.0):
#         self.kernel = kernel
#         self.C = C
#         self.alpha = None
#         self.b = None
#         self.X = None
#         self.y = None

#     def _kernel(self, X1, X2):
#         if self.kernel == 'linear':
#             return np.dot(X1, X2.T)
#         elif self.kernel == 'rbf':
#             gamma = 1 / X1.shape[1]
#             return np.exp(-gamma * np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=-1))
#         elif self.kernel == 'poly':
#             return (np.dot(X1, X2.T) + 1) ** 3
#         else:
#             raise ValueError('Invalid kernel type. Choose from "linear", "rbf", or "poly".')

#     def fit(self, X, y):
#         n_samples, n_features = X.shape
#         self.X = X
#         self.y = y

#         # Initialize Lagrange multipliers
#         self.alpha = np.zeros(n_samples)

#         # Compute the kernel matrix
#         K = self._kernel(X, X)

#         # Solve the dual optimization problem
#         for i in range(n_samples):
#             alpha_i_old = self.alpha[i]
#             L = max(0, self.alpha[i])
#             H = min(self.C, self.C + self.alpha[i])
#             grad_i = np.sum(self.alpha * self.y * K[:, i]) - self.y[i]
#             self.alpha[i] = self.alpha[i] + grad_i / K[i, i]
#             self.alpha[i] = np.clip(self.alpha[i], L, H)
#             delta_alpha_i = self.alpha[i] - alpha_i_old

#         # Compute the bias term
#         self.b = np.mean(self.y - np.dot(self.alpha * self.y, K))

#     def predict(self, X):
#         K = self._kernel(X, self.X)
#         return np.sign(np.dot(self.alpha * self.y, K.T) + self.b)

# # Usage example
# svm = SVM(kernel=args.kernel, C=1.5)
# svm.fit(H_train, Y_train)
# predicted = svm.predict(H_test)
# acc = np.sum(predicted == Y_test) / len(predicted)
# print(f'My Acc: {acc:.2f}')