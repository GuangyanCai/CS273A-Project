import numpy as np
from sklearn.metrics import roc_curve
from os.path import join 
from matplotlib import pyplot as plt

dir_pred = 'prediction'
svm_y_test = np.loadtxt(join(dir_pred, 'svm_y_test.txt'))
svm_yhat_test = np.loadtxt(join(dir_pred, 'svm_yhat_test.txt'))
mlp_y_test = np.loadtxt(join(dir_pred, 'mlp_y_test.txt'))
mlp_yhat_test = np.loadtxt(join(dir_pred, 'mlp_yhat_test.txt'))
rf_y_test = np.loadtxt(join(dir_pred, 'rf_y_test.txt'))
rf_yhat_test = np.loadtxt(join(dir_pred, 'rf_yhat_test.txt'))

svm_fpr, svm_tpr, _ = roc_curve(svm_y_test, svm_yhat_test)
mlp_fpr, mlp_tpr, _ = roc_curve(mlp_y_test, mlp_yhat_test)
rf_fpr, rf_tpr, _ = roc_curve(rf_y_test, rf_yhat_test)


plt.plot(svm_fpr, svm_tpr, label='svm')
plt.plot(mlp_fpr, mlp_tpr, label='mlp')
plt.plot(rf_fpr, rf_tpr, label='rf')
plt.plot(np.linspace(0, 1, num=20), np.linspace(0, 1, num=20), '--')
plt.legend(title='classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves of Our Classifiers')
plt.savefig('roc_plot.png')