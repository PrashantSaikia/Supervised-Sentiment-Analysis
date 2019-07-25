import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
import time

def model_evaluation(X_test, test_target, target_names, model, figsize=(4,3)):
	conf_mat = metrics.confusion_matrix(test_target, model.predict(X_test))
	fig, ax = plt.subplots(figsize=figsize)
	sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names)
	plt.ylabel('Actual')
	plt.xlabel('Predicted')
	plt.title('Confusion matrix (Test data)')
	plt.gcf().subplots_adjust(bottom=0.15) # To avoid xlabel from being cut off
	plt.savefig('plots/confusion_matrix_{}.png'.format(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))))

	res = metrics.classification_report(test_target, model.predict(X_test), target_names=target_names)
	with open('classification_report.txt', "a") as file:
		file.write(time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time())))
		file.write("\n"+res)
		file.write("========================================================================\n")