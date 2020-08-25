import random
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

spam_count = 0.0
nspam_count = 0.0
tpos = 0.0
tneg = 0.0
fpos = 0.0
fneg = 0.0
classifcation = 0

traindata = pd.read_csv(r"C:\Users\samar\OneDrive\Desktop\SpambaseClassification\spambase\spambase.data", header=None, dtype=float);
np_data = traindata.values;

spam = np_data[:1813,:]
i1 = np.arange(spam.shape[0])
np.random.shuffle(i1)

nspam = np_data[1813:,:]
i2 = np.arange(nspam.shape[0])
np.random.shuffle(i2)

trainspam = spam[:906,:]
trainnspam = nspam[:1394,:]

tstspam = spam[906:,:]
tstnspam = nspam[1394:,:]

ftrain = np.concatenate((trainspam,trainnspam),axis=0)
ftrain_target = ftrain[:,57]

ftest = np.concatenate((tstspam,tstnspam),axis=0)
ftest_target = ftest[:,57]

def formula(mean,std,a):
	np.seterr(divide='ignore')
	part1 = float(1 / (std * (np.sqrt(2 * np.pi))))
	part2 = float(np.exp(-1 * (np.square(a - mean))/(2 * np.square(float(std * std)))))
	res = part1 * part2
	return res

for i in range(0,ftrain.shape[0]):
	if(ftrain[i,57] == 1):
		spam_count += 1
	else:
		nspam_count += 1

priorspam = spam_count / len(ftrain);
priornspam = nspam_count / len(ftrain);

spam_mean = []
spam_sd = []

nspam_mean = []
nspam_sd = []

for i in range(0,ftrain.shape[1]):
	spamarray = []
	nspamarray = []

	for j in range(ftrain.shape[0]):
		if (ftrain[j][-1] == 1):
			spamarray.append(ftrain[j][i])
		else:
			nspamarray.append(ftrain[j][i])

	spam_mean.append(np.mean(spamarray))
	spam_sd.append(np.std(spamarray))
	nspam_mean.append(np.mean(nspamarray))
	nspam_sd.append(np.std(nspamarray))

for k in range(len(spam_sd)):
	if (spam_sd[k] == 0):
		spam_sd[k] = 0.0001

	if (nspam_sd[k] == 0):
		nspam_sd[k] = 0.0001

cresult = []

for i in range(ftest.shape[0]):
	temp1 = np.log(priorspam)
	temp2 = np.log(priornspam)

	for j in range(0,57):
		a = ftest[i][j]	
		temp1 += np.log(formula(spam_mean[j], spam_sd[j], a))
		temp2 += np.log(formula(nspam_mean[j], nspam_sd[j], a))
	classification = np.argmax([temp2, temp1])
	cresult.append(classification)

cmatrix = confusion_matrix(ftest_target, cresult)
print("\nConfusion matrix:\n",cmatrix)

for i in range(len(cresult)):
	if (cresult[i] == 1 and ftest_target[i] == 1):
		tpos += 1
	elif (cresult[i] == 0 and ftest_target[i] == 0):
		tneg += 1
	elif (cresult[i] == 1 and ftest_target[i] == 0):
		fpos += 1
	else:
		fneg += 1

acc = float(tpos + tneg) / (tpos + tneg + fneg + fpos)
prec = float(tpos) / (tpos + fpos)
rec = float(tpos) / (tpos + fneg)

print("Accuracy:    ",acc)
print("Precision:   ",prec)
print("Recall:      ",rec)
