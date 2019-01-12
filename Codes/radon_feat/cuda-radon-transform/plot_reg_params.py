import os,sys
import re
from glob import glob
import matplotlib.pyplot as plt

if len(sys.argv) <= 1:
	print("usage:  {filename(s)}")
	quit()
infiles = [fname for argv in sys.argv[1:] for fname in glob(argv)]

def plot_var_from_reg_param_file(filename, var_description):
	regs = []
	aucs = []

	with open(infile, 'r') as readme:
		for line in readme:
			line = line.replace('\r','').replace('\n','')
			if len(line) > 3 and ', ROC AUC ' in line:
				spl = line.split(', ') #replace('REGPARAM ','').replace(' accuracy: ',', ROC AUC ').split(', ROC AUC ')
				descrip = [re.sub('[0123456789\.]','',spiece).strip() for spiece in spl]
				numbers = [re.sub('[^0123456789\.]','',spiece) for spiece in spl]

				spl_idx = None
				for didx in range(len(descrip)):
					if descrip[didx] == var_description:
						spl_idx = didx

				if spl_idx is None:
					assert 0, var_description+" not found in file \'"+filename+"\'"
				assert descrip[0] == 'REGPARAM', str(descrip[0])

				regs.append(float(numbers[0]))
				aucs.append(float(numbers[spl_idx]))

	labelname = infile
	if labelname.endswith('.txt'):
		labelname = labelname[:-4]
	plt.scatter(regs, aucs, label=labelname)

# plot AUCs
plt.figure(1)
for infile in infiles:
	plot_var_from_reg_param_file(infile, 'ROC AUC')
plt.xlabel('regularization')
plt.ylabel('auc')
plt.xlim([0., 1.])
plt.ylim([0.87, 0.90])
plt.grid(True)
plt.legend(loc='lower right')

# plot distances from ROC diag
plt.figure(2)
for infile in infiles:
	plot_var_from_reg_param_file(infile, 'maxfromdiag')
plt.xlabel('regularization')
plt.ylabel('most distance from ROC diag')
plt.title( 'most distance from ROC diag')
plt.xlim([0., 1.])
#plt.ylim([0.87, 0.90])
plt.grid(True)
plt.legend(loc='lower right')

# plot thresholds
plt.figure(3)
for infile in infiles:
	plot_var_from_reg_param_file(infile, 'threshold')
plt.xlabel('regularization')
plt.ylabel('threshold for most distance from ROC diag')
plt.title( 'threshold for most distance from ROC diag')
plt.xlim([0., 1.])
#plt.ylim([0.87, 0.90])
plt.grid(True)
plt.legend(loc='lower right')

plt.show()
