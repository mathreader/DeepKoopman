from matplotlib import pyplot
import numpy as np

data_name = 'Lorenz1'
experiment_tag = 'experiment_18'
date = '2021_12_07_12_46_16_804693'

## Load Data
f = open('./DeepDMD_results/' + data_name + '_' + experiment_tag + '_' + date + '_error.csv')
f_lines = f.readlines()
len_f = len(f_lines) - 1

## Store Data
data = {}
data['epochs'] 			= np.zeros(len_f)
data['times'] 			= np.zeros(len_f)

data['train_MSE'] 		= np.zeros(len_f)
data['train_cond_num'] 	= np.zeros(len_f)
data['train_SL'] 		= np.zeros(len_f)
data['train_res']		= np.zeros(len_f)

data['val_MSE'] 		= np.zeros(len_f)
data['val_cond_num'] 	= np.zeros(len_f)
data['val_SL'] 			= np.zeros(len_f)
data['val_res']			= np.zeros(len_f)

for i in range(0,len_f):
	line = f_lines[i + 1].strip('\n').split(', ')

	data['epochs'][i] 			= int(line[0]);
	data['times'][i] 			= float(line[1]);

	data['train_MSE'][i] 		= float(line[2]);
	data['train_cond_num'][i] 	= float(line[3]);
	data['train_SL'][i] 		= float(line[4]);
	data['train_res'][i]		= float(line[5]);

	data['val_MSE'][i] 			= float(line[6]);
	data['val_cond_num'][i] 	= float(line[7]);
	data['val_SL'][i] 			= float(line[8]);
	data['val_res'][i]			= float(line[9]);

## Plot Results

#plot MSE over time
pyplot.plot(data['times']/60, data['val_MSE'])
pyplot.yscale('log')
pyplot.xlabel('Time (min)')
pyplot.ylabel('Prediction Validation Error')
pyplot.title(experiment_tag)
pyplot.savefig("./Plots/Prediction_Error_{}.png".format(experiment_tag))
pyplot.show()


#plot spectral leakage over time
pyplot.plot(data['times']/60, data['val_SL'])
pyplot.yscale('log')
pyplot.xlabel('Time (min)')
pyplot.ylabel('Spectral Leakage Error')
pyplot.title(experiment_tag)
pyplot.savefig("./Plots/SL_Error_{}.png".format(experiment_tag))
pyplot.show()

#plot residual over time
pyplot.plot(data['times']/60, data['val_res'])
pyplot.yscale('log')
pyplot.xlabel('Time (min)')
pyplot.ylabel('Residual')
pyplot.title(experiment_tag)
pyplot.savefig("./Plots/Residual_{}.png".format(experiment_tag))
pyplot.show()

