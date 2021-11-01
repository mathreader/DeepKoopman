from matplotlib import pyplot
import numpy as np

#extension = 'experiment_1_10_2021_08_20_17_30_50_887276' #experiment 1
#extension = 'experiment_2_10_2021_08_20_17_32_21_985846' #experiment 2
#extension = 'experiment_4_10_2021_08_27_09_45_22_292512' #experiment 4

# New ones 
extension = 'experiment_4_10_2021_08_28_18_31_30_929630' #actually experiment 5, though forgot to change name, frobenius squared norm + condition number on K + spectral leakage

# 
extension = 'experiment_6_10_2021_08_29_16_17_09_063377'

## Load Data
f = open('./DeepDMD_results/Pendulum_' + extension + '_error.csv')
f_lines = f.readlines()
len_f = len(f_lines) - 1

## Store Data
data = {}
data['epochs'] 			= np.zeros(len_f)
data['times'] 			= np.zeros(len_f)
data['train_MSE'] 		= np.zeros(len_f)
data['train_cond_num'] 	= np.zeros(len_f)
data['train_SL'] 		= np.zeros(len_f)
data['val_MSE'] 		= np.zeros(len_f)
data['val_cond_num'] 	= np.zeros(len_f)
data['val_SL'] 			= np.zeros(len_f)

for i in range(0,len_f):
	print(i)
	line = f_lines[i + 1].strip('\n').split(', ')

	data['epochs'][i] 			= int(line[0]);
	data['times'][i] 			= float(line[1]);
	data['train_MSE'][i] 		= float(line[2]);
	data['train_cond_num'][i] 	= float(line[3]);
	data['train_SL'][i] 		= float(line[4]);
	data['val_MSE'][i] 			= float(line[5]);
	data['val_cond_num'][i] 	= float(line[6]);
	data['val_SL'][i] 			= float(line[7]);

## Plot Results

#plot MSE over time
pyplot.plot(data['times']/60, data['val_MSE'])
pyplot.yscale('log')
pyplot.xlabel('Time (min)')
pyplot.ylabel('Prediction Validation Error')
pyplot.title('Experiment with loss as prediction error + regularization on condition number')
pyplot.savefig("./Plots/Prediction_Error_Experiment_4_10.png")
pyplot.show()


#plot spectral leakage over time
pyplot.plot(data['times']/60, data['val_SL'])
pyplot.yscale('log')
pyplot.xlabel('Time (min)')
pyplot.ylabel('Spectral Leakage Error')
pyplot.title('Experiment with loss as prediction error + regularization on condition number')
pyplot.savefig("./Plots/SL_Error_Experiment_4_10.png")
pyplot.show()
