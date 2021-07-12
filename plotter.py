'''
Plot results of feedforward networks vs deep koopman results.
'''

from matplotlib import pyplot
import numpy as np

## Load Data

# network structure: 80 80 80 80 2
f_feedforward = open('./feedforward_results/Pendulum_2021_07_11_12_03_04_002104_error.csv','r')
ff_lines = f_feedforward.readlines()
len_ff = len(ff_lines) - 1

# deep koopman network (with auxiliary network)
f_DeepKoopman = open('./exp2_best/Pendulum_2021_07_11_12_50_17_922906_error.csv')
dk_lines = f_DeepKoopman.readlines()
len_dk = len(dk_lines) - 1


## Store Data

# feedforward data
ff = {}
ff['epochs'] 		= np.zeros(len_ff)
ff['times'] 		= np.zeros(len_ff)
ff['train_loss'] 	= np.zeros(len_ff)
ff['val_loss'] 		= np.zeros(len_ff)
ff['comp_loss'] 	= np.zeros(len_ff)
ff['comp_loss_5'] 	= np.zeros(len_ff)
ff['comp_loss_15'] 	= np.zeros(len_ff)

for i in range(0,len_ff):
	line = ff_lines[i + 1].strip('\n').split(', ')

	ff['epochs'][i] 		= int(line[0]);
	ff['times'][i] 			= float(line[1]);
	ff['train_loss'][i] 	= float(line[2]);
	ff['val_loss'][i] 		= float(line[3]);
	ff['comp_loss'][i] 		= float(line[4]);
	ff['comp_loss_5'][i] 	= float(line[5]);
	ff['comp_loss_15'][i] 	= float(line[6]);

# Deep Koopman data
dk = {}
dk['epochs'] 		= np.zeros(len_dk)
dk['times'] 		= np.zeros(len_dk)
dk['train_loss'] 	= np.zeros(len_dk)
dk['val_loss'] 		= np.zeros(len_dk)
dk['comp_loss'] 	= np.zeros(len_dk)
dk['comp_loss_5'] 	= np.zeros(len_dk)
dk['comp_loss_15'] 	= np.zeros(len_dk)

for i in range(0,len_dk):
	line = dk_lines[i + 1].strip('\n').split(',')

	dk['epochs'][i] 		= int(float(line[23]));
	dk['times'][i] 			= float(line[22]);
	dk['train_loss'][i] 	= float(line[2]);
	dk['val_loss'][i] 		= float(line[3]);
	dk['comp_loss'][i] 		= float(line[17]);
	dk['comp_loss_5'][i] 	= float(line[19]);
	dk['comp_loss_15'][i] 	= float(line[21]);

## Plot Results
# plot comparison loss vs time
pyplot.plot(ff['times'], ff['comp_loss'], label = 'Feed-forward')
pyplot.plot(dk['times'], dk['comp_loss'], label = 'Deep Koopman')
pyplot.legend()

pyplot.yscale('log')
pyplot.xlabel('Time (s)')
pyplot.ylabel('Comparison Loss')
pyplot.title('Deep Koopman and Feed Forward Comparison')
pyplot.savefig("./Plots/Comparison_vs_Time.png")
pyplot.show()

# plot comparison loss after 5 steps vs time
pyplot.plot(ff['times'], ff['comp_loss_5'], label = 'Feed-forward')
pyplot.plot(dk['times'], dk['comp_loss_5'], label = 'Deep Koopman')
pyplot.legend()

pyplot.yscale('log')
pyplot.xlabel('Time (s)')
pyplot.ylabel('Comparison Loss After 5 Time Steps')
pyplot.title('Deep Koopman and Feed Forward 5 Step Comparison')
pyplot.savefig("./Plots/Comparison_5_vs_Time.png")
pyplot.show()

# plot comparison loss after 15 steps vs time
pyplot.plot(ff['times'], ff['comp_loss_15'], label = 'Feed-forward')
pyplot.plot(dk['times'], dk['comp_loss_15'], label = 'Deep Koopman')
pyplot.legend()

pyplot.yscale('log')
pyplot.xlabel('Time (s)')
pyplot.ylabel('Comparison Loss After 15 Time Steps')
pyplot.title('Deep Koopman and Feed Forward 15 Step Comparison')
pyplot.savefig("./Plots/Comparison_15_vs_Time.png")
pyplot.show()


