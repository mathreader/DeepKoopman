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

# network structure: 80 80 2 2 80 80 2
f_Koopman_no_aux = open('./feedforward_results/Pendulum_Koopman_no_aux_2021_07_11_13_33_43_884109_error.csv','r')
K_na_lines = f_Koopman_no_aux.readlines()
len_K_na = len(K_na_lines) - 1

# feedforward network trained on 1 step
f_feedforward1 = open('./feedforward_results/Pendulum_1shifts_2021_08_01_14_16_28_398600_error.csv','r')
ff1_lines = f_feedforward1.readlines()
len_ff1 = len(ff1_lines) - 1

# feedforward network trained on 5 steps
f_feedforward5 = open('./feedforward_results/Pendulum_5shifts_2021_08_01_14_18_06_067076_error.csv','r')
ff5_lines = f_feedforward5.readlines()
len_ff5 = len(ff5_lines) - 1

# feedforward network trained on 50 steps
f_feedforward50 = open('./feedforward_results/Pendulum_50shifts_2021_08_01_14_19_28_606253_error.csv','r')
ff50_lines = f_feedforward50.readlines()
len_ff50 = len(ff50_lines) - 1

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
dk['val_loss_50']	= np.zeros(len_dk)

for i in range(0,len_dk):
	line = dk_lines[i + 1].strip('\n').split(',')

	dk['epochs'][i] 		= int(float(line[23]));
	dk['times'][i] 			= float(line[22]);
	dk['train_loss'][i] 	= float(line[2]);
	dk['val_loss'][i] 		= float(line[3]);
	dk['comp_loss'][i] 		= float(line[17]);
	dk['comp_loss_5'][i] 	= float(line[19]);
	dk['comp_loss_15'][i] 	= float(line[21]);
	dk['val_loss_50'][i] 	= float(line[7]);

# Koopman without Auxiliary network data
K_na = {}
K_na['epochs'] 			= np.zeros(len_K_na)
K_na['times'] 			= np.zeros(len_K_na)
K_na['train_loss'] 		= np.zeros(len_K_na)
K_na['val_loss'] 		= np.zeros(len_K_na)
K_na['comp_loss'] 		= np.zeros(len_K_na)
K_na['comp_loss_5'] 	= np.zeros(len_K_na)
K_na['comp_loss_15']	= np.zeros(len_K_na)

for i in range(0,len_K_na):
	line = K_na_lines[i + 1].strip('\n').split(', ')

	K_na['epochs'][i] 		= int(line[0]);
	K_na['times'][i] 		= float(line[1]);
	K_na['train_loss'][i] 	= float(line[2]);
	K_na['val_loss'][i] 	= float(line[3]);
	K_na['comp_loss'][i] 	= float(line[4]);
	K_na['comp_loss_5'][i] 	= float(line[5]);
	K_na['comp_loss_15'][i] = float(line[6]);


# feedforward 1-step data
ff1 = {}
ff1['epochs'] 			= np.zeros(len_ff1)
ff1['times'] 			= np.zeros(len_ff1)
ff1['train_loss_1'] 	= np.zeros(len_ff1)
ff1['val_loss_1'] 		= np.zeros(len_ff1)
ff1['train_loss_5'] 	= np.zeros(len_ff1)
ff1['val_loss_5'] 		= np.zeros(len_ff1)
ff1['train_loss_50'] 	= np.zeros(len_ff1)
ff1['val_loss_50'] 		= np.zeros(len_ff1)


for i in range(0,len_ff1):
	line = ff1_lines[i + 1].strip('\n').split(', ')

	ff1['epochs'][i] 		= int(line[0]);
	ff1['times'][i] 		= float(line[1]);
	ff1['train_loss_1'][i] 	= float(line[2]);
	ff1['val_loss_1'][i] 	= float(line[3]);
	ff1['train_loss_5'][i] 	= float(line[4]);
	ff1['val_loss_5'][i] 	= float(line[5]);
	ff1['train_loss_50'][i] = float(line[6]);
	ff1['val_loss_50'][i] 	= float(line[7]);

# feedforward 5-step data
ff5 = {}
ff5['epochs'] 			= np.zeros(len_ff5)
ff5['times'] 			= np.zeros(len_ff5)
ff5['train_loss_1'] 	= np.zeros(len_ff5)
ff5['val_loss_1'] 		= np.zeros(len_ff5)
ff5['train_loss_5'] 	= np.zeros(len_ff5)
ff5['val_loss_5'] 		= np.zeros(len_ff5)
ff5['train_loss_50'] 	= np.zeros(len_ff5)
ff5['val_loss_50'] 		= np.zeros(len_ff5)


for i in range(0,len_ff5):
	line = ff5_lines[i + 1].strip('\n').split(', ')

	ff5['epochs'][i] 		= int(line[0]);
	ff5['times'][i] 		= float(line[1]);
	ff5['train_loss_1'][i] 	= float(line[2]);
	ff5['val_loss_1'][i] 	= float(line[3]);
	ff5['train_loss_5'][i] 	= float(line[4]);
	ff5['val_loss_5'][i] 	= float(line[5]);
	ff5['train_loss_50'][i] = float(line[6]);
	ff5['val_loss_50'][i] 	= float(line[7]);

# feedforward 50-step data
ff50 = {}
ff50['epochs'] 			= np.zeros(len_ff50)
ff50['times'] 			= np.zeros(len_ff50)
ff50['train_loss_1'] 	= np.zeros(len_ff50)
ff50['val_loss_1'] 		= np.zeros(len_ff50)
ff50['train_loss_5'] 	= np.zeros(len_ff50)
ff50['val_loss_5'] 		= np.zeros(len_ff50)
ff50['train_loss_50'] 	= np.zeros(len_ff50)
ff50['val_loss_50'] 	= np.zeros(len_ff50)


for i in range(0,len_ff50):
	line = ff50_lines[i + 1].strip('\n').split(', ')

	ff50['epochs'][i] 		= int(line[0]);
	ff50['times'][i] 		= float(line[1]);
	ff50['train_loss_1'][i] = float(line[2]);
	ff50['val_loss_1'][i] 	= float(line[3]);
	ff50['train_loss_5'][i] = float(line[4]);
	ff50['val_loss_5'][i] 	= float(line[5]);
	ff50['train_loss_50'][i]= float(line[6]);
	ff50['val_loss_50'][i] 	= float(line[7]);
	

## Plot Results
# plot comparison loss vs time
pyplot.plot(ff['times'], ff['comp_loss'], label = 'Feed-forward')
pyplot.plot(dk['times'], dk['comp_loss'], label = 'Deep Koopman')
pyplot.plot(K_na['times'], K_na['comp_loss'], label = 'Koopman Feed-forward')
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
pyplot.plot(K_na['times'], K_na['comp_loss_5'], label = 'Koopman Feed-forward')
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
pyplot.plot(K_na['times'], K_na['comp_loss_15'], label = 'Koopman Feed-forward')
pyplot.legend()

pyplot.yscale('log')
pyplot.xlabel('Time (s)')
pyplot.ylabel('Comparison Loss After 15 Time Steps')
pyplot.title('Deep Koopman and Feed Forward 15 Step Comparison')
pyplot.savefig("./Plots/Comparison_15_vs_Time.png")
pyplot.show()

# plot new loss for 1 step vs time
pyplot.plot(ff1['times'], ff1['val_loss_1'], label = 'trained on 1 step')
pyplot.plot(ff5['times'], ff5['val_loss_1'], label = 'trained on 5 steps')
pyplot.plot(ff50['times'], ff50['val_loss_1'], label = 'trained on 50 steps')

pyplot.legend()

pyplot.yscale('log')
pyplot.xlabel('Time (s)')
pyplot.ylabel('1 step loss')
pyplot.title('Feedforward networks trained on different shifts of data')
pyplot.savefig("./Plots/Shift1_Loss_vs_Time.png")
pyplot.show()

# plot new loss for 5 steps vs time
pyplot.plot(ff1['times'], ff1['val_loss_5'], label = 'trained on 1 step')
pyplot.plot(ff5['times'], ff5['val_loss_5'], label = 'trained on 5 steps')
pyplot.plot(ff50['times'], ff50['val_loss_5'], label = 'trained on 50 steps')

pyplot.legend()

pyplot.yscale('log')
pyplot.xlabel('Time (s)')
pyplot.ylabel('5 step loss')
pyplot.title('Feedforward networks trained on different shifts of data')
pyplot.savefig("./Plots/Shift5_Loss_vs_Time.png")
pyplot.show()

#plot new loss for 50 steps vs time
pyplot.plot(ff1['times'], ff1['val_loss_50'], label = 'trained on 1 step')
pyplot.plot(ff5['times'], ff5['val_loss_50'], label = 'trained on 5 steps')
pyplot.plot(ff50['times'], ff50['val_loss_50'], label = 'trained on 50 steps')
pyplot.plot(dk['times'], dk['val_loss_50']*(10**3), label = 'Deep Koopman')

pyplot.legend()

pyplot.yscale('log')
pyplot.xlabel('Time (s)')
pyplot.ylabel('50 step loss')
pyplot.title('Feedforward networks trained on different shifts of data vs Deep Koopman')
pyplot.savefig("./Plots/Shift50_Loss_vs_Time.png")
pyplot.show()

