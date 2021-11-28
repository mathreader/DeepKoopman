numICs = 5000;
rho = 1;
filenamePrefix = strcat('Lorenz', num2str(rho));

%set initial condition ranges (need to verify that these are good choices)
xrange = [-1,1];
yrange = [-1,1];
zrange = [-1,1];

tSpan = 0:0.02:1;

% generate test data (which is 10% the number of numICs)
seed = 1;
X_test = LorenzFn(xrange, yrange, zrange, round(.1*numICs), tSpan, seed, rho);
filename_test = strcat(filenamePrefix, '_test_x.csv');
dlmwrite(filename_test, X_test, 'precision', '%.14f')

% generate validation data (which is 20% the number of numICs)
seed = 2;
X_val = LorenzFn(xrange, yrange, zrange, round(.2*numICs), tSpan, seed, rho);
filename_val = strcat(filenamePrefix, '_val_x.csv');
dlmwrite(filename_val, X_val, 'precision', '%.14f')

% generate training data sets, each of which has 70% of numICs samples
for j = 1:6
	seed = 2+j;
	X_train = LorenzFn(xrange, yrange, zrange, round(.7*numICs), tSpan, seed, rho);
	filename_train = strcat(filenamePrefix, sprintf('_train%d_x.csv', j));
	dlmwrite(filename_train, X_train, 'precision', '%.14f')
end
