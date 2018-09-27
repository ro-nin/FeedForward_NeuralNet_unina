clc;
clear;

addpath('./mnist/');
addpath('./utils/');
addpath('./net functions/');

% 60000 x 784
train_im = loadMNISTImages('train-images.idx3-ubyte')';
% 60000 x 10
train_lb = loadMNISTLabels('train-labels.idx1-ubyte');
train_lb = train_lb';

train_lb(train_lb==0) = 10;
train_lb = dummyvar(train_lb);

% 784 x 10000
test_im = loadMNISTImages('t10k-images.idx3-ubyte')';
% 10000 x 10
test_lb = loadMNISTLabels('t10k-labels.idx1-ubyte');
test_lb = test_lb';

test_lb(test_lb==0) = 10;
test_lb = dummyvar(test_lb);

% Dimension of taining size
ts_size = 10000;

% Numeber of folds 
k = 5;

%calcolo degli slice del training set da utilizzare per il k-folding
slice_size = int32(ts_size / k); 
resized_im = train_im(1:ts_size, :);
resized_lb = train_lb(1:ts_size, :);

% Fixed hyperparam
errorFnc = @crossEntropyDerivative;

% Hyperparameters to test
netFnc = {{@tanH, @ReLU}, {@sigmoid, @identity}, {@tanH, @sigmoid}};
netNodes = [100, 200, 300, 500, 800];
netEtas = [0.7, 0.1, 0.05, 0.01, 0.008, 0.004];

for fnc = 1: length(netFnc)
    for node = netNodes
        for eta = netEtas
            for i = 0: k-1
                % Calculate the slice index for testing and for validation
                start_idx = slice_size * i + 1;
                stop_idx = start_idx + slice_size - 1;
                
                k_train_im = [resized_im(1:start_idx, :); resized_im(stop_idx:ts_size, :)];
                k_train_lb = [resized_lb(1:start_idx, :); resized_lb(stop_idx:ts_size, :)];
                
                k_test_im = resized_im(start_idx:stop_idx, :);
                k_test_lb = resized_lb(start_idx:stop_idx, :);
                
                net = neuralNet(784, [node, 10], netFnc{fnc}, errorFnc);

                net = train(net, k_train_im, k_train_lb, eta, size(k_train_im, 1), 1);
                
            end            
        end
    end
end
        

















