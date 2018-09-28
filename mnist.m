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

hiddenFnc = @sigmoid;
outputFcn = @softmax;
errorFnc = @crossEntropyDerivative;
TsSize = 900;
batchSize = 1;
eta = 0.001;
epochNumber=5;

% Create neural network
net = neuralNet(784, [250, 10], {hiddenFnc, outputFcn}, errorFnc);

errors = zeros(epochNumber, 1);

%start training for each epoch
tic
for epoch = 1: epochNumber
    net = train(net, train_im, train_lb, eta, TsSize, batchSize);
    fprintf('epoch: %d\n', epoch);

    
end

 timeElapsed = toc;
 fprintf("time elapsed for training: %.2f seconds\n", timeElapsed);
% Test neural net
correct = 0;
tic

[out,outputs] = forwardPropagation(net, test_im);

for i = 1: size(outputs{1,2}, 1)
    [val, idx] = max(outputs{1,2}(i,:));    
    if( idx == find( test_lb(i, :) ) )
        correct = correct + 1;
    end
end

elapsedTime = toc;
fprintf("acuracy: %.2f%%\n", (correct/size(test_im, 1))*100);
fprintf("elapsed time for testing: %.3f seconds\n", elapsedTime);


