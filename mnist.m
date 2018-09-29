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

hiddenFnc = @tanH;
outputFcn = @identity;
errorDerivative = @crossEntropyDerivative;
errorFnc = @crossEntropy;
TsSize = 3200;
batchSize = 32;
eta = 0.01;
epochNumber = 50;

% Create neural network
net = neuralNet(784, [250, 10], {hiddenFnc, outputFcn}, errorDerivative);

errorsOnline = zeros(epochNumber, 1);
errorsBatch = zeros(epochNumber, 1);

%start training for each epoch (online)
tic
fprintf('Online Training\n');
for epoch = 1: epochNumber
    net = train(net, train_im, train_lb, eta, TsSize, 1);  
    % Test neural net
    correct = 0;
    [~, out] = forwardPropagation(net, test_im, @softmax);
    
    for i = 1: size(out{1,2}, 1)
        [~, idx] = max(out{1,2}(i,:));
        if( idx == find( test_lb(i, :) ) )
            correct = correct + 1;
        end
    end
    errorsOnline(epoch) = calculateError(out{1,2}, test_lb, errorFnc);
    %errors(epoch) = sum(errorFnc(out{1,2},test_lb));
    fprintf('epoch: %3d; accuracy: %3.2f%%\n', epoch, (correct/size(test_im, 1))*100);
end
toc
tic
%start training for each epoch (miniBatch)
fprintf('MiniBatch Training\n');
net = neuralNet(784, [250, 10], {hiddenFnc, outputFcn}, errorDerivative);

for epoch = 1: epochNumber
    net = train(net, train_im, train_lb, eta, TsSize, batchSize);  
    % Test neural net
    correct = 0;
    [~, out] = forwardPropagation(net, test_im, @softmax);
    
    for i = 1: size(out{1,2}, 1)
        [~, idx] = max(out{1,2}(i,:));
        if( idx == find( test_lb(i, :) ) )
            correct = correct + 1;
        end
    end
    errorsBatch(epoch) = calculateError(out{1,2}, test_lb, errorFnc);
    %errors(epoch) = sum(errorFnc(out{1,2},test_lb));
    fprintf('epoch: %3d; accuracy: %3.2f%%\n', epoch, (correct/size(test_im, 1))*100);
end


hold on
warning off;
legend('Error');
title('Loss Decay');
xlabel('Epochs');
ylabel('Error (SUM of 10k errors)');
axis auto;
plot(errorsOnline(1:epochNumber), 'r','DisplayName','online');
plot(errorsBatch(1:epochNumber), 'b','DisplayName','batch');
hold off

elapsedTime = toc;
fprintf("time elapsed for execution: %.2f seconds\n", elapsedTime);


