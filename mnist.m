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
%activation function for hidden layer
hiddenFnc = @sigmoid;
%activation function for output layer
outputFcn = @identity;
errorDerivative = @crossEntropyDerivative;
errorFnc = @crossEntropy;
TsSize = 3200;
batchSize = 3200;
eta = 0.05;
epochNumber = 100;
hiddenNodes = 250;

% Create neural network
net = neuralNet(784, [hiddenNodes, 10], {hiddenFnc, outputFcn}, errorDerivative);
%store error and accuracy computed after each epoch on all test set
errorsOnline = zeros(epochNumber, 1);
errorsBatch = zeros(epochNumber, 1);
accuracyOnline = zeros(epochNumber, 1);
accuracyBatch = zeros(epochNumber, 1);

%start training for each epoch (online learning)
tic
fprintf('Online Training\n');
for epoch = 1: epochNumber
    %train the network with a batchSize of 1 (online)
    net = train(net, train_im, train_lb, eta, TsSize, 1);  
    % Test neural net
    correct = 0;
    [~, out] = forwardPropagation(net, test_im, @softmax);
    %compute correctly guessed exemples
    for i = 1: size(out{1,2}, 1)
        [~, idx] = max(out{1,2}(i,:));
        if( idx == find( test_lb(i, :) ) )
            correct = correct + 1;
        end
    end
    %store current error standard deviation(for plot purpose)evaluated on
    %each test case
    errorsOnline(epoch) = calculateError(out{1,2}, test_lb, errorFnc);
    %store accuracy for this epoch
    accuracyOnline(epoch)=(correct/size(test_im, 1))*100;
    fprintf('epoch: %3d; accuracy: %3.2f%%; error: %3f\n', epoch,accuracyOnline(epoch) ,errorsOnline(epoch));
end
toc

tic
%start training for each epoch (miniBatch)
fprintf('MiniBatch Training\n');
net = neuralNet(784, [hiddenNodes, 10], {hiddenFnc, outputFcn}, errorDerivative);

for epoch = 1: epochNumber
    net = train(net, train_im, train_lb, eta, TsSize, batchSize);  
    % Test neural net
    correct = 0;
    [~, out] = forwardPropagation(net, test_im, @softmax);
    %compute correctly guessed exemples
    for i = 1: size(out{1,2}, 1)
        [~, idx] = max(out{1,2}(i,:));
        if( idx == find( test_lb(i, :) ) )
            correct = correct + 1;
        end
    end
    %store current error standard deviation(for plot purpose)evaluated on
    %each test case
    errorsBatch(epoch) = calculateError(out{1,2}, test_lb, errorFnc);
    %store accuracy for this epoch
    accuracyBatch(epoch) = (correct/size(test_im, 1))*100;
    fprintf('epoch: %3d; accuracy: %3.2f%%; error: %3f\n', epoch,accuracyBatch(epoch) ,errorsBatch(epoch));
end

% plotting loss of online and batch learning with the same number of epochs
figure('Name', 'Error');
hold on
warning off;
legend('Error');
title('Loss Decay');
xlabel('Epochs');
ylabel('Error (SUM of 10k errors)');
axis auto;
plot(errorsOnline(1:epochNumber), 'r','DisplayName','online');
plot(errorsBatch(1:epochNumber), 'b','DisplayName','minibatch');
hold off
drawnow;

% plotting accuracy of online and batch learning with the same number of epochs
figure('Name', 'Accuracy');
hold on
warning off;
legend('Accuracy');
title('Total Accuracy');
xlabel('Epochs');
ylabel('Accuracy rate %');
axis auto;
plot(accuracyOnline(1:epochNumber), 'r','DisplayName','online');
plot(accuracyBatch(1:epochNumber), 'b','DisplayName','minibatch');
hold off
drawnow;


elapsedTime = toc;
fprintf("time elapsed for execution: %.2f seconds\n", elapsedTime);


