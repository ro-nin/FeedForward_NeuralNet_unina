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

%size of training set and validation set
TsSize = 32000;
VsSize = 8000;
%cut the training/val set
trainingSet=train_im(1:TsSize,:);
trainingSetLabel=train_lb(1:TsSize,:);
validationSet=train_im(TsSize+1:TsSize+1+VsSize,:);
validationSetLabel=train_lb(TsSize+1:TsSize+1+VsSize,:);

%activation function for hidden layer
hiddenFnc = @tanH;
%activation function for output layer
outputFcn = @ReLU;
errorDerivative = @crossEntropyDerivative;
errorFnc = @crossEntropy;

batchSize = 32;
eta = 0.01;
epochNumber = 500;
hiddenNodes = 250;

%epoch number cointainer for early stopping
finalEpochsMiniB = epochNumber;
finalEpochsOnline = epochNumber;

% Create neural network
net = neuralNet(784, [hiddenNodes, 10], {hiddenFnc, outputFcn}, errorDerivative);
% used to revert to previous net in early stopping
lastNet=net;
lastError=Inf;
%store error and accuracy computed after each epoch on all test set
errorsOnlineValidation = zeros(epochNumber, 1);
errorsOnlineTraining = zeros(epochNumber, 1);
errorsBatchValidation = zeros(epochNumber, 1);
errorsBatchTraining = zeros(epochNumber, 1);
accuracyOnline = zeros(epochNumber, 1);
accuracyBatch = zeros(epochNumber, 1);

tic
%start training for each epoch (miniBatch)
fprintf('MiniBatch Training\n');
for epoch = 1: epochNumber
    net = train(net, trainingSet, trainingSetLabel, eta, TsSize, batchSize);  
     % Evaluate Network
    %evaluate accuracy on current network
    correct = 0;
    [~, out] = forwardPropagation(net, validationSet, @softmax);
    %compute correctly guessed exemples
    for i = 1: size(out{1,2}, 1)
        [~, idx] = max(out{1,2}(i,:));
        if( idx == find( validationSetLabel(i, :) ) )
            correct = correct + 1;
        end
    end
    %store accuracy for this epoch
    accuracyBatch(epoch)=(correct/size(validationSet, 1))*100;
    %evaluate error on validation
    errorsBatchValidation(epoch) = calculateError(out{1,2}, validationSetLabel, errorFnc);
    %evaluate error on training
    [~, out] = forwardPropagation(net, trainingSet, @softmax);
    errorsBatchTraining(epoch) = calculateError(out{1,2}, trainingSetLabel, errorFnc);
    fprintf('epoch: %3d; accuracy: %3.2f%%; errorVal: %3f,errorTrain: %3f\n', epoch,accuracyBatch(epoch) ,errorsBatchValidation(epoch),errorsBatchTraining(epoch));
    %check early stopping every 5 epoch
    if(mod(epoch,5)==0)
        if(errorsBatchValidation(epoch)>lastError)
            %early stop
            net=lastNet;
            finalEpochsMiniB=epoch;
            break;
        else
            lastNet=net;
            lastError=errorsBatchValidation(epoch);
        end
    end
        
end
elapsedTime = toc;
fprintf("time elapsed for execution: %.2f minutes, epochs:%d\n", floor(elapsedTime/60),finalEpochsMiniB);

% plotting loss on validation and training in minibatch mode
figure('Name', 'Error');
hold on
warning off;
legend('Error');
title('Minibatch Loss');
xlabel('Epochs');
ylabel('Total Error');
axis auto;
plot(errorsBatchValidation(1:finalEpochsMiniB), 'r','DisplayName','Validation');
plot(errorsBatchTraining(1:finalEpochsMiniB), 'b','DisplayName','Training');
hold off
drawnow;

%accuracy test with minibatch network
correct = 0;
    [~, out] = forwardPropagation(net, test_im, @softmax);
    %compute correctly guessed exemples
    for i = 1: size(out{1,2}, 1)
        [~, idx] = max(out{1,2}(i,:));
        if( idx == find( test_lb(i, :) ) )
            correct = correct + 1;
        end
    end
fprintf('Minibatch accuracy on test set %2f\n', (correct/10000)*100);


lastNet=net;
lastError=Inf;
%start training for each epoch (online learning)
tic
fprintf('Online Training\n');
for epoch = 1: epochNumber
    %train the network with a batchSize of 1 (online)
    net = train(net, trainingSet, trainingSetLabel, eta, TsSize, 1);  
    % Evaluate Network
    %evaluate accuracy on current network
    correct = 0;
    [~, out] = forwardPropagation(net, validationSet, @softmax);
    %compute correctly guessed exemples
    for i = 1: size(out{1,2}, 1)
        [~, idx] = max(out{1,2}(i,:));
        if( idx == find( validationSetLabel(i, :) ) )
            correct = correct + 1;
        end
    end
    %store accuracy for this epoch
    accuracyOnline(epoch)=(correct/size(validationSet, 1))*100;
    %evaluate error on validation
    errorsOnlineValidation(epoch) = calculateError(out{1,2}, validationSetLabel, errorFnc);
    %evaluate error on training
    [~, out] = forwardPropagation(net, trainingSet, @softmax);
    errorsOnlineTraining(epoch) = calculateError(out{1,2}, trainingSetLabel, errorFnc);
    fprintf('epoch: %3d; accuracy: %3.2f%%; errorVal: %3f,errorTrain: %3f\n', epoch,accuracyOnline(epoch) ,errorsOnlineValidation(epoch),errorsOnlineTraining(epoch));
    %check early stopping every 5 epoch
    if(mod(epoch,5)==0)
        if(errorsOnlineValidation(epoch)>lastError)
            %early stop
            net=lastNet;
            finalEpochsOnline=epoch;
            break;
        else
            lastNet=net;
            lastError=errorsOnlineValidation(epoch);
        end
    end
end

fprintf("time elapsed for execution: %.2f minutes, epochs:%d\n", floor(toc/60),finalEpochsOnline);

%accuracy test with online network
correct = 0;
    [~, out] = forwardPropagation(net, test_im, @softmax);
    %compute correctly guessed exemples
    for i = 1: size(out{1,2}, 1)
        [~, idx] = max(out{1,2}(i,:));
        if( idx == find( test_lb(i, :) ) )
            correct = correct + 1;
        end
    end
fprintf('online accuracy on test set %2f\n', (correct/10000)*100);

% plotting loss on validation and training in online mode
figure('Name', 'Error');
hold on
warning off;
legend('Error');
title('Online Loss');
xlabel('Epochs');
ylabel('Total Error');
axis auto;
plot(errorsOnlineValidation(1:finalEpochsOnline), 'r','DisplayName','Validation');
plot(errorsOnlineTraining(1:finalEpochsOnline), 'b','DisplayName','Training');
hold off
drawnow;


% plotting loss of online and batch learning with the same number of epochs
figure('Name', 'Error');
hold on
warning off;
legend('Error');
title('Loss on Validation');
xlabel('Epochs');
ylabel('Total Error');
axis auto;
plot(errorsOnlineValidation(1:finalEpochsOnline), 'r','DisplayName','online');
plot(errorsBatchValidation(1:finalEpochsMiniB), 'b','DisplayName','minibatch');
hold off
drawnow;





