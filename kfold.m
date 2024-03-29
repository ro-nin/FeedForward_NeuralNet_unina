clc;
clear;

addpath('./mnist/');
addpath('./utils/');
addpath('./net functions/');

%load the training set
% 60000 (examples) x 784 (features)
train_im = loadMNISTImages('train-images.idx3-ubyte')';
%load the training set labels
% 60000 x 10
train_lb = loadMNISTLabels('train-labels.idx1-ubyte');
train_lb = train_lb';

train_lb(train_lb==0) = 10;
train_lb = dummyvar(train_lb);

%load the test set
% 784 x 10000
test_im = loadMNISTImages('t10k-images.idx3-ubyte')';
% 10000 x 10
test_lb = loadMNISTLabels('t10k-labels.idx1-ubyte');
test_lb = test_lb';

test_lb(test_lb==0) = 10;
test_lb = dummyvar(test_lb);

% Dimension of total training+validation set for k-fold, default: 300
ts_size = 3200;
% Number of folds
k = 10;

epochNumber=50;
batchSize=32;

%compute slice size for k-folding
slice_size = int32(ts_size / k);
resized_im = train_im(1:ts_size, :);
resized_lb = train_lb(1:ts_size, :);

% Fixed hyperparams
errorDerivative = @crossEntropyDerivative;
errorFnc = @crossEntropy;

% Hyperparameters to test
netFnc = {{@tanH, @identity}, {@sigmoid, @identity}, ...
          {@tanH, @ReLU},{@sigmoid,@sigmoid}};
netNodes = [250, 500, 800];
netEtas = [0.1,0.01, 0.001, 0.0001];

fprintf("Hidden function-Output function; Eta; Hidden nodes; Mean Accuracy; Mean error; Std Acc; Std Error\n");

bestAcc = -Inf;
bestErr = Inf;

elapsedTime = 0;
tic
%for each hyper param:
for fnc = 1: length(netFnc)
    %used to store current error values for plotting
    accNode = cell(length(netNodes),1);
    nodeCounter=0;
    for node = netNodes
        %used to store current error values for plotting
        nodeCounter=nodeCounter+1;
        accEtas=cell(length(netEtas),1);
        etaCounter = 0;
        for eta = netEtas
            etaCounter=etaCounter+1;
            %store current error and accuracy for the k slice
            k_error = zeros(k, 1);
            k_accuracy = zeros(k, 1);
            %rotate the leave one out k validation part
            for i = 0: k-1
                % Calculate the slice index for testing and
                % validation part
                start_idx = slice_size * i + 1;
                stop_idx = start_idx + slice_size - 1;
                %cut the submatrix for the training set
                k_train_im = [resized_im(1:start_idx-1, :); resized_im(stop_idx+1:ts_size, :)];
                k_train_lb = [resized_lb(1:start_idx-1, :); resized_lb(stop_idx+1:ts_size, :)];
                %cut the submatrix for the validation set
                k_test_im = resized_im(start_idx:stop_idx, :);
                k_test_lb = resized_lb(start_idx:stop_idx, :);
                
                %train the network with current params
                net = neuralNet(784, [node, 10], netFnc{fnc}, errorDerivative);
                %net = train(net, k_train_im, k_train_lb, eta, size(k_train_im, 1), 1);
                for epoch = 1: epochNumber
                    net = train(net, k_train_im, k_train_lb, eta, size(k_train_im, 1), batchSize);
                end
                
                %correctly predicted examples
                guessed = 0;
                %value storing computed error
                currError = 0;
                
                %test on the validation part
                [~, z] = forwardPropagation(net, k_test_im, @softmax);
                %compute guessed examples and errore
                for n = 1: size(z{1,2}, 1)
                    [val, idx] = max(z{1,2}(n,:));
                    if( idx == find( k_test_lb(n, :) ) )
                        guessed = guessed + 1;
                    end
                    currError = currError + sum(errorFnc(z{1,2}(n,:),k_test_lb(n, :)));
                end
                %store current error for compute average between k rotations later
                k_error(i+1) = currError;
                k_accuracy(i+1) = guessed / size(k_test_im, 1) * 100;
            end
            %compute mean between al the k rotations
            mean_accuracy =mean(k_accuracy);
            mean_error = mean(k_error);
            stdAcc = std(k_accuracy);
            stdError = std(k_error);
            
            fnc1 = func2str(netFnc{fnc}{1});
            fnc2 = func2str(netFnc{fnc}{2});
            str = sprintf("%s-%s; %.5f; %d; %.2f; %.2f; %.2f; %.2f\n", fnc1, fnc2, eta, node, mean_accuracy, mean_error,stdAcc,stdError);
            fprintf("%s", str);
            accEtas{etaCounter} = mean_accuracy;
            
            if (mean_accuracy > bestAcc)
                bestAccString = str;
                bestAcc = mean_accuracy;
            end
            if (mean_error < bestErr && mean_error > 0)
                bestErrString = str;
                bestErr = mean_error;
            end
        end
        accNode{nodeCounter}=accEtas;
    end
    
    elapsedTime = elapsedTime + toc;
    
    %plot the current functions
    plotBar(fnc1, fnc2, netNodes, netEtas, accNode);
end

fprintf('\nBest Accuracy: %s\n', bestAccString);
fprintf('Best Error: %s\n', bestErrString);

fprintf('Execution time: %d minutes\n', floor(elapsedTime/60));

