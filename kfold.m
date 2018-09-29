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

% Dimension of training size
ts_size = 100;
% Number of folds
k = 10;

%calcolo degli slice del training set da utilizzare per il k-folding
slice_size = int32(ts_size / k);
resized_im = train_im(1:ts_size, :);
resized_lb = train_lb(1:ts_size, :);

% Fixed hyperparam
errorDerivative = @crossEntropyDerivative;
errorFnc = @crossEntropy;

% Hyperparameters to test
netFnc = {{@tanH, @identity}, {@sigmoid, @identity}, {@sigmoid, @identity}, {@tanH, @ReLU}};
netNodes = [250, 500, 800];
netEtas = [0.1, 0.01, 0.001];

currError=0;
fprintf("Hidden function; Output function; Eta;	Hidden nodes; Mean Accuracy; C.E. Standard deviation\n");

for fnc = 1: length(netFnc)
    deviationsNodes = cell(length(netNodes),1);
    nodeCounter=0;
    for node = netNodes
        nodeCounter=nodeCounter+1;
        deviationsEtas=cell(length(netEtas),1);
        etaCounter=0;
        for eta = netEtas
            etaCounter=etaCounter+1;
            k_error = zeros(k, 1);
            k_accuracy = zeros(k, 1);
            for i = 0: k-1
                % Calculate the slice index for testing and for validation
                start_idx = slice_size * i + 1;
                stop_idx = start_idx + slice_size - 1;
                %cut the submatrix for the training set
                k_train_im = [resized_im(1:start_idx-1, :); resized_im(stop_idx+1:ts_size, :)];
                k_train_lb = [resized_lb(1:start_idx-1, :); resized_lb(stop_idx+1:ts_size, :)];
                %cut the submatrix for the validation set
                k_test_im = resized_im(start_idx:stop_idx, :);
                k_test_lb = resized_lb(start_idx:stop_idx, :);
                
                %train the network
                net = neuralNet(784, [node, 10], netFnc{fnc}, errorDerivative);
                net = train(net, k_train_im, k_train_lb, eta, size(k_train_im, 1), 1);
                
                guessed=0;
                currError=0;
                
                %test on the validation part
                [a,z] = forwardPropagation(net, k_test_im);
                for n = 1: size(z{1,2}, 1)
                    [val, idx] = max(z{1,2}(n,:));
                    if( idx == find( k_test_lb(n, :) ) )
                        guessed = guessed + 1;
                    end
                    %currError = currError + (sum( log(z{1,2}(n,:)) .* k_test_lb(n, :) ));
                    currError = currError + sum(errorFnc(z{1,2}(n,:),k_test_lb(n, :)));
                    %TODO: standard Deviation?
                end
                
                k_error(i+1) = currError;
                accuracy = guessed / size(k_test_im, 1) * 100;
                k_accuracy(i+1) = accuracy;
            end
            mean_accuracy = sum(k_accuracy) / k;
            
            mean = sum(k_error);
            variance = (1/(k-1)) * (sum((k_error - mean).^2));
            deviation = sqrt(variance);
            
            fnc1 = func2str(netFnc{fnc}{1});
            fnc2 = func2str(netFnc{fnc}{2});
            fprintf("%s; %s; %.3f; %d; %.2f; %f\n", fnc1, fnc2, eta, node, mean_accuracy, deviation);
            deviationsEtas{etaCounter}=deviation;
        end
        deviationsNodes{nodeCounter}=deviationsEtas;
    end
    %plot the current functions
    
    figure('Name',strcat('K-Fold:',fnc1,'-',fnc2));
    c = categorical({'250','500','800'});
    y = vertcat(cell2mat(deviationsNodes{1})' , cell2mat(deviationsNodes{2})',cell2mat(deviationsNodes{3})');
    b = bar(c,y);
end

