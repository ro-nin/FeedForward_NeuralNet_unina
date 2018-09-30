function out = train(net, input, targets, eta, TsSize, batchSize)
% Train the ANN
%
%net: the ann to train, see neuralNet.m
%
%input: the training set matrix.
%
%targets: matrix with the examples labels.
%
%eta: learning rate value.
%
%TsSize: size of the training set: number of examples.
%
%BatchSize: how many examples per weight adjustment, values:
%           1 = online learning,
%           TsSize = batch,
%           1< BatchSize < TsSize = minibatch. 
%

%iterate with batchSize step until the end of training set, feeding the
%portions of the training set for minibatch or the entire ts for the online
%mode
 for i = 1: batchSize: TsSize
    
    [dWeights, dBiases] = ...
        backPropagation(net, input(i: i+batchSize-1, :), targets(i:i+batchSize-1, :));
    net = gradientDescent(net, dWeights, dBiases, eta);
 end
 
    out = net;
    
end