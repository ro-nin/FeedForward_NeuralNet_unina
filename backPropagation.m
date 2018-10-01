function [dWeights, dBiases] = backPropagation(net, input, ...
                                               targets)
% Execute a back propagation
%
%net: the ann to train, see neuralNet.m
%
%input: the training set matrix.
%
%targets: matrix with the examples labels.
%
%

layers = net.numOfLayers;
[a, z] = forwardPropagation(net, input, @softmax);

delta = cell(1, layers);
%compute deltas of the output layer
delta{layers} = net.activationDerivative{layers}(a{layers}) .* ...
                                    net.errorFunctionDerivative(z{layers}, targets);
%compute deltas of the remaining layers backward
for layer = layers -1: -1: 1
    delta{layer} = net.activationDerivative{layer}(a{layer}) .* ...
                   (delta{layer+1} * net.weights{layer+1});
end
%compute derivatives with each delta
[dWeights, dBiases] = calculateDerivatives(net, delta, input, z);

end