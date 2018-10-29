function [dWeights, dBiases] = calculateDerivatives(net, delta, input, z)
%compute derivatives witch each delta for the gradients
%
%net: the ann to train, see neuralNet.m
%
%delta: stores the delta value for each node layer by layer
%
%input: the training set matrix.
%
%z: nodes output value

%start with the input as first layer
prev = input;

dWeights = cell(1, net.numOfLayers);
dBiases = cell(1, net.numOfLayers);
%compute derivative of the weights for each layer's weight and bias
for layer = 1: net.numOfLayers
    dBiases{layer} = sum(delta{layer},1);
    dWeights{layer} = delta{layer}' * prev;
    %store for next iteration
    prev = z{layer};
end

end

