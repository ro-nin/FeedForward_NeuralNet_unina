function [a, z] = forwardPropagation(net, input, afterProcessFunction)
% Forward propagation of the input through the ann
%
% net: ann created with neuralNet function
%
% input: matrix of the inputs, each row is an element and each column is
%        a feature of the example. 
%
% [a, z]: vector of arraycell.

a = cell(1, net.numOfLayers);
z = cell(1, net.numOfLayers);

prevOut = input;

for layer = 1: net.numOfLayers
    a{layer} = (prevOut * net.weights{layer}') + net.biases{layer};
    z{layer} = net.activationFnc{layer}(a{layer});
    prevOut = z{layer};
end

if exist('afterProcessFunction','var')
    z{net.numOfLayers} = afterProcessFunction(z{net.numOfLayers});
end


end


