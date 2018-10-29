function [a, z] = forwardPropagation(net, input, afterProcessFunction)
% Forward propagation of the input through the ann
%
% net: ann created with neuralNet function
%
% input: matrix of the inputs, each row is an element and each column is
%        a feature of the example.
%
% [a, z]: vector of arraycell.
%   z:output nodes after activation function
%   a:otput nodes before activation function

a = cell(1, net.numOfLayers);
z = cell(1, net.numOfLayers);

prevOut = input;
for layer = 1: net.numOfLayers
    %evaluate each node doing a product between incoming edges and outputs
    %and then add the bias value
    a{layer} = (prevOut * net.weights{layer}') + net.biases{layer};
    %apply the activation function on the value
    z{layer} = net.activationFnc{layer}(a{layer});
    prevOut = z{layer};
end
%apply post process (like softmax)
if exist('afterProcessFunction','var')
    z{net.numOfLayers} = afterProcessFunction(z{net.numOfLayers});
end

end