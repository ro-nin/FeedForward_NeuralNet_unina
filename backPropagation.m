function [dWeights, dBiases] = backPropagation(net, input, ...
                                               targets, errorFunctionDx)
% Execute a back propagation

layers = net.numOfLayers;

[a, z] = forwardPropagation(net, input, @softmax);

delta = cell(1, layers);

delta{layers} = net.activationDerivative{layers}(a{layers}) .* ...
                                    errorFunctionDx(z{layers}, targets);

for layer = layers -1: -1: 1
    delta{layer} = net.activationDerivative{layer}(a{layer}) .* ...
                   (delta{layer+1} * net.weights{layer+1});
end

[dWeights, dBiases] = calculateDerivatives(net, delta, input, z);

end