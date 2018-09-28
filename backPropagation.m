function [dWeights, dBiases] = backPropagation(net, input, ...
                                               targets)
% Execute a back propagation

layers = net.numOfLayers;

[a, z] = forwardPropagation(net, input, @softmax);

net.z = z;

delta = cell(1, layers);

delta{layers} = net.activationDerivative{layers}(a{layers}) .* ...
                                    net.errorFunction(z{layers}, targets);

for layer = layers -1: -1: 1
    delta{layer} = net.activationDerivative{layer}(a{layer}) .* ...
                   (delta{layer+1} * net.weights{layer+1});
end

[dWeights, dBiases] = calculateDerivatives(net, delta, input, z);

end