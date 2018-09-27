function [dWeights, dBiases] = calculateDerivatives(net, delta, input, z)

prev = input;

dWeights = cell(1, net.numOfLayers);
dBiases = cell(1, net.numOfLayers);

for layer = 1: net.numOfLayers
    dBiases{layer} = sum(delta{layer},1);
    dWeights{layer} = delta{layer}' * prev;
    prev = z{layer};
end

end

