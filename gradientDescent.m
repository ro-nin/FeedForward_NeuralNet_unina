function [net, deltaW, deltaB] = gradientDescent(net, dW, dB, eta)

deltaW = cell(1, net.numOfLayers);
deltaB = cell(1, net.numOfLayers);

for layer = 1: net.numOfLayers
    deltaW{layer} = eta * dW{layer};
    deltaB{layer} = eta * dB{layer};
    net.biases{layer} = net.biases{layer} - deltaB{layer};
    net.weights{layer} = net.weights{layer} - deltaW{layer};
end

end

