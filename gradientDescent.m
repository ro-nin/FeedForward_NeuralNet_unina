function net = gradientDescent(net, dW, dB, eta)

for layer = 1: net.numOfLayers
    net.biases{layer} = net.biases{layer} - eta * dB{layer};
    net.weights{layer} = net.weights{layer} - eta * dW{layer};
end

end

