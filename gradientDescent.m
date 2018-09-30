function net = gradientDescent(net, dW, dB, eta)
%update weights with the gradient descent
%
%net: ann to train
%
%dW: gradient relative to regular weights
%
%dB: gradient relative to bias weights
%
%eta: learning rate value

%update weights for each layer
for layer = 1: net.numOfLayers
    net.biases{layer} = net.biases{layer} - eta * dB{layer};
    net.weights{layer} = net.weights{layer} - eta * dW{layer};
end

end

