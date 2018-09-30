function net = neuralNet(numOfFeatures, layersNodes, ...
                         activationFnc, errorDerivative)
% Create an ann feedforward multilayer full connected.
%
% numOfFeatures: number indicating the number of input of the ann.
%
% layersNodes: array containing the number of nodes for each layer.
%
% activationFnc: function handler arraycell for activation 
%                functions for each layer
%
% errorDerivative: cost function derivative handle for the output layer
%
% net: neural network with weights, biases and activation function


% Store the struct (layes, nodes, activation functions, derivates 
% of activation functions and cost function) of the ann
net.numOfFeatures = numOfFeatures;
net.numOfLayers = size(layersNodes, 2);
net.activationFnc = activationFnc;
net.errorFunction = errorDerivative;

%link every activation function with corresponding derivative
for i = 1: length(activationFnc)
    if isequal(net.activationFnc{i},@sigmoid)
        fnc = @sigmoidDerivative;
    elseif isequal(net.activationFnc{i},@identity)
        fnc = @identityDerivative;
    elseif isequal(net.activationFnc{i},@tanH)
        fnc = @tanHDerivative;
    elseif isequal(net.activationFnc{i},@ReLU)
        fnc = @ReLUDerivative;
    elseif isequal(net.activationFnc{i},@softmax)
        fnc = @softMaxDerivative;
    else
        disp("error on activation function");
        return;
    end
    net.activationDerivative{i} = fnc;
end

% Value used for generate random values 
deviation = max(layersNodes)^(-0.5);

prev = numOfFeatures;

% Generate weights and biases
for i = 1: net.numOfLayers
    % Initialize weights with random value from the normal distribution (maybe dependent on toolbox).
    net.weights{i} = normrnd(0, deviation, layersNodes(i), prev);
    % Initialize biases with zeros
    net.biases{i} = zeros(1, layersNodes(i));
    prev = layersNodes(i);
end

net.numOfOut = prev;

end