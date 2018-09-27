function net = neuralNet(numOfFeatures, layersNodes, ...
                         activationFnc, errorFunction)
% Create an ann feedforeward multilayer fully connected.
%
% numOfFeatures: number indicating the number of input of the ann.
%
% layersNodes: array containing the number of nodes for each layer.
%
% activationFnc: function handler arraycell for activation 
%                functions for each layer
%
%
% errorFunction: cost function used to measure error on output
%
% net: neural network with weights, biases and activation function


% Store the struct (layes, nodes, activation functions, derivates 
% of activation functions and cost function) of the ann
net.numOfFeatures = numOfFeatures;
net.numOfLayers = size(layersNodes, 2);
net.activationFnc = activationFnc;
net.errorFunction = errorFunction;

% activationDerivate: cell array con le derivate delle funzioni d'errore
for i = 1: length(activationFnc)
    if isequal(net.activationFnc{i},@sigmoid)
        fnc = @sigmoidDerivative;
    elseif isequal(net.activationFnc{i},@identity)
        fnc = @identityDerivative;
    elseif isequal(net.activationFnc{i},@tanH)
        fnc = @tanHDerivative;
    elseif isequal(net.activationFnc{i},@ReLU)
        fnc = @ReLUDerivative;
    elseif isequal(net.activationFnc{i},@softmax_a)
        fnc = @softMaxDerivative;
    else
        disp("errore funzioni di attivazione");
        return;
    end
    net.activationDerivative{i} = fnc;
end

% Value used for generate random values 
deviation = max(layersNodes)^(-0.5);

prev = numOfFeatures;

% Generate weights and biases
for i = 1: net.numOfLayers
    % Initialize weights with random value from the normal distribution.
    net.weights{i} = normrnd(0, deviation, layersNodes(i), prev);
    % Initialize biases with zeros
    net.biases{i} = zeros(1, layersNodes(i));
    prev = layersNodes(i);
end

net.numOfOut = prev;

end