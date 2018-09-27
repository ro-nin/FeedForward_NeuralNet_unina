function net = neuralNet(numOfFeatures, layersNodes, ...
                         activationFnc, activationDerivative)
% Create an ann feedforeward multilayer fully connected.
%
% numOfFeatures: number indicating the number of input of the ann.
%
% layersNodes: array containing the number of nodes for each layer.
%
% activationFnc: function handler arraycell for activation 
%                functions for each layer
%
% activationDerivate: functions handler arraycell for the derivates
%					  of activation function, used in backpropagation
%
% costFnc: cost function used to measure error on output
%
% net: neural network with weights, biases and activation function


% Store the struct (layes, nodes, activation functions, derivates 
% of activation functions and cost function) of the ann
net.numOfFeatures = numOfFeatures;
net.numOfLayers = size(layersNodes, 2);
net.activationFnc = activationFnc;
net.activationDerivative = activationDerivative;

% Value used for generate random values 
deviation = max(layersNodes)^(-0.5);

prev = numOfFeatures;

% Generate weights and biases
for i = 1: net.numOfLayers
    % Initialize weights with random value from the normal distribution.
%     net.weights{i} = -0.09 * rand(layersNodes(i), prev);
%     
%     if (mod(rand(1,1), 2) == 1 )
%         net.weights{i} = net.weights{i} .* -1;
%     end
    
    net.weights{i} = normrnd(0, deviation, layersNodes(i), prev);
    % Initialize biases with zeros
    net.biases{i} = zeros(1, layersNodes(i));
    prev = layersNodes(i);
end

net.numOfOut = prev;

end