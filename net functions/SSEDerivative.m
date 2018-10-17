function out = SSEDerivative(y,t)
%Derivative of sum of squared error
    %y = output layer of the forward propagation
    %t = layer target
    out = y - t;
end

