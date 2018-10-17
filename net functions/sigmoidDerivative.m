function y = sigmoidDerivative(x)
%SIGMOID derivative 
%   x: single value or array

%evaluate derivative of sigmoid function


    y = sigmoid(x) .* (1-sigmoid(x));

end

