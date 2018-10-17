function y = sigmoid(x)
%SIGMOID activation
%   x: array or single value

    y = 1.0 ./ (1.0 + exp(-x));
end

