function y = softmax(x)
%softmax function
% x: array or single value
    y = exp(x) ./ sum(exp(x),2);
end

