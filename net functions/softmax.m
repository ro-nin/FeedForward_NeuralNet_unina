function y = softmax(x)
    y = exp(x) ./ sum(exp(x),2);
end

