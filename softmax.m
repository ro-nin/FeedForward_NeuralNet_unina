function out = softmax(x)
% Stable softmax function
    shiftx = x - max(x);
    exps = exp(shiftx);
    out = exps / sum(exps);
end