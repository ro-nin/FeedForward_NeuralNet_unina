function y = crossEntropy(output,target)
%CROSS ENTROPY
%   *output: single value or array
%   *target: element's labels
%Evaluate Cross Entropy Value.


    tmp = target;
    tmp(output > 0) = target(output > 0) .* log(output(output>0));
    tmp(output <= 0) = target(output <= 0) .* log(realmin('single'));
    y = -sum(tmp);
end

