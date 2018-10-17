function deviation = calculateError(out, lb, errorFnc) 
%Evaluate mean error, variance and then standard deviation 
%out: contains all the outputs layers coming from the test made
%lb: contains the label matrix
%errorFnc: the specified error formula relative to the error function used
%   on the output layer
N = size(out, 1);
errors = errorFnc(out, lb);
%error on multiple tests
mu = sum(errors) / N;
variance = sum((errors - mu) .^ 2) / N;
deviation = sqrt(variance);
end