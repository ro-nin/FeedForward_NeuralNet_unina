function deviation = calculateError(out, lb, errorFnc)  
N = size(out, 1);
errors = errorFnc(out, lb);
mu = sum(errors) / N;
variance = sum((errors - mu) .^ 2) / N;
deviation = sqrt(variance);
end