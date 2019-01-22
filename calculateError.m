function output = calculateError(out, lb, errorFnc) 

%output: contains all the outputs layers coming from the test made
%lb: contains the label matrix
%errorFnc: the specified error formula relative to the error function used
%   on the output layer
errors = errorFnc(out, lb);

output=sum(errors);
end