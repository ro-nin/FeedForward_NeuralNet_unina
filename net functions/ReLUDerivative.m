function out = ReLUDerivative(x)
%Relu Derivative

if x < 0
    out=0;
else
    out=ones(size(x));

end