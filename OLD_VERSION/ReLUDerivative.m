function out = ReLUDerivative(x)
%funzione derivata di ReLU
if x < 0
    out=0;
else
    out=1;

end