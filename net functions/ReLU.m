function out = ReLU(x)
%rectified linear unit activation.
if x < 0
    out=0;
else
    out=x;

end

