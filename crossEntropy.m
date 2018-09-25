function out = crossEntropy(y,t)
%Cross Entropy
    %y= layer di output della forward propagation
    %t= layer target
    %out= (y-t)./((1.-y).*y);
    
    %se presupponiamo SOFTMAX, senn� va quella di sopra
    out = y - t;
end

