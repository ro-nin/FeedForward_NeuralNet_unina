function y = crossEntropyDerivative(X, Y)
%CROSSENTROPYDX 
%   *X: singolo valore o array
%   *Y: singolo valore o array 

%Derivata della Cross Entropy nel caso in cui
%si utilizzi SoftMax per il post-processing della rete

    y = X - Y;

end

