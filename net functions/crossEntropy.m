function y = crossEntropy(output,target)
%CROSSENTROPY
%   *output: singolo valore o array
%   *target: etichette degli elementi

%Riferimenti: lezioni frontali, Bishop


%Calcola il valore della cross entropy.


    tmp = target;
    tmp(output > 0) = target(output > 0) .* log(output(output>0));
    tmp(output <= 0) = target(output <= 0) .* log(realmin('single'));
    y = -sum(tmp);
end

