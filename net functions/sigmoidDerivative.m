function y = sigmoidDerivative(x)
%SIGMOIDDX 
%   x: valore numerico o array
%   y: Array o valore numerico della
%      derivata della funzione sigmoide con argomento x

%Permete di calcolare la derivata della funzione sigmoide
%su singolo valore o array


    y = sigmoid(x) .* (1-sigmoid(x));

end

