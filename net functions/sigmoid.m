function y = sigmoid(x)
%SIGMOID
%   x: array di valori o singolo valore
%   y: array di valori o singolo valore

%Costituisce la funzione di output per tutti i
%nodi degli strati interni

    y = 1.0 ./ (1.0 + exp(-x));
end

