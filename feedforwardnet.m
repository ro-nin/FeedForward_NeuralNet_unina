function net = feedforwardnet(hiddenSizes, outFnc)
% Crea una rete feed forward full connected
%
% hiddenSize: array contenente in numero di neuroni per ogni strato,
%             il primo numero equivale a gli iput della rete
%
% outFnc: cell array contenente le funzioni di output di ogni strato
%
% output: struct che rappresenta la rete

    % Salvo il numero di input e il numero di neuroni per ogni strato
    net.hiddenSize = hiddenSizes;

    % Salvo l'array contenente le funzioni dei vari strati
    net.outFnc = outFnc;
    
    % len contiene il numero di strati della rete più 1 per l'input
    len = length(hiddenSizes);
    
    for i = 2 : len
        % per ciascuno strato creo una matrice di pesi tale per ogni 
        % nodo dello strato (i-1)-esimo (colonna) +1 per il bias 
        % ci siano una connesione con il nodo dello strato i-esimo (riga)
        net.weights{i-1} = rand(hiddenSizes(i), hiddenSizes(i-1) + 1);
    end
end
