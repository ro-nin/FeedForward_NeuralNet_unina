function out = forwardpropagation(net, input)
% Funzione che esegue la forward propagation e restituisce l'array
% contenente l'output della rete
%
% net: rete neurale
%
% input: input della rete espresso come vettore colonna
    
    % Numero di strati della rete + 1 per l'input
    len = length(net.hiddenSize);
 
    % Array contenente l'output devari strati
    z = cell(len, 1);

    % La prima riga conterrà l'input dato alla rete
    z{1} = [input; 1];
    
    % Per ogni strato della rete (1 è l'input)
    for l = 2: len
        
        % Numero di nodi nello strato l
        nodes = net.hiddenSize(l) + 1;
        
        % Array contenente il peso delle connesioni entranti sul nodo i
        w = net.weights{l-1};
        % Calcolo il contributo del nodo j dello strato l-1
        a =  w * z{l-1};
        
        % Estraggo la funzione dello strato l 
        fnc = net.outFnc{l-1};
        % Applico la funzione e salvo il riultato nell'array
        z{l} = [fnc(a); 1];
    end
    
    % Restituisco gli array di output
    % equivalente all'output della rete, rimuovendo l'ultimo valore
    % usato come fattore moltiplicativo del bias
    z{len}(nodes) = [];
    out = z;
end
