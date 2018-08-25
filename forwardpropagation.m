function out = forwardpropagation(net, input)
% Funzione che esegue la forward propagation e restituisce l'array
% contenente l'otuput della rete
%
% net: rete neurale
%
% input: input della rete 
    
    % Numero di starti della rete + 1 per l'input
    len = length(net.hiddenSize);
    
    % Array contenente l'output devari strati
    z = cell(len);
    
    % La prima riga conterrà l'input dato alla rete
    z{1} = input;
    
    % Per ogni strato della rete (1 è l'input)
    for l = 2: len
        
        % Numero di nodi nello strato l
        nodes = net.hiddenSize(l);
        
        % Array che conterrà gli output dei nodi dello strato l
        z{l} = zeros(nodes);
        
        % Per ogni nodo dello strato l
        for i = 1: nodes
            
            % Valori di attivazione del nodo al quale verrà applicata 
            % la funzione di output, inizializzata con il valore di bias
            a = net.biases{l - 1}(i);
           
            % Array contenente il peso delle connesioni entranti sul nodo i
            w = net.weights{l-1}(i);
            
            % Per ogni connessione entrante nel nodo i dello strato l
            for j = 1: length(w)
                % Calcolo il contributo del nodo j dello strato l-1
                a = a + w(j) * z{l-1}(j);
            end
            
            % Estraggo la funzione dello strato l 
            fnc = net.outFnc{l-1};
            % Applico la funzione e salvo il riultato nell'array
            z{l}(i) = fnc(a);
           
        end
    end
    
    % Restituisco l'ultimo array di output
    % equivalente all'output della rete
    out = z{len};
end


