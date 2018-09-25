function out = propagate(net, input)
%Funzione per la forward propagation
    %net: la rete neurale da utilizzare.
    %input: vettore contenente gli input.

%cell array che conterrà l'output degli strati, ed in prima posizione gli input    
out = cell(length(net.hiddenSizes), 1);
out{1} = input';

%calcolo valori di attivazione per ogni nodo degli strati
for i = 1: length(net.hiddenSizes) - 1
    out{i+1} = net.trainFnc{i}((net.weights{i} * out{i}) + net.biases{i});
    if isequal(net.trainFnc{i},@softmax_a)
        %completa softmax
        out{i+1} = out{i+1}/sum(out{i+1});
    end
end

end