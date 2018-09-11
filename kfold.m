%Script utilizzato per scegliere gli iperparametri utilizzando una
    %strategia di cross validation K-Folding.

%acquisizione del training set Mnist e delle label associate utilizzando le funzioni realizzate
    %dagli stessi manutentori del data set.
train_im = loadMNISTImages('train-images.idx3-ubyte');
train_lb = loadMNISTLabels('train-labels.idx1-ubyte');
train_lb = train_lb';

%trasformazione delle label degli esempi etichettati 0 in 10, per motivi di
%praticità sugli indici.
train_lb(train_lb==0) = 10;
%conversione da vettore contenente le label in matrice di target contenente
%vettori riga della forma (0,0,0,0,1,0,0,0,0).
train_lb = dummyvar(train_lb);

%grandezza del campione di training scelta per il k-folding
ts_size = 500;
%parametro per il k-folding, per dividere il training set in k
%sotto-insiemi
k = 5;

%calcolo degli slice del training set da utilizzare per il k-folding
slice_size = int32(ts_size / k);
resized_im = train_im(:, 1:ts_size);
resized_lb = train_lb(1:ts_size, :);

%hyperparametri scelti da testare
etas = [0.007, 0.008, 0.009,0.02, 0.03, 1.e-2];
nodes = [100, 200, 300, 400, 500];

best_error = inf;
best_eta = inf;
best_node = inf;

tic
%inizio della cross validation
for cur_eta = etas
    
    for cur_node = nodes
        
        for i = 0: k-1
            
            %selezione dello slice corrente per la validazione e i restanti
                %per il training
            start_idx = slice_size * i + 1;
            stop_idx = start_idx + slice_size - 1;
                        
            k_train_im = [resized_im(:, 1:start_idx), resized_im(:, stop_idx:ts_size)];
            k_train_lb = [resized_lb(1:start_idx, :); resized_lb(stop_idx:ts_size,:)];
            
            k_test_im = resized_im(:, start_idx:stop_idx);
            k_test_lb = resized_lb(start_idx:stop_idx, :);
                       
            %Crea la rete feedforward con i parametri da testare
            net = feedforwardnet([784, cur_node, 10], {@tanH, @ReLU}, @quadraticCost);
            
            sizeoftrain = size(k_train_im);
            sizeoftrain = sizeoftrain(2);
            
            %Addestramento della rete sui k-1 slice
            for im = 1: sizeoftrain
                net = train(net, k_train_im(:, im)', k_train_lb(im, :)', cur_eta);
            end
            
            guessed = 0;
            error = 0;
            
            sizeoftest = size(k_test_im);
            sizeoftest = sizeoftest(2);
            
            %Misura dell'errore, con forward propagation sul k slice di
            %test
            for t = 1: sizeoftest
                test = propagate(net, k_test_im(:, t)');
                [val, idx] = max(test{end});
                if(idx == find(k_test_lb(t, :)))
                    guessed = guessed + 1;
                end
                %calcolo dell'errore tramite Somma di quadrati.
                error = error + 0.5 * sum((test{end} - k_test_lb(t,:)').^2);
            end
            
            if error < best_error
                best_error = error;
                best_eta = cur_eta;
                best_node = cur_node;
            end
            
            rate = (guessed/sizeoftest) * 100;
            fprintf("guessesed: %d/%d - rate: %.2f%%\n", guessed, sizeoftest, rate);
            fprintf("total error: %.2f\n", error);           
        end 
    end 
end

toc

fprint("------------------------------------");
fprint("best error: %.2f \nbest eta: %f\n best number of nodes: %d\n", best_error, best_eta, best_node);
fprint("------------------------------------");
