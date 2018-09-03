function backprop(net, imagename, labelname)
    
    images = loadMNISTImages('train-images.idx3-ubyte');
    labels = loadMNISTLabels('train-labels.idx1-ubyte');
    
    % disp(images);
    % display_network(images(:,1:100)); % Show the first 100 images
    % disp(labels(1:10));
    
    N = size(images);
    N = N(2);
    
    % numero di strati della rete
    len = length(net.hiddenSize);
    
    targets = zeros(10, N);
  
    for i = 1: N
        targets(labels(i, 1)+1, i) = 1;
    end
    
    
    
    
    for i = 1: 1
       x = images(:, i);       
       delta = cell(len, 1);
       y = forwardpropagation(net, x);
       %TODO ADD BIAS
       % disp(y{len});
       % disp(targets(:,i))
       deltaKout = y{len} - targets(:,i);        
       % disp(deltaKout);
       zHn = y{1:len-1};
       OneMinuszHn = 1-zHn;
       deltaKhidden = zHn .* OneMinuszHn;
       
       last=deltaKout;
       for j = 1:len-1
           currLen = len-j;
           wKh = net.weights{len-j};
           temp=0;
           for h = 1:net.hiddenSize(currLen+1)
                column=wKh(1:net.hiddenSize(currLen+1),h);
                disp(column);
                temp=temp+ sum((column.*last));
           end
           deltaKhidden=deltaKhidden .* temp;
       end
       
    end
end







