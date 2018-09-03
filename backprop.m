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
    
    
    
    
    for i = 1: N
       x = images(:, i);       
       delta = cell(len, 1);
       y = forwardpropagation(net, x);
       
       % disp(y{len});
       % disp(targets(:,i))
       deltaKout = y{len} - targets(:,i);        
       % disp(deltaKout);
       deltaK
       
      
       
       
    end
end







