function out = SSE(y,t)
%Sum of squared errorrs
    %y = output layer of the forward propagation
    %t = layer target
    out = 0.5 * sum((y-t).^2);
end

