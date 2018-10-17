function out = tanH(x)
%Hyperbolic tangent activation
% x: single value or array
out= (exp(x)-exp(-x)) ./ (exp(x)+exp(-x));
end

