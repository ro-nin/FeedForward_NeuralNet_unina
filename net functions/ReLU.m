function out = ReLU(x)
%Funzione di attivazione rectified linear unit.
if x < 0
    out=0;
else
    out=x;

end

