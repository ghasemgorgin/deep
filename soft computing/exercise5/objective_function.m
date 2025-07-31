function fit = objective_function(x)
    fit = (1+cos(2*pi*x(1)*x(2)))*exp(-(abs(x(1))+abs(x(2)))/2);
    fit = -fit;  % Convert to minimization problem since PSO minimizes
return;