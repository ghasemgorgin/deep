function [selection_probability, fit, ave_fit, max_fit, opt_sol] = real_fit_eval(Population, N, m)
    for i = 1:N
        x(1) = Population(i,1);
        x(2) = Population(i,2);
        fit(i) = (1+cos(2*pi*x(1)*x(2)))*exp(-(abs(x(1))+abs(x(2)))/2);
    end
    selection_probability = fit/sum(fit);
    ave_fit = mean(fit);
    [max_fit, max_loc] = max(fit);
    opt_sol = Population(max_loc,:);
return;