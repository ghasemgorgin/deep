function [fit, ave_fit, min_fit, opt_sol] = constraint_fit_eval(Population, N, m)
    penalty_coeff = 1e4; % Penalty coefficient
    fit = zeros(N,1);
    for i = 1:N
        x = Population(i,:);
        f = x(1)^2 + x(2)^2; % Objective function (minimize)
        g = x(1) + x(2) - 1; % Constraint: should be <= 0
        penalty = 0;
        if g > 0
            penalty = penalty_coeff * g^2;
        end
        fit(i) = f + penalty;
    end
    ave_fit = mean(fit);
    [min_fit, min_loc] = min(fit);
    opt_sol = Population(min_loc,:);
return;