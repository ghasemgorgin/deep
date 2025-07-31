clc;
clear all;

% Parameters
N = 50;         % Population size
Pc = 0.9;       % Crossover probability
Pm = 0.1;       % Mutation probability
ITER = 50;      % Number of generations
m = 2;          % Number of variables
Lo = [0 0];     % Lower bounds
Hi = [2 2];     % Upper bounds

% Initialize population
Population = zeros(N, m);
for i = 1:N
    Population(i,:) = Lo + (Hi-Lo).*rand(1,m);
end

best_so_far = [];
Average_fitness = [];

for it = 1:ITER
    % Evaluate fitness with penalty
    [fit, ave_fit, min_fit, opt_sol] = constraint_fit_eval(Population, N, m);
    
    % Update best solution
    if it == 1
        best_so_far(it) = min_fit;
        final_sol = opt_sol;
    elseif min_fit < best_so_far(it-1)
        best_so_far(it) = min_fit;
        final_sol = opt_sol;
    else
        best_so_far(it) = best_so_far(it-1);
    end
    Average_fitness(it) = ave_fit;
    
    % Selection (roulette wheel based on inverse fitness)
    selection_probability = 1./(fit + 1e-6); % To avoid division by zero
    selection_probability = selection_probability / sum(selection_probability);
    mating_pool = real_roulette_wheel(Population, N, selection_probability);
    
    % Crossover
    new_pop = real_crossover(mating_pool, Pc, N, m, Lo, Hi);
    
    % Mutation
    Population = real_mutation(new_pop, Pm, N, m, Lo, Hi);
end

disp("Final solution and optimum penalized fitness:");
disp(final_sol);
disp("Best_so_far (min penalized fitness):");
disp(best_so_far(end));

% Plot results
x = 1:ITER;
figure;
plot(x, best_so_far, "k", x, Average_fitness, ".-k");
xlabel("Generation");
ylabel("Penalized Fitness (lower is better)");
legend("Best-so-far", "Average fitness");
title("GA with Constraint Handling (Penalty Function)");