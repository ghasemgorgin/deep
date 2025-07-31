clc;
clear all;

% Initial parameters
N = 50;          % Population size
Pc = 0.9;        % Crossover probability
Pm = 0.1;        % Mutation probability
ITER = 50;       % Number of generations
m = 2;           % Number of variables
Lo = [-4 -1.5];  % Lower bounds
Hi = [2 1];      % Upper bounds

% Initialize population (real-valued)
Population = zeros(N, m);
for i = 1:N
    Population(i,:) = Lo + (Hi-Lo).*rand(1,m);
end

best_so_far = [];
Average_fitness = [];

% Main loop
for it = 1:ITER
    % Evaluate fitness
    [selection_probability, fit, ave_fit, max_fit, opt_sol] = real_fit_eval(Population, N, m);
    
    % Update best solution
    if it == 1
        best_so_far(it) = max_fit;
        final_sol = opt_sol;
    elseif max_fit > best_so_far(it-1)
        best_so_far(it) = max_fit;
        final_sol = opt_sol;
    else
        best_so_far(it) = best_so_far(it-1);
    end
    Average_fitness(it) = ave_fit;
    
    % Selection
    mating_pool = real_roulette_wheel(Population, N, selection_probability);
    
    % Crossover
    new_pop = real_crossover(mating_pool, Pc, N, m);
    
    % Mutation
    Population = real_mutation(new_pop, Pm, N, m, Lo, Hi);
end

% Display results
disp("Final solution and optimum fitness:");
disp(final_sol);
disp("Best_so_far:");
disp(best_so_far(end));

% Plot results
x = 1:ITER;
figure;
plot(x, best_so_far, "k", x, Average_fitness, ".-k");
xlabel("Generation");
ylabel("Fitness Function");
legend("Best-so-far", "Average fitness");