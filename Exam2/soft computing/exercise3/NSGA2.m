clc;
clear all;

% Parameters
N = 100;         % Population size
Pc = 0.9;        % Crossover probability
Pm = 0.1;        % Mutation probability
ITER = 50;       % Number of generations
m = 2;           % Number of variables (for ZDT1, usually 30 variables are used, but 2 is fine for demonstration)
n_obj = 2;       % Number of objectives
Lo = [0 0];      % Lower bounds for ZDT1
Hi = [1 1];      % Upper bounds for ZDT1

% Initialize population
Population = zeros(N, m);
for i = 1:N
    Population(i,:) = Lo + (Hi-Lo).*rand(1,m);
end

% Arrays for storing results
pareto_front = [];
pareto_solutions = [];

% Main loop
for it = 1:ITER
    % Evaluate objectives
    [obj_values, fronts, crowding_distance] = evaluate_objectives(Population, N);
    
    % Selection
    parents = tournament_selection(Population, fronts, crowding_distance, N);
    
    % Crossover
    offspring = real_crossover(parents, Pc, N, m, Lo, Hi); % Pass Lo, Hi
    
    % Mutation
    offspring = real_mutation(offspring, Pm, N, m, Lo, Hi);
    
    % Combine parents and offspring
    combined_pop = [Population; offspring];
    combined_obj = evaluate_objectives(combined_pop, 2*N);
    
    % Select next generation
    [Population, obj_values] = select_next_generation(combined_pop, combined_obj, N);
    
    % Store Pareto front
    front1_idx = fronts{1};
    pareto_front = obj_values(front1_idx,:);
    pareto_solutions = Population(front1_idx,:);
end

% Plot results
figure;
scatter(obj_values(:,1), obj_values(:,2), 'b.');
hold on;
scatter(pareto_front(:,1), pareto_front(:,2), 'r*');
xlabel('Objective 1');
ylabel('Objective 2');
legend('Population', 'Pareto Front');
title('NSGA-II Results');