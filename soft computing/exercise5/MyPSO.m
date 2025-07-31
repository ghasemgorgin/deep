clc;
clear all;

% Parameters
N = 50;          % Population size (number of particles)
ITER = 50;       % Number of iterations
m = 2;           % Number of variables
w = 0.7;         % Inertia weight
c1 = 2;          % Personal learning coefficient
c2 = 2;          % Global learning coefficient
Lo = [-4 -1.5];  % Lower bounds (same as exercise 1)
Hi = [2 1];      % Upper bounds (same as exercise 1)

% Initialize positions and velocities
Position = zeros(N, m);
Velocity = zeros(N, m);
for i = 1:N
    Position(i,:) = Lo + (Hi-Lo).*rand(1,m);
    Velocity(i,:) = -1 + 2*rand(1,m);  % Initial velocity between -1 and 1
end

% Initialize personal best
pbest = Position;
pbest_fit = zeros(N,1);
for i = 1:N
    pbest_fit(i) = objective_function(pbest(i,:));
end

% Initialize global best
[gbest_fit, idx] = min(pbest_fit);
gbest = pbest(idx,:);

% Arrays for storing results
best_so_far = [];
average_fitness = [];

% Main loop
for it = 1:ITER
    % Update particles
    for i = 1:N
        % Update velocity
        r1 = rand(1,m);
        r2 = rand(1,m);
        Velocity(i,:) = w*Velocity(i,:) + ...
                       c1*r1.*(pbest(i,:) - Position(i,:)) + ...
                       c2*r2.*(gbest - Position(i,:));
        
        % Update position
        Position(i,:) = Position(i,:) + Velocity(i,:);
        
        % Bound position
        Position(i,:) = max(Lo, min(Hi, Position(i,:)));
        
        % Update personal best
        fit = objective_function(Position(i,:));
        if fit < pbest_fit(i)
            pbest_fit(i) = fit;
            pbest(i,:) = Position(i,:);
            
            % Update global best
            if fit < gbest_fit
                gbest_fit = fit;
                gbest = Position(i,:);
            end
        end
    end
    
    % Store results
    best_so_far(it) = gbest_fit;
    current_fits = zeros(N,1);
    for i = 1:N
        current_fits(i) = objective_function(Position(i,:));
    end
    average_fitness(it) = mean(current_fits);
end

% Display results
disp("Final solution:");
disp(gbest);
disp("Best fitness:");
disp(gbest_fit);

% Plot convergence
x = 1:ITER;
figure;
plot(x, best_so_far, "k", x, average_fitness, ".-k");
xlabel("Iteration");
ylabel("Fitness Function");
legend("Best-so-far", "Average fitness");
title("PSO Results");