function [new_pop] = real_crossover(mating_pool, Pc, N, m)
    alpha = 0.5;  % BLX-alpha parameter
    parent_num = randperm(N);
    new_pop = mating_pool;
    
    for j = 1:2:N
        if rand < Pc
            parent1 = mating_pool(parent_num(j),:);
            parent2 = mating_pool(parent_num(j+1),:);
            
            % BLX-alpha crossover
            for k = 1:m
                min_val = min(parent1(k), parent2(k));
                max_val = max(parent1(k), parent2(k));
                range = max_val - min_val;
                min_bound = min_val - range * alpha;
                max_bound = max_val + range * alpha;
                
                new_pop(j,k) = min_bound + (max_bound - min_bound) * rand;
                new_pop(j+1,k) = min_bound + (max_bound - min_bound) * rand;
            end
        end
    end
return;