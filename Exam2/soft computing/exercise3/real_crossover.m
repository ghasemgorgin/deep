function [new_pop] = real_crossover(mating_pool, Pc, N, m, Lo, Hi)
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
                min_bound_blx = min_val - range * alpha;
                max_bound_blx = max_val + range * alpha;
                
                % Generate offspring values
                offspring1_val = min_bound_blx + (max_bound_blx - min_bound_blx) * rand;
                offspring2_val = min_bound_blx + (max_bound_blx - min_bound_blx) * rand;

                % Enforce global bounds
                new_pop(j,k) = max(Lo(k), min(Hi(k), offspring1_val));
                new_pop(j+1,k) = max(Lo(k), min(Hi(k), offspring2_val));
            end
        end
    end
return;