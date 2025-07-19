function [Population] = real_mutation(new_pop, Pm, N, m, Lo, Hi)
    Population = new_pop;
    for i = 1:N
        for j = 1:m
            if rand < Pm
                % Random mutation within bounds
                Population(i,j) = Lo(j) + (Hi(j)-Lo(j))*rand;
            end
        end
    end
return;