function [obj_values, fronts, crowding_distance] = evaluate_objectives(Population, N)
    % Calculate objective values
    obj_values = zeros(N, 2);
    m = size(Population, 2); % Number of variables
    
    for i = 1:N
        x = Population(i,:);
        
        % ZDT1 Objective 1
        obj_values(i,1) = x(1);
        
        % ZDT1 Objective 2
        g_x = 1 + 9 * sum(x(2:m)) / (m-1);
        obj_values(i,2) = g_x * (1 - sqrt(obj_values(i,1) / g_x));
    end
    
    % Non-dominated sorting
    fronts = fast_non_dominated_sort(obj_values);
    
    % Calculate crowding distance
    crowding_distance = calculate_crowding_distance(obj_values, fronts);
end

function fronts = fast_non_dominated_sort(obj_values)
    N = size(obj_values, 1);
    S = cell(N,1);
    n = zeros(N,1);
    rank = zeros(N,1);
    fronts = {[]};
    
    for i = 1:N
        S{i} = [];
        n(i) = 0;
        for j = 1:N
            if i ~= j
                if dominates(obj_values(i,:), obj_values(j,:))
                    S{i} = [S{i} j];
                elseif dominates(obj_values(j,:), obj_values(i,:))
                    n(i) = n(i) + 1;
                end
            end
        end
        if n(i) == 0
            rank(i) = 1;
            fronts{1} = [fronts{1} i];
        end
    end
    
    i = 1;
    while ~isempty(fronts{i})
        Q = [];
        for j = fronts{i}
            for k = S{j}
                n(k) = n(k) - 1;
                if n(k) == 0
                    rank(k) = i + 1;
                    Q = [Q k];
                end
            end
        end
        i = i + 1;
        fronts{i} = Q;
    end
end

function d = calculate_crowding_distance(obj_values, fronts)
    N = size(obj_values, 1);
    d = zeros(N,1);
    
    for i = 1:length(fronts)
        front = fronts{i};
        if length(front) > 2
            for obj = 1:2
                [sorted_obj, idx] = sort(obj_values(front,obj));
                d(front(idx(1))) = Inf;
                d(front(idx(end))) = Inf;
                
                for j = 2:length(front)-1
                    d(front(idx(j))) = d(front(idx(j))) + ...
                        (sorted_obj(j+1) - sorted_obj(j-1)) / ...
                        (max(sorted_obj) - min(sorted_obj));
                end
            end
        elseif ~isempty(front)
            d(front) = Inf;
        end
    end
end

function d = dominates(x,y)
    d = all(x <= y) && any(x < y);
end