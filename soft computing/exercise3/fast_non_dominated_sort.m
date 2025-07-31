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