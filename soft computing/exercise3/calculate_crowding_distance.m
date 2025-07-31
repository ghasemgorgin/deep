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