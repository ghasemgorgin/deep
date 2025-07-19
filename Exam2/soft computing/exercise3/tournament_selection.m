function selected = tournament_selection(Population, fronts, crowding_distance, N)
    selected = zeros(size(Population));
    for i = 1:N
        idx1 = randi(N);
        idx2 = randi(N);
        
        rank1 = find(cellfun(@(x) any(x == idx1), fronts));
        rank2 = find(cellfun(@(x) any(x == idx2), fronts));
        
        if rank1 < rank2
            selected(i,:) = Population(idx1,:);
        elseif rank2 < rank1
            selected(i,:) = Population(idx2,:);
        else
            if crowding_distance(idx1) > crowding_distance(idx2)
                selected(i,:) = Population(idx1,:);
            else
                selected(i,:) = Population(idx2,:);
            end
        end
    end
end