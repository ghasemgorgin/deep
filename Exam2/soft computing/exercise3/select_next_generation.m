function [new_pop, new_obj] = select_next_generation(combined_pop, combined_obj, N)
    fronts = fast_non_dominated_sort(combined_obj);
    new_pop = [];
    new_obj = [];
    i = 1;
    
    while size(new_pop,1) + length(fronts{i}) <= N
        new_pop = [new_pop; combined_pop(fronts{i},:)];
        new_obj = [new_obj; combined_obj(fronts{i},:)];
        i = i + 1;
    end
    
    if size(new_pop,1) < N
        last_front = fronts{i};
        crowding_dist = calculate_crowding_distance(combined_obj, {last_front});
        [~, sorted_idx] = sort(crowding_dist(last_front), 'descend');
        n_needed = N - size(new_pop,1);
        new_pop = [new_pop; combined_pop(last_front(sorted_idx(1:n_needed)),:)];
        new_obj = [new_obj; combined_obj(last_front(sorted_idx(1:n_needed)),:)];
    end
end