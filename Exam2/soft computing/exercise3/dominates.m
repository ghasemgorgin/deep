function d = dominates(x,y)
    d = all(x <= y) && any(x < y);
end