function [output] = zero_one_scaling_1D(data, min_val, max_val, min_flatten, max_flatten)

if nargin < 2
    max_val = max(data);
    min_val = min(data);
    min_flatten = 0;
    max_flatten = 0;

elseif nargin < 4
    min_flatten = 0;
    max_flatten = 0;
end

[n, m] = size(data);

if length(unique(data)) < 2
    output = ones(1,m);
else
    output = (data - repmat(min_val,1,m))./repmat((max_val - min_val),1,m) ; 
end

if min_flatten == 1
    below_min = (data < min_val);
    output(below_min) = 0;
end

if max_flatten == 1
    above_max = (data > max_val);
    output(above_max) = 1;
end
