T = readtable('2017_Data_1.csv')
TNew = T(:,2:end);
dataOriginal = table2array(TNew);

dataOriginal = 
data = dataOriginal'; % when data imported has columns as features and rows as samples

%% Expected data - rows are features, and columns are individual samples
[n,m] = size(data);
count = 1;
for i = 1:n
    node_data = data(i,:);
    %abs(max(node_data) - min(node_data))
    if abs(max(node_data) - min(node_data)) > 0
        preprocess_param{i}{1}(1) = min(node_data); 
        preprocess_param{i}{1}(2) = max(node_data); 
        preprocess_param{i}{2} = 'zero_one_scaling';    
        pp_data(count,:) = zero_one_scaling_1D(node_data, preprocess_param{i}{1}(1), preprocess_param{i}{1}(2)) ;  
        count = count+1;
    else
        preprocess_param{i} = 'Feature not considered';
    end
end

% intialize the network architecture
netconfig.inputsize = size(pp_data,1);
netconfig.layersizes{1} = 2;
% sparsity parameter
sparsityparam = 0.3;

% Main step of AE learning
stack = train_stacked_denoise_autoencoder(pp_data, netconfig, sparsityparam); %, sparsityParam, beta, maxIter, lambda); % Train a stacked sparse autoencoder, as per the above neural net structure to get good initial weights

%% Transforming higher dimensional data to 2D data
hlevel_data = stackedAE_Out(pp_data, stack);

% Plotting the 2D data
plot(hlevel_data(1,:), hlevel_data(2,:), 'ob')
hold on
ind = [100,101,102,103,345,346,347]; % indices of alarm occurences
plot(hlevel_data(1,ind), hlevel_data(2,ind), 'or')