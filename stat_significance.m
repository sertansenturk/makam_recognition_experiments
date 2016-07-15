%%
clear
close all
clc

%% paths
addpath(genpath('./dlfm_code'))

%% data
exp_names = {'tonic', 'mode', 'joint'};

training = {'single', 'multi'};
distribution = {'pd', 'pcd'};
bin_size = [7.5, 15.0, 25.0, 50.0, 100.0];
kernel_width = [0, 7.5, 15.0, 25.0, 50.0, 100.0];
dists = {'l1', 'l2', 'l3', 'bhat', 'dis_intersect', 'dis_corr'};
k_neighbors = [1, 3, 5, 10, 15];
peak_ratio = [0.15];

test_dir = './data/testing/';

%% data
parsed_res_file = fullfile(test_dir, 'results_fold.mat');
if ~exist(parsed_res_file, 'file')
    %% parse experiments
    exp_template = struct('accuracy', [], 'training', [], 'distribution', [], ...
        'distance', [], 'bin_size', [], 'kernel_width', [], 'k_neighbors', []);
    
    exps = struct('tonic', exp_template, 'mode', exp_template,...
        'joint', exp_template);
    
    for ee = 1:numel(exp_names)
        for tt = 1:numel(training)
            for dd = 1:numel(distribution)
                for bb = 1:numel(bin_size)
                    for kk = 1:numel(kernel_width)
                        for di = 1:numel(dists)
                            for kn = 1:numel(k_neighbors)
                                bin_str = num2str(bin_size(bb), '%.1f');
                                if strcmp(bin_str, '0.0')
                                    bin_str = '0';
                                end
                                kernel_str = num2str(kernel_width(kk), '%.1f');
                                if strcmp(kernel_str, '0.0')
                                    kernel_str = '0';
                                end
                                
                                param_str = [training{tt} '--' ...
                                    distribution{dd} '--'...
                                    strrep(bin_str, '.', '_') '--'...
                                    strrep(kernel_str, '.', '_') '--'...
                                    dists{di} '--'...
                                    num2str(k_neighbors(kn)) '--' ...
                                    strrep(num2str(peak_ratio(1)), '.', '_')];
                                
                                for ff = 0:9  % folds
                                    fold_dir = fullfile(test_dir, ...
                                        exp_names{ee}, param_str, ...
                                        ['fold' num2str(ff)]);
                                    eval_file = fullfile(fold_dir, ...
                                        'evaluation.json');
                                    if ~exist(eval_file, 'file')
                                        if exist(fold_dir, 'dir')
                                            error(['Extra fold ' fold_dir])
                                        end
                                        if strcmp(training{tt}, 'single') && ...
                                                k_neighbors(kn) ~= 1
                                            
                                        elseif kernel_width(kk) == 0 || ...
                                                bin_size(bb) > kernel_width(kk) * 3
                                            
                                        else
                                            error(['Missing exp: ' fold_dir])
                                        end
                                    else
                                        if ff == 0
                                            disp(eval_file)
                                        end
                                        
                                        fold = external.jsonlab.loadjson(...
                                            eval_file);
                                        
                                        % add accuracy
                                        if strcmp(exp_names{ee}, 'tonic')
                                            exps.(exp_names{ee}).accuracy = ...
                                                [exps.(exp_names{ee}).accuracy; ...
                                                fold.overall.tonic_accuracy];
                                        elseif strcmp(exp_names{ee}, 'mode')
                                            exps.(exp_names{ee}).accuracy = ...
                                                [exps.(exp_names{ee}).accuracy; ...
                                                fold.overall.mode_accuracy];
                                        elseif strcmp(exp_names{ee}, 'joint')
                                            exps.(exp_names{ee}).accuracy = ...
                                                [exps.(exp_names{ee}).accuracy; ...
                                                fold.overall.joint_accuracy];
                                        end
                                        
                                        % add parameters
                                        exps.(exp_names{ee}).training = ...
                                            [exps.(exp_names{ee}).training; ...
                                            training(tt)];
                                        exps.(exp_names{ee}).distribution = ...
                                            [exps.(exp_names{ee}).distribution; ...
                                            distribution(dd)];
                                        exps.(exp_names{ee}).bin_size = ...
                                            [exps.(exp_names{ee}).bin_size; ...
                                            bin_size(bb)];
                                        exps.(exp_names{ee}).kernel_width = [...
                                            exps.(exp_names{ee}).kernel_width; ...
                                            kernel_width(kk)];
                                        exps.(exp_names{ee}).distance = ...
                                            [exps.(exp_names{ee}).distance; ...
                                            dists(di)];
                                        exps.(exp_names{ee}).k_neighbors = ...
                                            [exps.(exp_names{ee}).k_neighbors; ...
                                            k_neighbors(kn)];
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    save(parsed_res_file, 'exps')
else
    results_fold = load(parsed_res_file);
    exps = results_fold.exps;
    clear results_fold
end

%% Statistical significance
%%% Here the commented lines are the starting variables. In the next step
%%% only the best variables without statistical significance among
%%% themselves are kept
exp_name = 'joint';
params = struct();
% params.training = {'single', 'multi'};
params.training = {'multi'};

% params.distribution = {'pd', 'pcd'};
params.distribution = {'pcd'};

params.distance = {'l1', 'l2', 'l3', 'bhat', 'dis_intersect', 'dis_corr'};
% params.distance = {'bhat'};

% params.bin_size = [7.5, 15.0, 25.0, 50.0, 100.0];
params.bin_size = [7.5, 15.0, 25.0];

% params.kernel_width = [0, 7.5, 15.0, 25.0];
params.kernel_width = [7.5, 15.0];

% params.k_neighbors = [1, 3, 5, 10, 15];
params.k_neighbors = [5, 10, 15];

cons_param_names = fieldnames(params);
bool = true(size(exps.(exp_name).accuracy));
for ff = 1:numel(cons_param_names)
    bool = bool & ismember(exps.(exp_name).(cons_param_names{ff}), ...
        params.(cons_param_names{ff}));
end

% get the values and groups
vals = exps.(exp_name).accuracy(bool);
group_names = {'training', 'distribution', 'bin_size', 'kernel_width',...
    'distance', 'k_neighbors'};
g = cell(1, numel(group_names));
for ff = 1:numel(group_names)
    g{ff} = exps.(exp_name).(group_names{ff})(bool);
end

% anovan
chk_groups = {'k_neighbors'};
chk_bool = ismember(group_names, chk_groups);
[p, ~, stats] = anovan(vals, g(chk_bool), 'model', 'interaction', 'varnames', group_names(chk_bool));
figure
results = multcompare(stats, 'Dimension', [1:length(g(chk_bool))], 'alpha', 0.01);