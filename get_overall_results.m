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
eval_results = struct();
for ee = 1:numel(exp_names)
    it = 1;
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
                            
                            res_file = fullfile(test_dir, ...
                                exp_names{ee}, param_str, 'overall_eval.json');
                            if ~exist(res_file, 'file')
                                if strcmp(training{tt}, 'single') && ...
                                        k_neighbors(kn) ~= 1
                                    
                                elseif kernel_width(kk) == 0 || ...
                                        bin_size(bb) > kernel_width(kk) * 3
                                    
                                else
                                    error(['Missing exp: ' fold_dir])
                                end
                            else
                                disp([exp_names{ee} ': ' param_str])
                                
                                tmp_eval = external.jsonlab.loadjson(res_file);
                                % add accuracy
                                if strcmp(exp_names{ee}, 'tonic')
                                    eval_results.tonic(it) = struct('param', ...
                                        param_str, 'accuracy', ...
                                        tmp_eval.tonic_accuracy / 100);
                                elseif strcmp(exp_names{ee}, 'mode')
                                    eval_results.mode(it) = struct('param', ...
                                        param_str, 'accuracy', ...
                                        tmp_eval.mode_accuracy / 100);
                                elseif strcmp(exp_names{ee}, 'joint')
                                    eval_results.joint(it) = struct('param', ...
                                        param_str, 'accuracy', ...
                                        tmp_eval.joint_accuracy / 100,...
                                        'tonic_accuracy', ...
                                        tmp_eval.tonic_accuracy / 100,...
                                        'mode_accuracy', ...
                                        tmp_eval.mode_accuracy / 100);
                                end
                                it = it + 1;
                            end
                        end
                    end
                end
            end
        end
    end
end

%% sort results
[~, idx] = sort([eval_results.tonic.accuracy], 'descend');
eval_results.tonic = eval_results.tonic(idx);

[~, idx] = sort([eval_results.mode.accuracy], 'descend');
eval_results.mode = eval_results.mode(idx);

[~, idx] = sort([eval_results.joint.accuracy], 'descend');
eval_results.joint = eval_results.joint(idx);

%% save
[~] = external.jsonlab.savejson([], eval_results, ...
    fullfile(test_dir, 'evaluation_overall.json'));
