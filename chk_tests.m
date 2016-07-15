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

%% parse experiments
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
                                results_file = fullfile(fold_dir, ...
                                    'results.json');
                                if ~exist(results_file, 'file')
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
                                        disp(results_file)
                                    end
                                    
                                    fold = external.jsonlab.loadjson(...
                                        results_file);
                                    
                                    
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

