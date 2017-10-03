%This file can be used to install SLIM / check that it is working properly
%
%Author:      Berk Ustun | ustunb@mit.edu | www.berkustun.com
%Reference:   SLIM for Optimized Medical Scoring Systems, http://arxiv.org/abs/1502.04269
%Repository:  <a href="matlab: web('https://github.com/ustunb/slim_for_matlab')">slim_for_matlab</a>

%%

%run from the slim_for_matlab directory
slim_repo_dir = pwd;

%Add SLIM to the working directory
code_dir = [slim_repo_dir,'/src/'];
test_dir = [slim_repo_dir,'/test/'];
addpath([slim_repo_dir,'/src/'])

%Check that Key Files are Present
key_repository_files = ...
    {'/data/breastcancer_processed_dataset.mat', ...
    '/demos/slim_ex_1_quickstart.m',...
    '/demos/slim_ex_2_customizing_coefficient_sets.m',...
    '/demos/slim_ex_3_limiting_model_size.m',...
    '/demos/slim_ex_4_TPR_FPR_constraints.m',...
    '/check_installation.m', ...
    '/src/createSLIM.m', ...
    '/src/getSLIMSummary.m', ...
    '/src/SLIMCoefficientConstraints.m', ...
    '/src/SLIMCoefficientFields.m', ...
    '/src/printScoringSystem.m',...
    };

for file = key_repository_files
    assert(exist([slim_repo_dir,file{1}], 'file')==2, sprintf('file: %s cannot be found', file{1}))
end

%% Run Tests
cd(test_dir)

test_files = dir('*.m');
test_files = {test_files(:).name};
for test = test_files
    run(test{1})
end

%% Make Sure CPLEX Works

%Make Sure that CPLEX works
try
    LP = Cplex();
    fprintf('Found CPLEX version %s\n', LP.getVersion());
catch
    error('Could not find CPLEX; make sure that CPLEX is installed and in the MATLAB search path')
end
