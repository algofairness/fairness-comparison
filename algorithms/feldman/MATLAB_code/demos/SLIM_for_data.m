% SLIM Example #1
%
%This example shows how to create a basic SLIM scoring system
%
%Author:      Berk Ustun | ustunb@mit.edu | www.berkustun.com
%Reference:   SLIM for Optimized Medical Scoring Systems, http://arxiv.org/abs/1502.04269
%Repository:  <a href="matlab: web('https://github.com/ustunb/slim_for_matlab')">slim_for_matlab</a>

%% Load Dataset and Setup Warnings

demo_dir = [pwd,'\'];
repo_dir = [pwd,'\'];
cd(demo_dir);

code_dir = [repo_dir,'src\'];
data_dir = [repo_dir,'data\'];
addpath(code_dir);
X = csvread([data_dir,'\arrests_data\arrests_X_train.csv']);
Y = csvread([data_dir,'\TF_retrain\train_preds.csv']);
Y(find(Y==0)) = -1;

Y_name = 'Class';
X_names = {'x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13', ...
    'x14','x15','x16','x17','x18','x19','x20','x21','x22','x23','x24','x25','x26', ...
    'x27','x28','x29','x30','x31','x32','x33','x34','x35','x36','x37','x38','x39', ...
    'x40','x41','x42','x43','x44','(Intercept)'};

% X_names = {'x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13', ...
%     'x14','x15','x16','x17','x18','x19','x20','x21','x22','x23','x24','x25','x26', ...
%     'x27','x28','x29','x30','x31','x32','x33','x34','x35','x36','x37','x38','x39', ...
%     'x40','x41','x42','x43','x44','x45','x46','x47','x48','x49','x50','x51','x52', ...
%     'x53','x54','x55','x56','x57','x58','x59','x60','x61','x62','(Intercept)'};

% columns_used = [2,3,4,5];
% rows_used = [1:1000,3000:4000];
% train = csvread([data_dir, '\train_synth.csv']);
% X = horzcat(train(rows_used,columns_used),repmat(1,size(rows_used,2),1));
% Y = train(rows_used,6);
% Y(find(Y==0)) = -1;
% headers = textread([data_dir,'\headers_synth.csv'], '%s', 'whitespace',',');
% X_names = [headers(columns_used);'(Intercept)'];
% Y_name = headers(6);

warning on SLIM:Coefficients    %'on' shows warnings about SLIM Coefficient Set
warning on SLIM:CreateSLIM      %'on' shows warnings about SLIM IP Creation

%% Setup SLIM input


input.X                             = X;       %X should include a column of 1s to act as an intercept
input.Y                             = Y;
input.X_names                       = X_names; %the intercept should have the name '(Intercept)'
input.Y_name                        = Y_name;

%set misclassification costs
%by default, if w_pos and w_neg are not provided, w_pos = w_neg = 1.00
%if w_pos and w_neg are provided, then SLIM will normalize values so that w_pos + w_neg = 2.00
input.w_pos                         = 1.00;
input.w_neg                         = 1.00;

%set sparsity penalty parameter: C_0
%C_0 should be set as the % gain in accuracy required for a feature to have a non-zero coefficient
input.C_0                          = 0.001;
input.C_1                          = 0.00001;

%coefficient constraints
%by default, each coefficient is an integer in [-10,10]
coefCons = SLIMCoefficientConstraints(X_names);  %by default each coefficient lambda_j is an integer from -10 to 10
coefCons = coefCons.setfield('(Intercept)', 'C_0j', 0); %the regularization penalty for the intercept should be set to 0 manually
input.coefConstraints = coefCons;

%simple operational constraints (see ex_3, ex_4 for examples)
%input.L0_min                        = 0;
%input.L0_max                        = P;
%input.err_min                       = 0;
%input.err_max                       = 1;
%input.pos_err_min                   = 0;
%input.pos_err_max                   = 1;
%input.neg_err_min                   = 0;
%input.neg_err_max                   = 1;

%createSLIM create a Cplex object, slim_IP and provides useful info in slim_info
[slim_IP, slim_info] = createSLIM(input);

%% Use CPLEX to train SLIM IP

%set CPLEX solver parameters
slim_IP.Param.timelimit.Cur                  = 30;  %timelimit in seconds
slim_IP.Param.threads.Cur                    = 2;   %# of threads; >1 starts a parallel solver
slim_IP.Param.output.clonelog.Cur            = 0;   %disable CPLEX's clone log
slim_IP.Param.mip.tolerances.lowercutoff.Cur = 0;
slim_IP.Param.mip.tolerances.mipgap.Cur      = eps; %use maximal precision for IP solver (only recommended for testing)
slim_IP.Param.mip.tolerances.absmipgap.Cur   = eps; %use maximal precision for IP solver (only recommended for testing)
slim_IP.Param.mip.tolerances.integrality.Cur = eps; %use maximal precision for IP solver (only recommended for testing)
slim_IP.Param.emphasis.mip.Cur               = 1;   %mip solver strategy
slim_IP.Param.randomseed.Cur                 = 0;

%slim_IP.DisplayFunc = [] %uncomment to prevent on screen CPLEX display

%solve the SLIM IP
% slim_IP.solve

%once you solve the SLIM IP, you can solve it again
slim_IP.Param.timelimit.Cur = 60*5;
slim_IP.solve

%get summary statistics
summary = getSLIMSummary(slim_IP, slim_info);

summary;
summary.model_string

% Print the retrain values
[accuracy,confusion_mat,~] = testSLIM(summary.coefficients,X,Y)
% Y values will remain the same for all repaired sets
% test_y = csvread('data\arrests_data\arrests_Y_test.csv');
% test_y(find(test_y)==0) = -1;
% repair_dir = 'data\TF_retrain\repaired\';
% output_dir = 'data\TF_retrain\output_data_test\';
% repair_files = dir(repair_dir);
% temp_accuracy = [];
% for i = 3:size(repair_files,1)
%     display(['Repair data: ' [repair_dir,repair_files(i).name]])
%     test_data = csvread([repair_dir,repair_files(i).name]);
%     [accuracy,confusion_mat,preds] = testSLIM(summary.coefficients,test_data,test_y);
%     filename = [output_dir,repair_files(i).name, '.data'];
%     writeData(filename,summary,accuracy,confusion_mat,preds);
%     
%     % For graphs
%     temp_accuracy = [temp_accuracy ; {filename, accuracy}];
% end
% 
% T = cell2table(temp_accuracy);
% writetable(T,[output_dir 'all_accuracys.dat']);
% 
% % Run the model on the training data to get the predictions
% [accuracy,confusion_mat,preds] = testSLIM(summary.coefficients,X,Y);
% writeData([output_dir,'train_output.csv.data'],summary,accuracy,confusion_mat,preds);


