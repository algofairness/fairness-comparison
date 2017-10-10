% SLIM Example #1
%
%This example shows how to create a basic SLIM scoring system
%
%Author:      Berk Ustun | ustunb@mit.edu | www.berkustun.com
%Reference:   SLIM for Optimized Medical Scoring Systems, http://arxiv.org/abs/1502.04269
%Repository:  <a href="matlab: web('https://github.com/ustunb/slim_for_matlab')">slim_for_matlab</a>

%% Load Breastcancer Dataset and Setup Warnings

demo_dir = [pwd,'\'];
repo_dir = [pwd,'\'];
cd(demo_dir);

code_dir = [repo_dir,'src\'];
data_dir = [repo_dir,'data\'];
addpath(code_dir);
load([data_dir, 'breastcancer_processed_dataset.mat']);

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
input.C_0                          = 0.01;

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
slim_IP.Param.threads.Cur                    = 1;   %# of threads; >1 starts a parallel solver
slim_IP.Param.output.clonelog.Cur            = 0;   %disable CPLEX's clone log
slim_IP.Param.mip.tolerances.lowercutoff.Cur = 0;
slim_IP.Param.mip.tolerances.mipgap.Cur      = eps; %use maximal precision for IP solver (only recommended for testing)
slim_IP.Param.mip.tolerances.absmipgap.Cur   = eps; %use maximal precision for IP solver (only recommended for testing)
slim_IP.Param.mip.tolerances.integrality.Cur = eps; %use maximal precision for IP solver (only recommended for testing)
slim_IP.Param.emphasis.mip.Cur               = 1;   %mip solver strategy
slim_IP.Param.randomseed.Cur                 = 0;

%slim_IP.DisplayFunc = [] %uncomment to prevent on screen CPLEX display

%solve the SLIM IP
slim_IP.solve

%once you solve the SLIM IP, you can solve it again
slim_IP.Param.timelimit.Cur = 20;
slim_IP.solve

%get summary statistics
summary = getSLIMSummary(slim_IP, slim_info);