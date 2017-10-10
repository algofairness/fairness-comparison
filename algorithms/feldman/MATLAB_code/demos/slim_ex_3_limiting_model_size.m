% SLIM Example #3
%
%This example shows how to train a SLIM scoring system with a hard limit on the number of features
%
%Author:      Berk Ustun | ustunb@mit.edu | www.berkustun.com
%Reference:   SLIM for Optimized Medical Scoring Systems, http://arxiv.org/abs/1502.04269
%Repository:  <a href="matlab: web('https://github.com/ustunb/slim_for_matlab')">slim_for_matlab</a>

%% Load Breastcancer Dataset and Setup Warnings

demo_dir = [pwd,'/'];
cd('..');
repo_dir = [pwd,'/'];
cd(demo_dir);

code_dir = [repo_dir,'src/'];
data_dir = [repo_dir,'data/'];
addpath(code_dir);
load([data_dir, 'breastcancer_processed_dataset.mat']);

warning on SLIM:Coefficients    %'on' shows warnings about SLIM Coefficient Set
warning on SLIM:CreateSLIM      %'on' shows warnings about SLIM IP Creation

%% Add a Constraint on the Min/Max # of Features

%Setup SLIM input struct
input                               = struct();
input.X                             = X;
input.X_names                       = X_names;
input.Y                             = Y;
input.Y_name                        = Y_name;
input.w_pos                         = 1.00;
input.w_neg                         = 1.00;

coefCons = SLIMCoefficientConstraints(X_names);
coefCons = coefCons.setfield('(Intercept)', 'C_0j', 0);
input.coefCons = coefCons;


%add a L0 constraints to limit classifiers to 1 to 3 features
%note: L0 constraints only apply to features such that C_0j > 0
%thus, the feature limit does not apply to '(Intercept)'
input.L0_min = 1; %minimum # of non-zero coefficients
input.L0_max = 3; %maximum # of non-zero coefficients


%set the sparsity penalty to a small enough value to obtain the most
%accurate scoring system that obeys the feature-limit constraints
%note: see paper for derivation
N = size(input.X, 1);
L0_regularized_variables = coefCons.C_0j~=0.0;

w_total = input.w_pos + input.w_neg;
w_pos   = 2.00*(input.w_pos/w_total);
w_neg   = 2.00*(input.w_neg/w_total);

input.C_0 = 0.9*min(w_pos/N,w_neg/N) / min(input.L0_max, sum(L0_regularized_variables));

%create SLIM
[slim_IP, slim_info] = createSLIM(input);

%% Use CPLEX to train SLIM with Feature Constraints

%set default CPLEX solver parameters
slim_IP.Param.emphasis.mip.Cur               = 1;   %mip solver strategy
slim_IP.Param.timelimit.Cur                  = 60;  %timelimit in seconds
slim_IP.Param.randomseed.Cur                 = 0;
slim_IP.Param.threads.Cur                    = 1;   %# of threads; >1 starts a parallel solver
slim_IP.Param.output.clonelog.Cur            = 0;   %disable CPLEX's clone log
%slim_IP.Param.mip.tolerances.lowercutoff.Cur = 0;
%slim_IP.Param.mip.tolerances.mipgap.Cur      = eps; %use maximal precision for IP solver (only recommended for testing)
%slim_IP.Param.mip.tolerances.absmipgap.Cur   = eps; %use maximal precision for IP solver (only recommended for testing)
%slim_IP.Param.mip.tolerances.integrality.Cur = eps; %use maximal precision for IP solver (only recommended for testing)

%slim_IP.DisplayFunc = [] %uncomment to prevent on screen CPLEX display

%solve the SLIM IP
slim_IP.solve

%get summary statistics
summary = getSLIMSummary(slim_IP, slim_info)

%check that scoring system obeys model size constraints
assert(sum(summary.coefficients(L0_regularized_variables)~=0)>=input.L0_min)
assert(sum(summary.coefficients(L0_regularized_variables)~=0)<=input.L0_max)

