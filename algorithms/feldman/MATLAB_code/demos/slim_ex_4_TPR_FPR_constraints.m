% SLIM Example #4
%
%This example shows how to train a scoring system that obeys hard limits on the true positive rate / false positive rate
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

%%  Train SLIM Scoring System with Min TPR of 99%

%Setup usual input struct
input                               = struct();
input.display_warnings              = true;
input.X                             = X;
input.X_names                       = X_names;
input.Y                             = Y;
input.Y_name                        = Y_name;

coefCons = SLIMCoefficientConstraints(X_names);
coefCons = coefCons.setfield('(Intercept)', 'C_0j', 0);
input.coefCons = coefCons;

%Add constraint to limit the maximum error on positive examples to 1%
%Since positive error = 1 - TPR, so this will limit TPR to at least 99%
input.pos_err_max = 0.01;

%Set w_neg large enough to guarantee that SLIM hits the TPR constraint
N     = size(X,1);
N_pos = sum(Y==1);
N_neg = N - N_pos;

input.w_neg = 2*N_pos/(1+N_pos);
input.w_pos = 2-input.w_neg;


%Set C_0j small enough to make sure that we do not sacrifice accuracy
L0_regularized_variables = coefCons.C_0j~=0.0;
input.C_0 = 0.9*min(input.w_pos/N,input.w_neg/N)/ min(sum(L0_regularized_variables));

%create SLIM IP
[slim_IP, slim_info] = createSLIM(input);

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

summary = getSLIMSummary(slim_IP, slim_info)

%check that scoring system obeys TPR constraint
assert(summary.true_positive_rate>=1-input.pos_err_max)

%%  Train SLIM Scoring System with Max FPR of 25%

%Setup usual input struct
input                               = struct();
input.display_warnings              = true;
input.X                             = X;
input.X_names                       = X_names;
input.Y                             = Y;
input.Y_name                        = Y_name;

coefCons = SLIMCoefficientConstraints(X_names);
coefCons = coefCons.setfield('(Intercept)', 'C_0j', 0);
input.coefCons = coefCons;

%To get SLIM scoring system with maximum TPR such that max FPR <= 0.25

%1. Add a constraint on the negative error rate since:
%
%FPR = -ve examples predicted as +ve / # of -ve examples
%    = neg_error_rate
%
input.neg_err_max = 0.25;

%2. Set w_pos large enough to guarantee that SLIM hits the FPR constraint
N     = size(X,1);
N_pos = sum(Y==1);
N_neg = N - N_pos;

input.w_pos = 2*N_neg/(1+N_neg);
input.w_neg = 2-input.w_pos;

%Set C_0j small enough to make sure that we do not sacrifice accuracy
L0_regularized_variables = coefCons.C_0j~=0.0;
input.C_0 = 0.9*min(input.w_pos/N,input.w_neg/N)/ min(sum(L0_regularized_variables));

%create SLIM IP
[slim_IP, slim_info] = createSLIM(input);

%set default CPLEX solver parameters
slim_IP.Param.emphasis.mip.Cur               = 1;   %mip solver strategy
slim_IP.Param.timelimit.Cur                  = 60;  %timelimit in seconds
slim_IP.Param.randomseed.Cur                 = 0;
slim_IP.Param.threads.Cur                    = 1;   %# of threads; >1 starts a parallel solver
slim_IP.Param.output.clonelog.Cur            = 0;   %disable CPLEX's clone log
slim_IP.Param.mip.tolerances.lowercutoff.Cur = 0;
%slim_IP.Param.mip.tolerances.mipgap.Cur      = eps; %use maximal precision for IP solver (only recommended for testing)
%slim_IP.Param.mip.tolerances.absmipgap.Cur   = eps; %use maximal precision for IP solver (only recommended for testing)
%slim_IP.Param.mip.tolerances.integrality.Cur = eps; %use maximal precision for IP solver (only recommended for testing)

%slim_IP.DisplayFunc = [] %uncomment to prevent on screen CPLEX display

%solve the SLIM IP
slim_IP.solve

%get summary statistics
summary = getSLIMSummary(slim_IP, slim_info)

%check that scoring system obeys FPR constraint
assert(summary.false_positive_rate<=input.neg_err_max)
