% SLIM Example #2
%
%This example shows how to customize coefficient constraints for a SLIM scoring system
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

%% Create a CoefficientConstraints object

%Coefficient values can be specified in a SLIMCoefficientConstraints object
%
%SLIMCoefficientConstraints objects are composed of the following fields
%
% coefConstraints.name(j)       name of feature j; must be non-empty
%
% coefConstraints.ub(j)         upperbound on the value of the coefficient for feature j
%
% coefConstraints.lb(j)         lowerbound on the value of the coefficient for feature j
%
% coefConstraints.type(j)       type of coefficent; can either be:
%                               'integer': coefficient for feature j is an integer between [coefConstraints.lb(j), coefConstraints.ub(j)]
%                               'custom': coefficient for feature j is a value specified by coefConstraints.values(j)
%
% coefConstraints.values(j)     array containing all feasible values of the coefficient for feature j the set of values for coefficient j;
%                               values should include 0; if not coefficient j cannot be dropped
%
% coefConstraints.sign(j)       constraint on the sign of the coefficient
%                               coefSet(j).sign = 1 means coefficient j >= 0
%                               coefSet(j).sign =-1 means coefficient j <= 0
%                               coefSet(j).sign = NaN means no constraint
%
% coefConstraints.C_0j(j)       customized sparsity for C_0j; if C_0j = NaN, then a global C_0 parameter is used
%

%coefficient constraints can be created using by specifying the number of coefficients
coefCons = SLIMCoefficientConstraints(length(X_names));

%or by using a cell array with variable names
coefCons = SLIMCoefficientConstraints(X_names);

%users can see all information regarding coefficient constraints on the console
coefCons

%field values can be accessed directly:
variable_name = coefCons.variable_name
ub = coefCons.ub
lb = coefCons.lb
sign = coefCons.sign
type = coefCons.type
values = coefCons.values
C_0j = coefCons.C_0j

%or by using the getfield function
coefCons.getfield('variable_name')
coefCons.getfield('ub')
coefCons.getfield('lb')
coefCons.getfield('sign')
coefCons.getfield('type')
coefCons.getfield('values')
coefCons.getfield('C_0j')

%all fields can be changed directly
coefCons.ub = 5;     %set upperbound for all coefficients to 5
coefCons.lb = -5;    %set lowerbound for all coefficients to -5

%or by using the setfield function
coefCons = coefCons.setfield('ub', 5);
coefCons = coefCons.setfield('lb', -5);

%setfield and getfield work for individual features
coefCons = coefCons.setfield('(Intercept)', 'ub', 10);
coefCons = coefCons.setfield('(Intercept)', 'lb', -10);
assert(coefCons.getfield('(Intercept)', 'ub') == 10)
assert(coefCons.getfield('(Intercept)', 'lb') == -10)

%setfield and getfield work as well as subsets of features
coefCons = coefCons.setfield({'NormalNucleoli', 'Mitoses'}, 'ub', 6);
assert(isequal(coefCons.getfield({'NormalNucleoli', 'Mitoses'}, 'ub'), 6*ones(2,1)));

%% Sign Constraints
%we can restrict a coefficient to only positive/negative values by setting its sign
%this enforces positive/negative associations between a feature and its outcome
%note that changing sign will also change the ub/lb of a variable

%set sign for 'BareNuclei' to positive and 'SingleEpithelialCellSize' to negative
coefCons = coefCons.setfield('BareNuclei', 'sign', 1);
assert(coefCons.getfield('BareNuclei', 'lb')==0)

coefCons = coefCons.setfield('SingleEpithelialCellSize', 'sign', -1);
assert(coefCons.getfield('SingleEpithelialCellSize', 'ub')==0)


%% Custom Coefficient Values

%users can restrict coefficients to a 'custom' set of values
coefCons = coefCons.setfield('NormalNucleoli', 'values', [-7, -5,-2,-1, 0, 1, 2, 5, 7]);

%variables with a customized set of coefficients are labeled with the type 'custom'
assert(strcmp(coefCons.getfield('NormalNucleoli', 'type'),'custom'));
assert(coefCons.getfield('NormalNucleoli', 'ub')==7);
assert(coefCons.getfield('NormalNucleoli', 'lb')==-7);

%setting sign/ub/lb for 'custom' coefficients directly changes the set of values

%for instance, we can set ub = 6 for 'NormalNucleoi' to drop any values > 6
coefCons = coefCons.setfield('NormalNucleoli', 'ub', 6);

%note that the new ub for 'NormalNuclei' is 5 since max(coefSet.getfield('NormalNucleoli', 'values')) == 5
assert(coefCons.getfield('NormalNucleoli', 'ub')==5);

%similarly, we can set the sign = 1 to drop any values < 0
coefCons = coefCons.setfield('NormalNucleoli', 'sign', 1);
assert(coefCons.getfield('NormalNucleoli', 'lb')==0);

%by default, SLIM is easier to solve without customized coefficient sets
%so if a 'custom' coefficient is a set of consecutive integers then SLIM will
%automatically reset the the type to integer.
coefCons = coefCons.setfield('ClumpThickness', 'values', -6:6);
assert(strcmp(coefCons.getfield('ClumpThickness', 'type'),'integer'));
assert(isempty(coefCons.getfield('ClumpThickness', 'values')));

%% Sparsity (L0-Regularization Penalty)
%users can set sparsity penalties for each feature using the C_0j field

%say we set the sparsity penalty for 'ClumpThickness' to 0.05
coefCons = coefCons.setfield('ClumpThickness', 'C_0j', 0.05);
%this means that SLIM will only use 'ClumpThickness' if it yields a 5% gain in training accuracy

%remember -- always set the sparsity penalty for the intercept to 0 manually
coefCons = coefCons.setfield('(Intercept)', 'C_0j', 0.00);


%% Train SLIM with Customized Coefficient Set

%Setup SLIM input struct
input.coefConstraints               = coefCons;
input.X                             = X;       %X should include a column of 1s to act as an intercept
input.Y                             = Y;
input.X_names                       = X_names; %the intercept should have the name '(Intercept)'
input.Y_name                        = Y_name;
input.w_pos                         = 1.00;
input.w_neg                         = 1.00;
input.C_0                           = 0.01;

[slim_IP, slim_info] = createSLIM(input);

%set CPLEX solver parameters
slim_IP.Param.emphasis.mip.Cur               = 1;   %mip solver strategy
slim_IP.Param.timelimit.Cur                  = 60;  %timelimit in seconds
slim_IP.Param.randomseed.Cur                 = 0;
slim_IP.Param.threads.Cur                    = 1;   %# of threads; >1 starts a parallel solver
slim_IP.Param.output.clonelog.Cur            = 0;   %disable CPLEX's clone log

%slim_IP.DisplayFunc = [] %uncomment to prevent on screen CPLEX display

%solve SLIM
slim_IP.solve

%get summary
summary = getSLIMSummary(slim_IP, slim_info);







