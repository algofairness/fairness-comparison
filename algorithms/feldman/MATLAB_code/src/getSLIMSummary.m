function results = getSLIMSummary(IP, info)
%Summarizes the output of a SLIM IP
%Output is a struct that contains the following information:
%
%coefficients        1 x P vector of scoring system coefficients 
%scores              N x 1 vector of scores on the training data
%predictions         N x 1 vector of predicted labels on the training data
%error_rate          error of the scoring system on the training data
%model_size          # of non-zero coefficeints with C_0j>0
%true_positive_rate  TPR of the scoring system on the training data
%false_positive_rate FPR of the scoring system on the training data
%runtime             runtime of the solver
%status              CPLEX status code
%objective_value     objective value of the SLIM IP
%upperbound          upperbound on the objective value of the SLIM IP (should be =objective value)
%lowerbound          lowerbound on the objective value of the SLIM IP (best objective value of the LP relaxation)
%integrality_gap     (upperbound-lowerbound)/upperbound;
%node_count          # of nodes processed by CPLEX
%simplex_iterations  # of simpled iterations by CPLEX
%solution_pool       struct with additional feasible scoring systems 
%model_string        print-out of the best scoring system found by CPLEX so far
%
%Author:      Berk Ustun 
%Contact:     ustunb@mit.edu
%Reference:   SLIM for Optimized Medical Scoring Systems, http://arxiv.org/abs/1502.04269
%Repository:  <a href="matlab: web('https://github.com/ustunb/slim_for_matlab')">slim_for_matlab</a>

results.coefficients        = NaN;
results.scores              = NaN;
results.predictions         = NaN;
results.error_rate          = NaN;
results.model_size          = NaN;
results.true_positive_rate  = NaN;
results.false_positive_rate = NaN;
results.runtime             = NaN;
results.status              = NaN;
results.objective_value     = NaN;
results.upperbound          = NaN;
results.lowerbound          = NaN;
results.integrality_gap     = NaN;
results.node_count          = NaN;
results.simplex_iterations  = NaN;
results.solution_pool       = struct('solution_pool',[],'objvals',[]);
results.model_string        = '';

%Get Model + Process Model Based Results
display('-------------------------')
Solution = IP.Solution
display('-------------------------')
if isfield(Solution,'x') %CPLEX may not have found a feasible solution yet
    
    coefs      = Solution.x(info.indices.lambdas);
    coefs      = coefs(:);
    
    %Identify Intercept (for model size calculations)
    intercept_matches = cellfun(@(j) ~isempty(regexpi(j, '.*intercept.*')), info.X_names, 'UniformOutput', true);
    if sum(intercept_matches) == 1
        intercept_coef = coefs(intercept_matches);
    elseif sum(intercept_matches) > 1
        intercept_ind   = find(intercept_matches, 1, 'First');
        intercept_coef  = coefs(intercept_ind);
    elseif sum(intercept_matches) == 0
        intercept_coef = 0;
        warning('intercept is missing, or cannot be identified from variable names\n make sure that intercept variable name matches intercept')
    end
    
    intercept_included = intercept_coef ~= 0;
    
    pos_ind = info.Y > 0;
    neg_ind = ~pos_ind;
    display('################################################')
    results.model_string        = printScoringSystem(coefs, info.X_names, info.Y_name)
    display('################################################')
    results.coefficients        = coefs(:)';
    results.scores              = info.X*coefs;
    results.predictions         = (1.*(results.scores>0)) + (-1.*(results.scores<=0));
    results.error_rate          = mean(info.Y ~= results.predictions);
    results.model_size          = sum(coefs~=0) - intercept_included;
    results.true_positive_rate  = mean(results.predictions(pos_ind)==1);
    results.false_positive_rate = mean(results.predictions(neg_ind)==1);
    
end


%Process Computational Results
try results.runtime         = Solution.time; end
try results.status          = sprintf('%s (%d)', Solution.statusstring, Solution.status); end
try results.objective_value = Solution.objval; end
try results.upperbound = Solution.cutoff; end
try results.lowerbound = Solution.bestobjval; end
try results.integrality_gap = Solution.miprelgap; end
try results.node_count = Solution.nodecnt; end
try results.simplex_iterations = Solution.mipitcnt; end
try
    solution_pool = Solution.pool.solution;
    pooled_solutions = {solution_pool.x};
    solution_pool.solutions = arrayfun(@(n) pooled_solutions(indices.lambdas)', 1:length(pooled_solutions), 'UniformOutput', false);
    solution_pool.objvals = [solution_pool.objval]';
    solution_pool = rmfield(solution_pool,'ax');
    solution_pool = rmfield(solution_pool,'objval');
    results.solution_pool = solution_pool;
end

end