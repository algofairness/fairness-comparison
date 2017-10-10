function model_string = printScoringSystem(coefs, X_names, Y_name)
%Prints a scoring system from coefficients. 
%
%Author:      Berk Ustun 
%Contact:     ustunb@mit.edu
%Reference:   SLIM for Optimized Medical Scoring Systems, http://arxiv.org/abs/1502.04269
%Repository:  <a href="matlab: web('https://github.com/ustunb/slim_for_matlab')">slim_for_matlab</a>

%sanity checks
assert(length(X_names)==length(coefs),'length of X_names and coefficients should match')
switch class(Y_name)
    case 'cell'
        assert(length(Y_name)==1,'Y_name must be a 1x1 cell containing a string')
        assert(ischar(Y_name{1}), 'Y_name must be a 1x1 cell containing a string')
        Y_name = Y_name{1};
    case 'char'
        %do nothing
    otherwise
        error('Y_name must be a string or a 1x1 cell containing a string')
end

intercept_matches = cellfun(@(j) ~isempty(regexpi(j, '.*intercept.*')), X_names, 'UniformOutput', true);

if sum(intercept_matches) == 1
    
    intercept_ind   = find(intercept_matches);
    intercept_name  = X_names(intercept_ind);
    intercept_value = coefs(intercept_ind);
    X_names(intercept_ind) = [];
    coefs(intercept_ind) = [];
    
elseif sum(intercept_matches) > 1
    
    intercept_ind   = find(intercept_matches, 1, 'First');
    intercept_name  = X_names(intercept_ind);
    intercept_value = coefs(intercept_ind);
    
    warning('found multiple matches for intercept; picking "%s" as intercept variable', intercept_name);
    
    X_names(intercept_ind) = [];
    coefs(intercept_ind) = [];
    
elseif sum(intercept_matches) == 0
    
    intercept_ind   = NaN;
    intercept_name  = '';
    intercept_value = 0;
    warning('intercept is missing, or cannot be identified from variable names\n make sure that intercept variable name matches intercept')
    
end

title_string    = sprintf('\npredict "%s = +1" if score > %d\n', Y_name, -intercept_value);

%reorder from large to small
to_drop = coefs == 0;
coefs(to_drop)   = [];
X_names(to_drop) = [];
[coefs, sort_ind] = sort(coefs,'descend');
X_names = X_names(sort_ind);

score_function_string = sprintf('score =\t');
for n = 1:length(coefs)
    coef_name   = X_names{n};
    coef_value  = coefs(n);
    if n == 1
        if coef_value == 1 %remove + sign from start
            score_row = sprintf('%s\n', coef_name);
        elseif coef_value == -1
            score_row = sprintf('- %s\n', coef_name);
        else
            score_row = sprintf('%s * %s\n', num2str(coef_value), coef_name);
        end
    elseif n==length(coefs) %no new line at end
        if coef_value == 1
            score_row = sprintf('\t\t\t+ %s', coef_name);
        elseif coef_value == -1
            score_row = sprintf('\t\t\t- %s', coef_name);
        else
            score_row = sprintf('\t\t\t%s * %s', num2str(coef_value), coef_name);
        end
    else %standard case
        if coef_value == 1
            score_row = sprintf('\t\t\t+ %s\n', coef_name);
        elseif coef_value == -1
            score_row = sprintf('\t\t\t- %s\n', coef_name);
        else
            score_row = sprintf('\t\t\t%s * %s\n', num2str(coef_value), coef_name);
        end
    end
    
    score_function_string = sprintf('%s%s',score_function_string, score_row);
    
end

model_string = sprintf('%s%s',title_string, score_function_string);

end

