classdef SLIMCoefficientConstraints
    %Helper class to store, access, and change information used for
    %coefficients in a SLIM model.
    %
    %Fields include:
    %
    %variable_name:     names of each feature, X_1...X_d by default
    %
    %ub:                upperbound on coefficient set, 10 by default
    %
    %lb:                lowerbound on coefficient set, -10 by default
    %
    %type:              type of coefficient, 'integer' by default
    %                   if 'integer' then coefficient will take on integer values in [lb,ub]
    %                   if 'custom' then coefficient will take any value specified in 'values'
    %
    %values:            values that can be taken by a 'custom' coefficient, empty by default
    %
    %sign:              sign of the coefficient set, =0  by default
    %                   -1 means coefficient is restricted to negative values
    %                   1 means coefficient is restricted to positive values
    %                   0 means coefficient can take on positive/negative values
    %
    %                   setting sign will override conflicts with lb/ub/values field,
    %                   so if [lb,ub] = [-10,10] and we set sign to 1, then [lb,ub] will change to [0,10]
    %
    %C_0j:              custom sparsity parameter, NaN by default
    %                   must be either NaN or a value in [0,1]
    %                   C_0j represents the minimum % accuracy that feature j must add
    %                   in order to be included in the SLIM classifier (i.e. have a non-zero
    %                   coefficient
    %
    %Author:      Berk Ustun | ustunb@mit.edu | www.berkustun.com
    %Reference:   SLIM for Optimized Medical Scoring Systems, http://arxiv.org/abs/1502.04269
    %Repository:  <a href="matlab: web('https://github.com/ustunb/slim_for_matlab')">slim_for_matlab</a>
    
    properties(Access=private)
        constraintArray;
        n_variables;
    end
    
    properties(Dependent)
        variable_name
        ub
        lb
        type
        values
        sign
        C_0j
    end
    
    methods
        
        %% Constructor to create new coefConstraints object
        function obj = SLIMCoefficientConstraints(varargin)
            
            if nargin > 0
                
                if nargin == 1 && isnumeric(varargin{1})
                    obj.n_variables = varargin{1};
                    variable_names = arrayfun(@(j) sprintf('X_%d',j), 1:obj.n_variables, 'UniformOutput', false);
                elseif nargin == 1 && iscell(varargin{1})
                    variable_names = varargin{1};
                    obj.n_variables = length(variable_names);
                end
                
                varargin
                tmpArray = cellfun(@(n) SLIMCoefficientFields(n), variable_names, 'UniformOutput', false);
                obj.constraintArray = [tmpArray{:}];
                obj.checkRep()
                
            end
            
        end
        
        %% Native Getter Methods
        
        function variable_name = get.variable_name(obj)
            variable_name = {obj.constraintArray(:).variable_name}';
        end
        
        function ub = get.ub(obj)
            ub = [obj.constraintArray(:).ub]';
        end
        
        function lb = get.lb(obj)
            lb = [obj.constraintArray(:).lb]';
        end
        
        function sign = get.sign(obj)
            sign = [obj.constraintArray(:).sign]';
        end
        
        function C_0j = get.C_0j(obj)
            C_0j = [obj.constraintArray(:).C_0j]';
        end
        
        function type = get.type(obj)
            type = {obj.constraintArray(:).type}';
        end
        
        function values = get.values(obj)
            values = {obj.constraintArray(:).values}';
        end
        
        %% Native Setter Methods
        
        function obj = set.variable_name(obj, new_name)
            
            if ischar(new_name)
                new_name = {new_name};
            end
            
            if length(new_name)==1
                new_name = repmat(new_name, obj.n_variables, 1);
            else
                assert(length(new_name)==length(obj.constraintArray))
            end
            
            for i = 1:length(obj.constraintArray)
                obj.constraintArray(i).variable_name = new_name{i};
            end
            obj.checkRep()
        end
        
        function obj = set.values(obj, new_values)
            
            if isnumeric(new_values)
                new_values = {new_values};
            end
            
            if length(new_values)==1
                new_values = repmat(new_values, obj.n_variables, 1);
            else
                assert(length(new_values)==length(obj.constraintArray),...
                    sprintf('new values must have length = 1 or %d', obj.n_variables))
            end
            
            for i = 1:length(obj.constraintArray)
                obj.constraintArray(i).values = new_values{i};
            end
            obj.checkRep()
        end
        
        function obj = set.type(obj, new_type)
            if length(new_type)==1
                new_type = repmat(new_type, obj.n_variables, 1);
            else
                assert(length(new_type)==length(obj.constraintArray),...
                    sprintf('new type must have length = 1 or %d', obj.n_variables))
            end
            
            for i = 1:length(obj.constraintArray)
                obj.constraintArray(i).type = new_type(i);
            end
            obj.checkRep()
        end
        
        function obj = set.ub(obj, new_ub)
            if length(new_ub)==1
                new_ub = repmat(new_ub, obj.n_variables, 1);
            else
                assert(length(new_ub)==length(obj.constraintArray),...
                    sprintf('new ub values must have length = 1 or %d', obj.n_variables))
            end
            
            for i = 1:length(obj.constraintArray)
                obj.constraintArray(i).ub = new_ub(i);
            end
            obj.checkRep()
        end
        
        function obj = set.lb(obj, new_lb)
            if length(new_lb)==1
                new_lb = repmat(new_lb, obj.n_variables, 1);
            else
                assert(length(new_lb)==length(obj.constraintArray))
            end
            
            for i = 1:length(obj.constraintArray)
                obj.constraintArray(i).lb = new_lb(i);
            end
            obj.checkRep()
        end
        
        function obj = set.sign(obj, new_sign)
            if length(new_sign)==1
                new_sign = repmat(new_sign, obj.n_variables, 1);
            else
                assert(length(new_sign)==length(obj.constraintArray))
            end
            
            for i = 1:length(obj.constraintArray)
                obj.constraintArray(i).sign = new_sign(i);
            end
            obj.checkRep()
        end
        
        
        function obj = set.C_0j(obj, new_C_0j)
            if length(new_C_0j)==1
                new_C_0j = repmat(new_C_0j, obj.n_variables, 1);
            else
                assert(length(new_C_0j)==length(obj.constraintArray))
            end
            
            for i = 1:length(obj.constraintArray)
                obj.constraintArray(i).C_0j = new_C_0j(i);
            end
            obj.checkRep()
        end
        
        %% Name-Based Getter/Setter Methods
        
        function field_values = getfield(obj, varargin)
            %Get field values for one or more variables
            %
            %variable_names names of the variables for which the field_name will be set to field_value
            %               can either be a single char or a cell array of strings
            %               all variable names must match the variable_name field of the SLIMCoefficientConstraint object
            %
            %field_name  any field name of a SLIMCoefficientConstraints object
            
            if nargin == 2
                variable_names = obj.variable_name;
                field_name = varargin{1};
            elseif nargin == 3
                variable_names = varargin{1};
                field_name = varargin{2};
            else
                error('invalid number of inputs')
            end
            
            assert(sum(strcmp(field_name,{'variable_name','lb','ub','type','values','sign','C_0j'}))==1, ....
                'field name must be: variable_name, lb, ub, values, type, sign, C_0j');
            
            if ischar(variable_names)
                variable_names = {variable_names};
            end
            
            current_names = obj.variable_name;
            variable_names = unique(variable_names, 'stable');
            
            valid_name_ind = ismember(variable_names, current_names, 'legacy');
            valid_names = variable_names(valid_name_ind);
            
            if any(~valid_name_ind)
                invalid_names = variable_names(~valid_name_ind);
                error_msg = sprintf('could not find variable matching name: %s\n', invalid_names{:});
                error_msg = error_msg(1:end-1); %remove last \n
                error(error_msg);
            end
            
            switch field_name
                case {'variable_name', 'type', 'values'}
                    field_values = cell(length(valid_names),1);
                    counter = 1;
                    for i = 1:length(valid_names)
                        ind = find(strcmp(current_names, valid_names{i}));
                        for j = 1:length(ind)
                            field_values(counter) = {obj.constraintArray(ind(j)).(field_name)};
                            counter = counter + 1;
                        end
                    end
                    
                    if length(field_values)==1
                        field_values = field_values{1};
                    end
                    
                otherwise
                    field_values = nan(length(valid_names),1);
                    counter = 1;
                    for i = 1:length(valid_names)
                        ind = find(strcmp(current_names, valid_names{i}));
                        for j = 1:length(ind)
                            field_values(counter) = obj.constraintArray(ind(j)).(field_name);
                            counter = counter + 1;
                        end
                    end
            end
            
            obj.checkRep()
            
        end
        
        function obj = setfield(obj, varargin)
            %Set the field value for one or more variables
            %
            %variable_names names of the variables for which the field_name will be set to field_value
            %               can either be a single char or a cell array of strings
            %               all variable names must match the variable_name field of the SLIMCoefficientConstraint object
            %
            %
            %field_name  limited to be 'variable_name','lb','ub','values','sign','C_0j'
            %
            %field_value limited to valid values of the fields above
            
            if nargin == 3
                variable_names = obj.variable_name;
                field_name = varargin{1};
                field_value = varargin{2};
            elseif nargin == 4
                variable_names = varargin{1};
                field_name = varargin{2};
                field_value = varargin{3};
            else
                error('invalid number of inputs')
            end
            
            assert(sum(strcmp(field_name,{'variable_name','lb','ub','values','sign','C_0j'}))==1, ....
                'field name must be: variable_name, lb,ub, values, sign, C_0j');
            
            if ischar(variable_names)
                variable_names = {variable_names};
            end
            
            current_names = obj.variable_name;
            variable_names = unique(variable_names, 'stable');
            valid_name_ind = ismember(variable_names, current_names, 'legacy');
            valid_names = variable_names(valid_name_ind);
            
            if any(~valid_name_ind)
                invalid_names = variable_names(~valid_name_ind);
                error_msg = sprintf('could not find variable matching name: %s\n', invalid_names{:});
                error_msg = error_msg(1:end-1); %remove last \n
                error(error_msg);
            end
            
            for i = 1:length(valid_names)
                ind = find(strcmp(current_names, valid_names{i}));
                for j = 1:length(ind)
                    obj.constraintArray(ind(j)).(field_name) = field_value;
                end
            end
            
            obj.checkRep()
            
        end
        
        %% Display, Warnings and CheckRep
        
        function print_warning(obj, msg)
            warning('SLIM:ConstraintWarning', msg);
        end
        
        
        function disp(obj)
            headers = {'variable_name','type','lb','ub','sign','values','C_0j'};
            info = [...
                obj.variable_name, ...
                obj.type, ...
                num2cell(obj.lb), ...
                num2cell(obj.ub), ...
                num2cell(int8(obj.sign)), ...
                obj.values, ...
                num2cell(obj.C_0j)...
                ];
            
            coefTable = [headers;info];
            disp(coefTable)
        end
        
        function n = numel(obj)
            n =  obj.n_variables;
        end
        
        
        function checkRep(obj)
            
            assert(obj.n_variables>0, 'need at least 1 variable');
            
            %check sizes
            assert(length(obj.variable_name)==obj.n_variables, sprintf('variable_name field should have exactly %d entries', obj.n_variables));
            assert(length(obj.ub)==obj.n_variables, sprintf('ub field should have exactly %d entries', obj.n_variables));
            assert(length(obj.lb)==obj.n_variables, sprintf('ub field should have exactly %d entries', obj.n_variables));
            assert(length(obj.C_0j)==obj.n_variables, sprintf('C_0j field should have exactly %d entries', obj.n_variables));
            assert(length(obj.sign)==obj.n_variables, sprintf('sign field should have exactly %d entries', obj.n_variables));
            assert(length(obj.type)==obj.n_variables, sprintf('type field should have exactly %d entries', obj.n_variables));
            assert(length(obj.values)==obj.n_variables, sprintf('values field should have exactly %d entries', obj.n_variables));
            
            %check types
            assert(iscell(obj.variable_name), 'variable_name field should be a cell');
            assert(iscell(obj.type), 'type field should be a cell');
            assert(iscell(obj.values), 'values field should be a cell');
            assert(isnumeric(obj.ub), 'ub field should be a array');
            assert(isnumeric(obj.lb), 'lb field should be a array');
            assert(isnumeric(obj.C_0j), 'C_0j field should be a array');
            assert(isnumeric(obj.sign), 'sign field should be a array');
            
            %check entries
            assert(all(cellfun(@(x) ~isempty(x), obj.variable_name)), 'each variable_names entry should be non-empty');
            assert(all(strcmp('integer',obj.type)| strcmp('custom',obj.type)),'type must be *integer* or *custom*');
            assert(all(obj.C_0j<=1 | obj.C_0j>=0 | isnan(obj.C_0j)), 'each C_0j entry should be between 0 and 1 or NaN');
            assert(all(obj.sign==0 | obj.sign==1 | obj.sign==-1), 'sign must either be -1,0,1');
            
            %check ub/lb
            assert(all(obj.ub >= obj.lb), 'ub should be > lb for each coefficient');
            pos_sign = obj.sign==1;
            neg_sign = obj.sign==-1;
            %no_sign == obj.sign==0;
            assert(all(obj.ub(neg_sign)<=0), 'ub <= for any coefficient s.t. sign == - 1');
            assert(all(obj.lb(pos_sign)>=0), 'lb <= for any coefficient s.t. sign == - 1');
            
            custom_ind = strcmp('custom', obj.type);
            if any(custom_ind)
                assert(all(cellfun(@(x) ~isempty(x), obj.values(custom_ind), 'UniformOutput', true)), 'if coefficient uses custom set of values, then values field must be non-empty and non-NaN');
            end
            
        end
        
    end
    
end

%% Helper Functions

%apply a function to each entry of a cell/numeric array
%returns numeric array containing function output
function values = apply(f, entries)

assert(isa(f,'function_handle'), 'first input needs to be a function handle')
assert(iscell(entries)||isnumeric(entries), 'second input needs to be cell array or numeric array')
n = length(entries);
if iscell(entries)
    values = cellfun(f, entries, 'UniformOutput', false);
elseif isnumeric(entries)
    values = arrayfun(f, entries, 'UniformOutput', true);
end

end
