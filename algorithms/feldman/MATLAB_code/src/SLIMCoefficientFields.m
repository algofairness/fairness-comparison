classdef SLIMCoefficientFields
    %Helper class to store, access, and change information used for a coefficient
    %constraint in a SLIM model. SLIMCoefficientFields stores information for one coefficient.
    %
    %The following fields are stored and created for a single coefficient:
    %
    %variable_name:     variable_name of the coefficient
    %ub:                upperbound on coefficient set, 10 by default
    %lb:                lowerbound on coefficient set, -10 by default
    %
    %type:      can either be 'integer' or 'custom', 'integer' by default
    %           if 'integer' then coefficient will take on integer values in [lb,ub]
    %           if 'custom' then coefficient will take any value specified in 'values'
    %
    %values:    array containing discrete values that a 'custom' coefficient can
    %           take.
    %
    %sign:      sign of the coefficient set, =NaN by default
    %           -1 means coefficient will be positive <=0
    %            1 means coefficient will be negative >=0
    %           0 means no sign constraint
    %
    %           setting sign overrides conflicts with lb/ub/values field,
    %           so if a user sets sign to 1, and [lb,ub] = [-10,10] then
    %           [lb,ub] will change to [0,10]
    %
    %C_0j:      custom feature selection parameter,
    %           must be either NaN or a value in [0,1]
    %           C_0j represents the minimum % accuracy that feature j must add
    %           in order to be included in the SLIM classifier (i.e. have a non-zero
    %           coefficient)
    %
    %Author:      Berk Ustun | ustunb@mit.edu | www.berkustun.com
    %Reference:   SLIM for Optimized Medical Scoring Systems, http://arxiv.org/abs/1502.04269
    %Repository:  <a href="matlab: web('https://github.com/ustunb/slim_for_matlab')">slim_for_matlab</a>
    
    properties
        variable_name;
        ub = 10;
        lb = -10;
        values = [];
        C_0j = NaN;
    end
    
    properties(Dependent)
        sign
        type
    end
    
    
    methods
        
        %create new coefConstraints object
        function obj = SLIMCoefficientFields(varargin)
            if nargin == 1
                obj.variable_name = varargin{1};
            end
            obj.checkRep()
        end
        
        %% Getter Functions
        function variable_name = get.variable_name(obj)
            variable_name = obj.variable_name;
        end
        
        function ub = get.ub(obj)
            switch obj.type
                case 'integer'
                    ub = obj.ub;
                case 'custom'
                    ub = max(obj.values);
            end
        end
        
        function lb = get.lb(obj)
            switch obj.type
                case 'integer'
                    lb = obj.lb;
                case 'custom'
                    lb = min(obj.values);
            end
        end
        
        function type = get.type(obj)
            if isempty(obj.values)
                type = 'integer';
            else
                type = 'custom';
            end
        end
        
        function sign = get.sign(obj)
            if (obj.ub > 0) && (obj.lb>=0)
                sign = 1;
            elseif (obj.ub <=0) && (obj.lb<0)
                sign = -1;
            else
                sign = 0;
            end
            
        end
        
        function C_0j = get.C_0j(obj)
            C_0j = obj.C_0j;
        end
        
        function values = get.values(obj)
            values = obj.values;
        end
        
        %% Setter Functions
        function obj = set.variable_name(obj, new_name)
            assert(~isempty(new_name), 'name must be set to a non-empty string')
            assert(ischar(new_name), 'name must be a char array')
            obj.variable_name = new_name;
        end
        
        function obj = set.ub(obj, new_ub)
            switch obj.type
                case 'integer'
                    obj.ub = new_ub;
                case 'custom'
                    current_values = obj.values;
                    new_values = current_values(current_values<=new_ub);
                    obj.values = new_values;
            end
            obj.checkRep()
        end
        
        function obj = set.lb(obj, new_lb)
            switch obj.type
                case 'integer'
                    obj.lb = new_lb;
                case 'custom'
                    current_values = obj.values;
                    new_values = current_values(current_values>=new_lb);
                    obj.values = new_values;
            end
            obj.checkRep()
        end
        
        function obj = set.type(obj, new_type)
            error('type is determined automatically; users cannot set type')
        end
        
        function obj = set.sign(obj, new_sign)
            
            assert(length(new_sign)==1, 'sign must be -1,0,-1');
            assert(isnumeric(new_sign), 'sign must be -1,0,-1');
            assert(any(new_sign==[-1,0,1]), 'sign must be -1,0,-1');
            
            %obj.sign = new_sign;
            
            current_ub = obj.ub;
            current_lb = obj.lb;
            
            if new_sign == 1
                
                assert(current_ub > 0, 'if setting sign to +1 then make sure that ub > 0, otherwise coefficient will always be 0')
                if (current_lb < 0)
                    warning_msg = sprintf('setting lb to 0 for %s since sign = +1', obj.variable_name);
                    obj.print_warning(warning_msg);
                    obj.lb = 0;
                end
                
            elseif new_sign == -1
                
                assert(current_lb < 0, 'if setting sign to +1 then make sure that ub > 0, otherwise coefficient will always be 0')
                if (current_lb < 0)
                    warning_msg = sprintf('setting ub to 0 for %s since sign = -1', obj.variable_name);
                    obj.print_warning(warning_msg);
                    obj.ub = 0;
                end
                
            end
            
            obj.checkRep()
        end
        
        function obj = set.C_0j(obj, new_C_0j)
            if ~isnan(new_C_0j)
                assert(new_C_0j >=0, 'C_0j must be a value in [0,1)')
                assert(new_C_0j <=1, 'C_0j must be a value in [0,1)')
            end
            obj.C_0j = new_C_0j;
            obj.checkRep()
        end
        
        function obj = set.values(obj, new_values)
            
            assert(isnumeric(new_values), 'values must be a numeric array filled with unique values')
            
            %remove any NaN entries
            new_values = new_values(~isnan(new_values));
            
            %remove duplicate entries
            if ~isequal(new_values, unique(new_values,'stable'))
                obj.print_warning('values vector contains duplicate elements; will only use unique elements of values')
                new_values = unique(new_values);
            end
            
            %sort
            new_values = sort(new_values);
            new_values = new_values(:)';
            
            if all(new_values~=0)
                warning_msg = sprintf('custom values for %s do not include 0\n%s', ...
                    obj.variable_name, 'you should set C_0j to 0 to prevent L0-regularization for this variable');
                obj.print_warning(warning_msg);
            end
            
            new_lb = min(new_values);
            new_ub = max(new_values);
            is_integer_set = isequal(floor(new_values), new_lb:new_ub);
            
            if is_integer_set
                warning_msg = sprintf('custom values for %s are consecutive integers from %d to %d\nwill store to integers to improve performance',...
                    obj.variable_name, new_lb, new_lb, obj.variable_name);
                obj.print_warning(warning_msg);
                obj.values = [];
                obj.ub = new_ub;
                obj.lb = new_lb;
                
            else
                obj.values = new_values;
            end
            
            obj.checkRep();
            
        end
        
        %% Warnings and CheckRep
        
        function print_warning(obj, msg)
            warning('SLIM:ConstraintWarning', msg);
        end
        
        function checkRep(obj)
            
            %check types
            assert(ischar(obj.variable_name), 'variable_name field should be a char');
            assert(ischar(obj.type), 'type field should be a char');
            assert(isnumeric(obj.values), 'values field should be a array');
            assert(isnumeric(obj.ub), 'ub field should be numeric');
            assert(isnumeric(obj.lb), 'lb field should be numeric');
            assert(isnumeric(obj.C_0j), 'C_0j field should be numeric');
            assert(isnumeric(obj.sign), 'sign field should be numeric');
            
            %check entries
            assert(~isempty(obj.variable_name), 'variable_name should be non-empty')
            assert(strcmp('integer',obj.type)| strcmp('custom',obj.type),'type must be *integer* or *custom*');
            assert(obj.C_0j<=1 | obj.C_0j>=0 | isnan(obj.C_0j), 'C_0j entry should be between 0 and 1 or NaN');
            assert(obj.sign==0 | obj.sign==1 | obj.sign==-1, 'sign must either be -1,0,1');
            
            %check ub/lb
            assert(obj.ub >= obj.lb, 'ub should be >= lb');
            if obj.sign==1
                assert(obj.lb>=0, 'lb must be >= 0 for any coefficient with sign = +1');
            elseif obj.sign==-1
                assert(obj.ub<=0, 'ub must be <= 0 for any coefficient with sign = -1');
            end
            
            if (obj.ub>0 && obj.lb>0) || (obj.ub<0 && obj.lb<0)
                warning_msg = sprintf('valid coefficient values for variable %s do not include = 0\n%s',...
                    obj.variable_name,'consider C_0j to 0 to prevent L0-regularization for this variable');
                obj.print_warning(warning_msg);
            end
            
            if strcmp('custom', obj.type);
                assert(~(isempty(obj.values)|any(isnan(obj.values))));
                assert(min(obj.values)==obj.lb);
                assert(max(obj.values)==obj.ub);
            end
        end
        
        
    end
    
end
