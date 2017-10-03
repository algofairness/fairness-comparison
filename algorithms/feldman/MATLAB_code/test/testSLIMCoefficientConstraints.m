function testSLIMCoefficientConstraints()
%Unit tests for SILMCoefficientConstraints class
%Need to be run from slim_for_matlab\test\
%
%Author:      Berk Ustun | ustunb@mit.edu | www.berkustun.com
%Reference:   SLIM for Optimized Medical Scoring Systems, http://arxiv.org/abs/1502.04269
%Repository:  <a href="matlab: web('https://github.com/ustunb/slim_for_matlab')">slim_for_matlab</a>

%% Setup Testing Environment
clc;
%dbstop if error %uncomment to stop at error

test_dir = [pwd,'/'];
cd('..');
repo_dir = [pwd,'/'];
cd(test_dir);

code_dir = [repo_dir,'src/'];
data_dir = [repo_dir,'data/'];
addpath(code_dir);
data = load([data_dir, 'breastcancer_processed_dataset.mat']);

warning off backtrace
warning off SLIM:ConstraintWarning
warning off SLIM:IPWarning

%% Run Tests

testConstructor()
testConstructorNoNames()
testSetNameScalar()
testSetNameVector()
testSetUBScalar()
testSetUBVector()
testSetLBScalar()
testSetLBVector()
testSetSignScalar()
testSetSignVector()
testSetC0jScalar()
testSetC0jVector()
testSetValuesScalar()
testSetValuesVector()

testSetNameByName()
testSetUBByName()
testSetLBByName()
testSetSignByName()
testSetC0jByName()
testSetValuesByName()

testGetUBByName()
testGetLBByName()
testGetC0jByName()
testGetSignByName()
testGetTypeByName()
testGetValuesByName()

testGetUBByName()
testGetLBByName()
testGetC0jByName()
testGetSignByName()
testGetTypeByName()
testGetValuesByName()


%% Unit Tests

    function testConstructor()
        
        c = SLIMCoefficientConstraints(data.X_names);
        n_variables = length(data.X_names);
        assert(numel(c)==n_variables);
        
        assert(isequal(c.variable_name(:), data.X_names(:)));
        assert(isequal(c.ub, repmat(10, n_variables, 1)));
        assert(isequal(c.lb, repmat(-10, n_variables, 1)));
        assert(isequal(c.sign, zeros(n_variables,1)));
        assert(all(isnan(c.C_0j)));
        assert(all(cellfun(@(x) isempty(x), c.values, 'UniformOutput', true)));
        assert(all(cellfun(@(x) strcmp('integer',x), c.type, 'UniformOutput', true)));
        
    end

    function testConstructorNoNames()
        
        c = SLIMCoefficientConstraints(length(data.X_names));
        n_variables = length(data.X_names);
        expectedNames = arrayfun(@(j) sprintf('X_%d',j), 1:n_variables, 'UniformOutput', false);
        
        assert(isequal(c.variable_name(:), expectedNames(:)));
        assert(isequal(c.ub, repmat(10, n_variables, 1)));
        assert(isequal(c.lb, repmat(-10, n_variables, 1)));
        assert(isequal(c.sign, zeros(n_variables,1)));
        assert(all(isnan(c.C_0j)));
        assert(all(cellfun(@(x) isempty(x), c.values, 'UniformOutput', true)));
        assert(all(cellfun(@(x) strcmp('integer',x), c.type, 'UniformOutput', true)));
        
    end

    function testSetNameScalar()
        
        c = SLIMCoefficientConstraints(length(data.X_names));
        c.variable_name = 'V';
        assert(all(cellfun(@(x) strcmp('V',x), c.variable_name, 'UniformOutput', true)));
        
    end

    function testSetNameVector()
        
        c = SLIMCoefficientConstraints(data.X_names);
        n_variables = length(data.X_names);
        newNames = arrayfun(@(j) sprintf('X_%d',j), 1:n_variables, 'UniformOutput', false);
        c.variable_name = newNames;
        assert(isequal(c.variable_name(:), newNames(:)));
        
    end

    function testSetUBScalar()
        c = SLIMCoefficientConstraints(length(data.X_names));
        n_variables = length(data.X_names);
        assert(isequal(c.ub, repmat(10, n_variables, 1)));
        c.ub = 5;
        assert(isequal(c.ub, repmat(5, n_variables, 1)));
    end

    function testSetUBVector()
        c = SLIMCoefficientConstraints(length(data.X_names));
        n_variables = length(data.X_names);
        assert(isequal(c.ub, repmat(10, n_variables, 1)));
        c.ub = repmat(5,n_variables);
        assert(isequal(c.ub, repmat(5, n_variables, 1)));
    end

    function testSetLBScalar()
        c = SLIMCoefficientConstraints(length(data.X_names));
        n_variables = length(data.X_names);
        assert(isequal(c.lb, repmat(-10, n_variables, 1)));
        c.lb = -5;
        assert(isequal(c.lb, repmat(-5, n_variables, 1)));
    end

    function testSetLBVector()
        c = SLIMCoefficientConstraints(length(data.X_names));
        n_variables = length(data.X_names);
        assert(isequal(c.lb, repmat(-10, n_variables, 1)));
        c.lb = repmat(-5,n_variables);
        assert(isequal(c.lb, repmat(-5, n_variables, 1)));
    end

    function testSetSignScalar()
        
        c = SLIMCoefficientConstraints(length(data.X_names));
        c.sign = 1;
        assert(all(c.lb==0))
        
        c = SLIMCoefficientConstraints(length(data.X_names));
        c.sign = -1;
        assert(all(c.ub==0))
        
        c = SLIMCoefficientConstraints(length(data.X_names));
        c.sign = 0;
        assert(all(c.ub==10))
        assert(all(c.lb==-10))
        
    end

    function testSetSignVector()
        
        c = SLIMCoefficientConstraints(length(data.X_names));
        n_variables = length(data.X_names);
        
        pos_ind = mod(1:n_variables,3)==1;
        no_ind = mod(1:n_variables,3)==2;
        neg_ind = mod(1:n_variables,3)==0;
        
        new_signs = pos_ind*1 + neg_ind*-1;
        c.sign = new_signs;
        
        assert(all(c.ub(no_ind)==10))
        assert(all(c.lb(no_ind)==-10))
        assert(all(c.sign(no_ind)==0))
        
        assert(all(c.ub(pos_ind)==10))
        assert(all(c.lb(pos_ind)==0))
        assert(all(c.sign(pos_ind)==1))
        
        assert(all(c.ub(neg_ind)==0))
        assert(all(c.lb(neg_ind)==-10))
        assert(all(c.sign(neg_ind)==-1))
        
    end

    function testSetC0jScalar()
        
        c = SLIMCoefficientConstraints(length(data.X_names));
        c.C_0j = 0.5;
        assert(all(c.C_0j==0.5));
        
    end

    function testSetC0jVector()
        
        c = SLIMCoefficientConstraints(length(data.X_names));
        n_variables = length(data.X_names);
        
        nan_ind = mod(1:n_variables,3)==1;
        zero_ind = mod(1:n_variables,3)==2;
        half_ind = mod(1:n_variables,3)==0;
        
        new_C_0j = nan(n_variables,1);
        new_C_0j(zero_ind) = 0;
        new_C_0j(half_ind) = 0.5;
        
        c.C_0j = new_C_0j;
        
        assert(all(isnan(c.C_0j(nan_ind))));
        assert(all(c.C_0j(zero_ind)==0));
        assert(all(c.C_0j(half_ind)==0.5));
        
    end


    function testSetValuesScalar()
        
        c = SLIMCoefficientConstraints(length(data.X_names));
        c.values = [1, 2, 5];
        assert(all(c.ub==5))
        assert(all(c.lb==1))
        assert(all(c.sign==1))
        assert(all(cellfun(@(x) isequal([1,2,5],x), c.values, 'UniformOutput', true)));
        
        c = SLIMCoefficientConstraints(length(data.X_names));
        c.values = -[1, 2, 5];
        assert(all(c.ub==-1))
        assert(all(c.lb==-5))
        assert(all(c.sign==-1))
        assert(all(cellfun(@(x) isequal(sort(-[1,2,5]),x), c.values, 'UniformOutput', true)));
        
        %with duplicates in any order and shape
        c = SLIMCoefficientConstraints(length(data.X_names));
        c.values = fliplr([1,1,2,1,1,1, 2, 5,5,5,5, 5])';
        assert(all(c.ub==5))
        assert(all(c.lb==1))
        assert(all(c.sign==1))
        assert(all(cellfun(@(x) isequal([1,2,5],x), c.values, 'UniformOutput', true)));
        
        c = SLIMCoefficientConstraints(length(data.X_names));
        c.values = fliplr(-[1,2,2,1,2,1,1,1, 2, 5,5,5,5, 5])';
        assert(all(c.ub==-1))
        assert(all(c.lb==-5))
        assert(all(c.sign==-1))
        assert(all(cellfun(@(x) isequal(sort(-[1,2,5]),x), c.values, 'UniformOutput', true)));
        
    end

    function testSetValuesVector()
        
        c = SLIMCoefficientConstraints(length(data.X_names));
        n_variables = length(data.X_names);
        
        pos_ind = mod(1:n_variables,3)==1;
        no_ind = mod(1:n_variables,3)==2;
        neg_ind = mod(1:n_variables,3)==0;
        
        new_values = repmat({[]}, n_variables, 1);
        new_values(no_ind) = {[-6:6,0,1,2,3,3,5]};
        new_values(pos_ind) = {[1,2,2,2,2,5]};
        new_values(neg_ind) = {-[1,1,1,1,2,2,2,2,2,2,2,5,5,5,5,5]};
        
        c.values = new_values;
        
        assert(all(strcmp(c.type(no_ind),'integer')));
        assert(all(c.ub(no_ind)==6))
        assert(all(c.lb(no_ind)==-6))
        assert(all(c.sign(no_ind)==0))
        assert(all(cellfun(@(x) isempty(x), c.values(no_ind), 'UniformOutput', true)));
        
        assert(all(strcmp(c.type(pos_ind),'custom')));
        assert(all(c.ub(pos_ind)==5))
        assert(all(c.lb(pos_ind)==1))
        assert(all(c.sign(pos_ind)==1))
        assert(all(cellfun(@(x) isequal(sort([1,2,5]),x), c.values(pos_ind), 'UniformOutput', true)));
        
        assert(all(strcmp(c.type(neg_ind),'custom')));
        assert(all(c.ub(neg_ind)==-1))
        assert(all(c.lb(neg_ind)==-5))
        assert(all(c.sign(neg_ind)==-1))
        assert(all(cellfun(@(x) isequal(sort(-[1,2,5]),x), c.values(neg_ind), 'UniformOutput', true)));
        
    end


    function testSetNameByName()
        
        c = SLIMCoefficientConstraints(length(data.X_names));
        n_variables = length(data.X_names);
        variable_names = arrayfun(@(j) sprintf('X_%d',j), 1:n_variables, 'UniformOutput', false);
        
        c = c.setfield({'X_1','X_5'},'variable_name', 'Y');
        assert(all(strcmp(c.variable_name{strcmp(variable_names,'X_1')},'Y')))
        assert(all(strcmp(c.variable_name{strcmp(variable_names,'X_5')},'Y')))
        
        
    end


    function testSetUBByName()
        
        c = SLIMCoefficientConstraints(length(data.X_names));
        n_variables = length(data.X_names);
        variable_names = arrayfun(@(j) sprintf('X_%d',j), 1:n_variables, 'UniformOutput', false);
        
        c = c.setfield({'X_1','X_5'},'ub', 20);
        assert(c.ub(strcmp(variable_names,'X_1'))==20)
        assert(c.ub(strcmp(variable_names,'X_5'))==20)
        
        
    end


    function testSetLBByName()
        
        c = SLIMCoefficientConstraints(length(data.X_names));
        n_variables = length(data.X_names);
        variable_names = arrayfun(@(j) sprintf('X_%d',j), 1:n_variables, 'UniformOutput', false);
        
        c = c.setfield({'X_1','X_5'},'lb', -20);
        assert(c.lb(strcmp(variable_names,'X_1'))==-20)
        assert(c.lb(strcmp(variable_names,'X_5'))==-20)
        
        
    end

    function testSetSignByName()
        
        c = SLIMCoefficientConstraints(length(data.X_names));
        n_variables = length(data.X_names);
        variable_names = arrayfun(@(j) sprintf('X_%d',j), 1:n_variables, 'UniformOutput', false);
        
        c = c.setfield({'X_1','X_5'},'sign', 0);
        assert(c.sign(strcmp(variable_names,'X_1'))==0)
        assert(c.ub(strcmp(variable_names,'X_1'))==10)
        assert(c.lb(strcmp(variable_names,'X_1'))==-10)
        
        assert(c.sign(strcmp(variable_names,'X_5'))==0)
        assert(c.ub(strcmp(variable_names,'X_5'))==10)
        assert(c.lb(strcmp(variable_names,'X_5'))==-10)
        
        c = c.setfield({'X_1','X_5'},'sign', 1);
        assert(c.sign(strcmp(variable_names,'X_1'))==1)
        assert(c.ub(strcmp(variable_names,'X_1'))==10)
        assert(c.lb(strcmp(variable_names,'X_1'))==0)
        
        assert(c.sign(strcmp(variable_names,'X_5'))==1)
        assert(c.ub(strcmp(variable_names,'X_5'))==10)
        assert(c.lb(strcmp(variable_names,'X_5'))==0)
        
        c = SLIMCoefficientConstraints(length(data.X_names));
        c = c.setfield({'X_1','X_5'},'sign', -1);
        assert(c.sign(strcmp(variable_names,'X_1'))==-1)
        assert(c.ub(strcmp(variable_names,'X_1'))==0)
        assert(c.lb(strcmp(variable_names,'X_1'))==-10)
        
        assert(c.sign(strcmp(variable_names,'X_5'))==-1)
        assert(c.ub(strcmp(variable_names,'X_5'))==0)
        assert(c.lb(strcmp(variable_names,'X_5'))==-10)
        
    end

    function testSetC0jByName()
        
        c = SLIMCoefficientConstraints(length(data.X_names));
        n_variables = length(data.X_names);
        variable_names = arrayfun(@(j) sprintf('X_%d',j), 1:n_variables, 'UniformOutput', false);
        
        c = c.setfield({'X_1','X_5'},'C_0j', 0.5);
        assert(c.C_0j(strcmp(variable_names,'X_1'))==0.5)
        assert(c.C_0j(strcmp(variable_names,'X_5'))==0.5)
        
    end


    function testSetValuesByName()
        
        c = SLIMCoefficientConstraints(length(data.X_names));
        n_variables = length(data.X_names);
        variable_names = arrayfun(@(j) sprintf('X_%d',j), 1:n_variables, 'UniformOutput', false);
        variable_ind = strcmp(variable_names,'X_2');
        
        c = c.setfield('X_2','values', [1, 2, 5]);
        assert(isequal(c.values(variable_ind), {[1,2,5]}))
        
    end


    function testGetUBByName()
        
        c = SLIMCoefficientConstraints(length(data.X_names));
        
        c = c.setfield({'X_1','X_5'},'ub', 0);
        
        test_variable_names = {'X_6', 'X_1'};
        test_ub = c.getfield(test_variable_names, 'ub');
        assert(test_ub(strcmp('X_1',test_variable_names))==0);
        assert(test_ub(strcmp('X_6',test_variable_names))==10);
        
        assert(isequal(c.getfield('ub'), c.ub));
    end

    function testGetLBByName()
        
        c = SLIMCoefficientConstraints(length(data.X_names));
        
        c = c.setfield({'X_1','X_5'},'lb', 0);
        
        test_variable_names = {'X_6', 'X_1'};
        test_lb = c.getfield(test_variable_names, 'lb');
        assert(test_lb(strcmp('X_1',test_variable_names))==0);
        assert(test_lb(strcmp('X_6',test_variable_names))==-10);
        
        assert(isequal(c.getfield('lb'), c.lb));
    end


    function testGetC0jByName()
        
        c = SLIMCoefficientConstraints(length(data.X_names));
        
        c = c.setfield({'X_1','X_5'},'C_0j', 0.5);
        
        test_variable_names = {'X_6', 'X_1'};
        testC_0j = c.getfield(test_variable_names, 'C_0j');
        assert(isnan(testC_0j(strcmp('X_6',test_variable_names))));
        assert(testC_0j(strcmp('X_1',test_variable_names))==0.5);
        
        assert(isequaln(c.getfield('C_0j'), c.C_0j));
        
    end

    function testGetTypeByName()
        
        c = SLIMCoefficientConstraints(length(data.X_names));
        
        c = c.setfield('X_1','values', [1, 2, 5]);
        
        test_variable_names = {'X_6', 'X_1'};
        test_type = c.getfield(test_variable_names, 'type');
        assert(strcmp(test_type(strcmp('X_1',test_variable_names)), 'custom'))
        assert(strcmp(test_type(strcmp('X_6',test_variable_names)), 'integer'))
        
        assert(isequal(c.getfield('type'), c.type));
        assert(isequal(c.getfield('values'), c.values));
        
    end

    function testGetSignByName()
        
        c = SLIMCoefficientConstraints(length(data.X_names));
        
        c = c.setfield({'X_1','X_5'},'sign', 1);
        
        test_variable_names = {'X_6', 'X_1'};
        test_lb = c.getfield(test_variable_names, 'lb');
        assert(test_lb(strcmp('X_1',test_variable_names))==0);
        assert(test_lb(strcmp('X_6',test_variable_names))==-10);
        
        test_ub = c.getfield(test_variable_names, 'ub');
        assert(test_ub(strcmp('X_1',test_variable_names))==10);
        assert(test_ub(strcmp('X_6',test_variable_names))==10);
        
        test_sign = c.getfield(test_variable_names, 'sign');
        assert(test_sign(strcmp('X_1',test_variable_names))==1);
        assert(test_sign(strcmp('X_6',test_variable_names))==0);
        
        assert(isequal(c.getfield('sign'), c.sign));
        assert(isequal(c.getfield('lb'), c.lb));
        assert(isequal(c.getfield('ub'), c.ub));
        
    end

    function testGetValuesByName()
        
        c = SLIMCoefficientConstraints(length(data.X_names));
        
        c = c.setfield('X_2','values', [1, 2, 5]);
        test_values = c.getfield({'X_2','X_1'}, 'values');
        assert(isequal(test_values(1), {[1,2,5]}));
        assert(isequal(test_values(2), {[]}));
        
    end


end

