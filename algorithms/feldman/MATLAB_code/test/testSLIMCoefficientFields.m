function testSLIMCoefficientFields()
%Unit tests for SLIMCoefficientFields class
%Need to be run from slim_for_matlab\test\
%
%Author:      Berk Ustun | ustunb@mit.edu | www.berkustun.com
%Reference:   SLIM for Optimized Medical Scoring Systems, http://arxiv.org/abs/1502.04269
%Repository:  <a href="matlab: web('https://github.com/ustunb/slim_for_matlab')">slim_for_matlab</a>

%% Setup Testing Environment
clc;

warning off SLIM:ConstraintWarning
warning off SLIM:IPWarning

%% Run Tests;

testConstructor()
testChangeSignToPositive()
testChangeSignToNegative()
testChangeSignToZero()
testSetUBToNegativeChangesSign()
testSetLBToPositiveChangesSign()
testSetValues()
testSetRepeatedValues()
testSetValuesAsConsecutiveIntegers()
testSetBoundsForCustomType()

%% Unit Tests

    function testConstructor()
        c = SLIMCoefficientFields('male');
        assert(strcmp(c.variable_name,'male'));
        assert(strcmp(c.type,'integer'));
        assert(c.ub == 10);
        assert(c.lb == -10);
        assert(c.sign == 0);
        assert(isnan(c.C_0j));
        assert(isempty(c.values));
    end

    function testChangeSignToPositive()
        c = SLIMCoefficientFields('male');
        c.sign = 1;
        assert(c.ub == 10);
        assert(c.lb == 0);
        assert(c.sign == 1);
    end

    function testChangeSignToNegative()
        c = SLIMCoefficientFields('male');
        c.sign = -1;
        assert(c.ub == 0);
        assert(c.lb == -10);
        assert(c.sign == -1);
    end

    function testChangeSignToZero()
        c = SLIMCoefficientFields('male');
        c.sign = 0;
        assert(c.ub == 10);
        assert(c.lb == -10);
        assert(c.sign == 0);
    end

    function testSetUBToNegativeChangesSign()
        c = SLIMCoefficientFields('male');
        c.ub = -10;
        assert(c.ub == -10);
        assert(c.sign == -1);
    end

    function testSetLBToPositiveChangesSign()
        c = SLIMCoefficientFields('male');
        c.lb = 1;
        assert(c.lb == 1);
        assert(c.sign == 1);
    end

    function testSetValues()
        c = SLIMCoefficientFields('male');
        c.values = [-5,-2,-1,0,1,2,5];
        assert(strcmp(c.type,'custom'))
        assert(c.lb==-5)
        assert(c.ub==5)
        assert(all(c.values==[-5,-2,-1,0,1,2,5]));
    end

    function testSetRepeatedValues()
        c = SLIMCoefficientFields('male');
        c.values = repmat([-5,-2,-1,0,1,2,5],1,2);
        assert(strcmp(c.type,'custom'))
        assert(c.lb==-5)
        assert(c.ub==5)
        assert(all(c.values==[-5,-2,-1,0,1,2,5]));
    end

    function testSetValuesAsConsecutiveIntegers()
        c = SLIMCoefficientFields('male');
        c.values = [-20:20];
        assert(isempty(c.values));
        assert(c.ub==20)
        assert(c.lb==-20)
        assert(strcmp(c.type,'integer'))
        assert(c.sign == 0);
    end

    function testSetBoundsForCustomType()
        
        c = SLIMCoefficientFields('male');
        c.values = [-7,-5,-2,-1,0,1,2,5,7];
        assert(strcmp(c.type,'custom'))
        
        c.ub = 3;
        assert(c.ub == 2);
        assert(c.lb == -7);
        assert(c.sign == 0);
        assert(isequal(c.values, [-7, -5,-2,-1,0,1,2]))
        assert(strcmp(c.type,'custom'))
        
        c.lb = -3;
        assert(c.ub == 2);
        assert(c.lb == -2);
        assert(c.sign == 0);
        assert(isequal(c.values, []))
        assert(strcmp(c.type,'integer'))
        
    end
   

end

