function [ accuracy,confusion_mat,preds ] = testSLIM( coefs, x, y )
%TESTSLIM Summary of this function goes here
%   Detailed explanation goes here
    % Calculate the scores
    scores = x*coefs';
    % Make the predictions
    preds = 1-(scores == -1);
    
    preds(find(preds==0)) = -1;
    y(find(y==0))= -1;
    
    % See how well it did
    error_rate = sum(preds ~= y)/size(y,1);
    accuracy = 1-error_rate;
    confusion_mat = [sum(bsxfun(@and,(preds==-1),(y==-1))), ...
                     sum(bsxfun(@and,(preds==-1),(y==1))); ...
                     sum(bsxfun(@and,(preds==1),(y==-1))), ...
                     sum(bsxfun(@and,(preds==1),(y==1)))];
                     

end

