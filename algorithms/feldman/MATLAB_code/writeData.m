function [ output_args ] = writeData( filename, summary,accuracy,confusion_mat,preds )
%WRITEDATA Summary of this function goes here
%   Detailed explanation goes here
FileId =  fopen(filename,'w');
fprintf(FileId,strcat('Model:',summary.model_string));
fprintf(FileId,'\nAccuracy: ');
fprintf(FileId,num2str(accuracy));
fprintf(FileId,strcat('\nConfusion Matrix: \n', num2str(confusion_mat(1,1)),'\t', ...
                 num2str(confusion_mat(1,2)),'\n', num2str(confusion_mat(2,1)), ... 
                 '\t',num2str(confusion_mat(2,2)),'\n'));
fprintf(FileId,'Predictions: \n');
fprintf(FileId,'%d\n',preds);
fclose(FileId);

end

