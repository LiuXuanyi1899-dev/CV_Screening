function [data]=readdata(filename,index)
[~,~,data]=xlsread(filename,1,index);
end