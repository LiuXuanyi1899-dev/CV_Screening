function union=dataprocess(filename,index)
data=readdata(filename,index);
drange_cell_nan=data(1:84,1);     
drange_matrix_nan=cell2mat(drange_cell_nan); 
len=length(drange_matrix_nan);
is_null=isnan(drange_matrix_nan);
for i=2:len
    if is_null(i)==1 && is_null(i-1)==0
        drange_matrix_nan(i)=drange_matrix_nan(i-1);
    end
end
year=1999.25:0.25:2020.00;  
union=[year',drange_matrix_nan];
[row,~]=find(isnan(union));
union(row,:)=[];
drange=union(:,2);
len=length(drange);
subplot(2,1,1);
plot(drange,'o');

title('原始数据');



w=1+0.4*log(len);
error_d=abs(drange-mean(drange)) > w*std(drange);
drange(error_d)=[];
[row,~]=find(error_d()==1);
union(row,:)=[];
subplot(2,1,2);
plot(drange,'rs');
title('处理后');




end