function [error]=polyfit_match(union)
year=union(:,1);
x0=year';
drange=union(:,2);
drange_matrix_row=drange';

y0=smooth(drange_matrix_row,15,"loess");

p=polyfit(x0',y0,4);
y=polyval(p,x0);
disp(vpa(poly2sym(p)),10);

error=sum(sqrt((y0-y').^2))/length(y0);

plot(x0,drange_matrix_row,'o',x0,y0,x0,y0,'*');
end
