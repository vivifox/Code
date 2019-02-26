function y = sinweight(x,d)

y = zeros(d, d);
for i = 1:d
    for j = 1:d
            y(i,j) = sin(10*pi*x)+0.1*i+0.1*j;
            %y(i,j) = sin(2*pi*x)+i+j;
    end
end
end
