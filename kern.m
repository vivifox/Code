function v = kern(x,h)
       v = 1/(sqrt(2*pi)*h)*exp(-(x/h)^2/2);
end