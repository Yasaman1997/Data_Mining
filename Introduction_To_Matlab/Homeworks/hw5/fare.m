function cost = fare(miles,age)
    cost = 2;
    miles = round(miles);
    if miles <= 10
        cost = cost + max([0 (miles-1)]) * 0.25;
    else
        cost = cost + 9 * 0.25 + (miles-10) * 0.1;
    end
    if age <= 18 || age >= 60
        cost = cost * 0.8;
    end
end