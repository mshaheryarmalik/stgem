function [output] = mo3d(input)
    x1 = input{1};
    x2 = input{2};
    x3 = input{3};
    % disp({x1, x2, x3});
    h1 = 305 - 100 * (sin(x1 / 3) + sin(x2 / 3) + sin(x3 / 3));
    h2 = 230 - 75 * (cos(x1 / 2.5 + 15) + cos(x2 / 2.5 + 15) + cos(x3 / 2.5 + 15));
    h3 = (x1 - 7) ^ 2 + (x2 - 7) ^ 2 + (x3 - 7) ^ 2 - (cos((x1 - 7) / 2.75) + cos((x2 - 7) / 2.75) + cos((x3 - 7) / 2.75));
    output = {h1, h2, h3 - 20};
end
