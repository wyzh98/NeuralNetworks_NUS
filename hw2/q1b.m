max_iter = 1e4;
max_err = 1e-3;
rollout = [];
x = rand;
y = rand;

for iter = 1:max_iter
    z = (1 - x)^2 + 100*(y - x^2)^2;
    dzdx = 2*x - 400*x*(- x^2 + y) - 2;
    dzdy = - 200*x^2 + 200*y;
    Hinv = [1/(2*(200*x^2 - 200*y + 1)), x/(200*x^2 - 200*y + 1);
        x/(200*x^2 - 200*y + 1), (600*x^2 - 200*y + 1)/(200*(200*x^2 - 200*y + 1))];
    d = Hinv * [dzdx; dzdy];
    x = x - d(1);
    y = y - d(2);
    rollout(iter, :) = [x y z];
    if z-0 < max_err
        fprintf('Converge to minima (%.4f, %.4f) at iteration %d!\n', x, y, iter);
        break
    end
end

plot(rollout(:, 1), rollout(:, 2), 'linewidth', 2);
hold on; scatter(rollout(1, 1), rollout(1, 2), 'filled');
hold on; scatter(rollout(iter, 1), rollout(iter, 2), 'filled');
xlabel('x'), ylabel('y');

x0 = -0.5:0.01:1.5;
y0 = -0.5:0.01:1.5;
[xx, yy] = meshgrid(x0, y0);
zz = (1 - xx).^2 + 100*(yy - xx.^2).^2;
hold on, grid on;
contour(xx, yy, zz, 20);

figure();
plot(rollout(:, 3), 'linewidth', 2);
xlabel('Iteration'), ylabel('z');
