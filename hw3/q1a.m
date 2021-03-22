clear;
stddev = 0.1;
x_test = -1:0.01:1;
x_train = -1:0.05:1;
n_train = size(x_train, 2);
y_test = 1.2*sin(pi*x_test) - cos(2.4*pi*x_test);
y_train = 1.2*sin(pi*x_train) - cos(2.4*pi*x_train) + 0.3*randn(1, n_train);

r_train = x_train' - x_train;
phi_train = exp(-r_train.^2/(2*stddev^2));
w = inv(phi_train) * y_train';

r = x_test' - x_train;
phi = exp(-r.^2/(2*stddev^2));
y_pred = (phi * w)';
loss = mse(y_pred, y_test);
fprintf('Loss(mse): %f\n', loss);

scatter(x_train, y_train, 'b*');
hold on, plot(x_test, y_test, 'g--', 'linewidth', 1.5);
hold on, plot(x_test, y_pred, 'r', 'linewidth', 1.5);
grid on;
h1 = xlabel('$x$'); h2 = ylabel('$y$');
h3 = legend('Train set', 'Ground truth', 'RBFN output', 'Location', 'northwest');
h4 = title('RBFN function approximation.');
set([h1 h2 h3 h4], 'Interpreter', 'latex');

