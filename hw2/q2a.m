n_hidden = [10];
x       = -3:0.01:3;
x_train = -2:0.05:2;
x_test  = -3:0.01:3;
y       = 1.2*sin(pi*x)-cos(2.4*pi*x);
y_train = 1.2*sin(pi*x_train)-cos(2.4*pi*x_train);
xc = num2cell(x_train);
yc = num2cell(y_train);
net = feedforwardnet(n_hidden);
% net.trainParam.lr=0.001;
for i=1:2
    net = adapt(net, xc, yc);
    y_test = net(x_test);
    perf = perform(net, y_test, y);
    if perf < 3
        disp(i)
        break
    end
end
y_test = net(x_test);
perf = perform(net, y_test, y);
fprintf('Perf: %.4f\n', perf);

h1 = plot(x, y, 'g--');
hold on;
h2 = scatter(x_train, y_train, 'b*');
hold on;
h3 = plot(x_test, y_test, 'r', 'linewidth', 1.5);
xlabel('x'), ylabel('y');
hold off;
line([2,2], [-3,3], 'linestyle', '--'), line([-2,-2], [-3,3], 'linestyle', '--');
legend([h1, h2, h3], {'Actual Fn', 'Train Set', 'NN Output'});
title('Number of hidden neuron:', n_hidden);
