% Matric No. A0225440R (mod(40, 1) + 1 = 2)
% Group 2 (Smile face detection)
clear;
groupID = 2;

%% Data input and format
img_train_dir = dir('Face Database/TrainImages/*.jpg');
img_test_dir = dir('Face Database/TestImages/*.jpg');
label_train_dir = dir('Face Database/TrainImages/*.att');
label_test_dir = dir('Face Database/TestImages/*.att');
n_train = size(label_train_dir, 1);
n_test = size(label_test_dir, 1);
img_train = zeros(10201, n_train);
img_test = zeros(10201, n_test);
label_train = zeros(1, n_train);
label_test = zeros(1, n_test);
img_resize_train = zeros(2601, n_train);
img_resize_test = zeros(2601, n_test);

for i = 1:n_train
    img_name = img_train_dir(i).name;
    label_name = label_train_dir(i).name;
    img = imread(['Face Database/TrainImages/', img_name]);
    img = rgb2gray(img);
    imgre = imresize(img, 0.5);
    img_v = img(:);
    imgre_v = imgre(:);
    img_train(:, i) = img_v(1:10201);
    img_resize_train(:, i) = imgre_v(1:2601);
    label = load(['Face Database/TrainImages/', label_name]);
    label_train(i) = label(groupID);
end
for i = 1:n_test
    img_name = img_test_dir(i).name;
    label_name = label_test_dir(i).name;
    img = imread(['Face Database/TestImages/', img_name]);
    img = rgb2gray(img);
    imgre = imresize(img, 0.5);
    img_v = img(:);
    imgre_v = imgre(:);
    img_test(:, i) = img_v(1:10201);
    img_resize_test(:, i) = imgre_v(1:2601);
    label = load(['Face Database/TestImages/', label_name]);
    label_test(i) = label(groupID);
end

%% Q1 Label distribution
n_smile_train = sum(label_train);
n_smile_test = sum(label_test);
label_matrix = [n_smile_train, n_train-n_smile_train; n_smile_test, n_test-n_smile_test];
x = ['Training', 'Testing'];
b = bar(label_matrix);
xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints;
labels1 = string(b(1).YData);
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')
xtips2 = b(2).XEndPoints;
ytips2 = b(2).YEndPoints;
labels2 = string(b(2).YData);
text(xtips2,ytips2,labels2,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')
legend('Smile','Non-smile');

%% Q2 Rosenblatt’s perceptron (single layer perceptron)
net = perceptron;
net.trainParam.epochs = 1;
max_epoch = 500;
accu_train = zeros(1, max_epoch);
accu_test = zeros(1, max_epoch);
for epoch = 1:max_epoch
    idx = randperm(n_train);
    net = train(net, img_train(:, idx), label_train(:, idx));
    label_pred0 = net(img_train);
    label_pred = net(img_test);
    label_pred0 = activate_fn(label_pred0);
    label_pred = activate_fn(label_pred);
    accu_train(epoch) = 1 - mean(abs(label_pred0 - label_train));
    accu_test(epoch) = 1 - mean(abs(label_pred - label_test));
    disp(epoch);
end
figure();
plot(accu_train, 'linewidth', 1.5);
hold on;
plot(accu_test, 'linewidth', 1.5);
xlabel('Epoch'), ylabel('Accuracy'), title('Single layer perceptron (original image)');
legend('Train Accuracy', 'Test Accuracy', 'location', 'southeast');
fprintf('Accuracy of single layer network (train): %.4f\n', accu_train(end));
fprintf('Accuracy of single layer network (test): %.4f\n', accu_test(end));

%% Q3 Resized images
net_resize = perceptron;
net_resize.trainParam.epochs = 1;
max_epoch = 500;
accu_resize_train = zeros(1, max_epoch);
accu_resize_test = zeros(1, max_epoch);
for epoch = 1:max_epoch
    idx = randperm(n_train);
    net_resize = train(net_resize, img_resize_train(:, idx), label_train(:, idx));
    label_pred0 = net_resize(img_resize_train);
    label_pred = net_resize(img_resize_test);
    label_pred0 = activate_fn(label_pred0);
    label_pred = activate_fn(label_pred);
    accu_resize_train(epoch) = 1 - mean(abs(label_pred0 - label_train));
    accu_resize_test(epoch) = 1 - mean(abs(label_pred - label_test));
    disp(epoch);
end
figure();
plot(accu_resize_train, 'linewidth', 1.5);
hold on;
plot(accu_resize_test, 'linewidth', 1.5);
xlabel('Epoch'), ylabel('Accuracy'), title('Single layer perceptron (resized image)');
legend('Train Accuracy', 'Test Accuracy', 'location', 'southeast');
fprintf('Accuracy of single layer network (train): %.4f\n', accu_resize_train(end));
fprintf('Accuracy of single layer network (test): %.4f\n', accu_resize_test(end));

%% Q4 MLP batch
n_hidden = [100];
max_epoch = 100;
net_mlp = patternnet(n_hidden);
net_mlp.trainParam.epochs = max_epoch;
net_mlp = train(net_mlp, img_train, label_train);
label_pred0 = net_mlp(img_train);
label_pred = net_mlp(img_test);
label_pred0 = activate_fn(label_pred0);
label_pred = activate_fn(label_pred);
perf = perform(net_mlp, label_pred, label_test);
accu_train = 1 - mean(abs(label_pred0 - label_train));
accu_test = 1 - mean(abs(label_pred - label_test));
fprintf('Loss(MSE): %.4f\n', perf);
fprintf('Accuracy of single layer network (train): %.4f\n', accu_train);
fprintf('Accuracy of single layer network (test): %.4f\n', accu_test);

%% Q5 MLP sequential
n_hidden = [200];
max_epoch = 500;
num_train = 900;
net = patternnet(n_hidden);
net.trainParam.epochs = 1;
accu_train = zeros(1, max_epoch);
accu_test = zeros(1, max_epoch);
accu_valid = zeros(1, max_epoch);
accmax = 0;

for epoch = 1:max_epoch
    idx = randperm(num_train);
    idx_val = randperm(n_train - num_train) + num_train;
    net = adapt(net, img_train(:, idx), label_train(:, idx));
    label_pred0 = net(img_train(:, idx));
    label_predv = net(img_train(:, idx_val));
    label_pred = net(img_test);
    label_pred0 = activate_fn(label_pred0);
    label_predv = activate_fn(label_predv);
    label_pred = activate_fn(label_pred);
    acc = 1 - mean(abs(label_predv - label_train(:, idx_val)));
    if acc > accmax
        accmax = acc;
        epoch0 = epoch;
    end
    accu_train(epoch) = 1 - mean(abs(label_pred0 - label_train(:, idx)));
    accu_test(epoch) = 1 - mean(abs(label_pred - label_test));
    accu_valid(epoch) = acc;
    disp(epoch);
end
figure();
plot(accu_train, 'linewidth', 1.5);
hold on;
plot(accu_valid, 'linewidth', 1.5);
hold on;
plot(accu_test, 'linewidth', 1.5);
hold on;
xlabel('Epoch'), ylabel('Accuracy'), title('Multi layer perceptron (sequential)');
legend('Train Accuracy', 'Validation Accuracy', 'Test Accuracy', 'location', 'southeast');
fprintf('Accuracy of single layer network (train): %.4f\n', accu_train(epoch0));
fprintf('Accuracy of single layer network (valid): %.4f\n', accu_valid(epoch0));
fprintf('Accuracy of single layer network (test): %.4f\n', accu_test(epoch0));
