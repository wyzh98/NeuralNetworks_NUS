% Matric No. A0225440R
% Choose class 0(N) and 4(A)
clear;
lambda = 0;
n_center = 2;
stddev = 5000;

%% Data init
load('characters10.mat');
trainIdx = find(train_label == 0 | train_label == 4); % select classes 0, 4
trainLabel = train_label(trainIdx);
trainData = train_data(trainIdx, :);
trainLabel(trainLabel == 4) = 1; % class 4 labeled as 1
testIdx = find(test_label == 0 | test_label == 4); % select classes 0, 4
testLabel = test_label(testIdx);
testData = test_data(testIdx, :);
testLabel(testLabel == 4) = 1; % class 4 labeled as 1
n_train = length(trainLabel);
n = length(testLabel);

%% Cluster
center = randi(255, [n_center, size(train_data, 2)]);
cluster = zeros(1, n_train);
while true
    % Classify
    prev_cluster = cluster;
    for i = 1:n_train
        min_distance = inf;
        for j = 1:n_center
            distance = norm(double(trainData(i, :)) - center(j, :));
            if distance < min_distance
                min_distance = distance;
                cluster(i) = j;
            end
        end
    end
    % Judge
    if sum(prev_cluster~=cluster, 'all') == 0
        break
    end
    % Update
    for i = 1:n_center
        center(i, :) = mean(trainData(cluster==i, :), 1);
    end
end

%% RBFN
r_train = dist(trainData, uint8(center'));
phi_train = exp(-r_train.^2/(2*stddev^2));
w = pinv(phi_train'*phi_train + lambda*eye(n_center)) * phi_train'*trainLabel;

r = dist(testData, uint8(center'));
phi = exp(-r.^2/(2*stddev^2));
testPred = phi * w;
trainPred = phi_train * w;
loss = mse(testPred, testLabel);
loss_train = mse(trainPred, trainLabel);

%% Performance and plot
trainAcc = zeros(1, 1000);
testAcc = zeros(1, 1000);
thr = zeros(1, 1000);
for i = 1: 1000
    t = (max(trainPred)-min(trainPred)) * (i-1)/1000 + min(trainPred);
    thr(i) = t;
    trainAcc(i) = (sum(trainLabel(trainPred<t)==0) + sum(trainLabel(trainPred>=t)==1)) / n_train;
    testAcc(i) = (sum(testLabel(testPred<t)==0) + sum(testLabel(testPred>=t)==1)) / n;
end
figure();
plot(thr, trainAcc, '.-', thr, testAcc, '^-');
ylim([0.45 1]); grid on;
h1 = legend('Train','Test');
h2 = xlabel('Threshold'); h3 = ylabel('Accuracy');
fprintf('Loss train: %f\nLoss test: %f\nAcc train max: %f\nAcc test max: %f\n', ...
        loss_train, loss, 100*max(trainAcc), 100*max(testAcc));
    
figure();
n_size = sqrt(size(center, 2));
for i = 1:n_center
    subplot(1, n_center, i);
    imshow(reshape(center(i, :), [n_size n_size]), [0 255]);
    title(sprintf('Cluster: %d', i), 'Interpreter', 'latex');
end
figure();
mean0 = mean(trainData(trainLabel==0, :), 1);
mean1 = mean(trainData(trainLabel==1, :), 1);
subplot(1, 2, 1);
imshow(reshape(mean0, [n_size n_size]), [0 255]);
h4 = title('Mean: 1');
subplot(1, 2, 2);
imshow(reshape(mean1, [n_size n_size]), [0 255]);
h5 = title('Mean: 2');

if n_center == 2
    figure();
	if sum(abs(mean0 - center(1, :))) < sum(abs(mean1 - center(1, :)))
        residual0 = abs(mean0 - center(1, :));
        residual1 = abs(mean1 - center(2, :));
    else
        residual0 = abs(mean1 - center(1, :));
        residual1 = abs(mean0 - center(2, :));
    end
    subplot(1, 2, 1);
    imshow(reshape(residual0, [n_size n_size]), [0 max(residual0)]);
    title('Residual: 1', 'Interpreter', 'latex');
    subplot(1, 2, 2);
    imshow(reshape(residual1, [n_size n_size]), [0 max(residual1)]);
    title('Residual: 2', 'Interpreter', 'latex');
end

set([h1 h2 h3 h4 h5], 'Interpreter', 'latex');
