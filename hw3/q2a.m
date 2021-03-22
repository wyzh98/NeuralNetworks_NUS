% Matric No. A0225440R
% Choose class 0(N) and 4(A)
clear;
lambda = 0;
stddev = 100;

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

for lambda = [0 0.1 1 10 100 1000]
    r_train = dist(trainData');
    phi_train = exp(-r_train.^2/(2*stddev^2));
    w = pinv(phi_train'*phi_train + lambda*eye(n_train)) * phi_train'*trainLabel;

    r = dist(testData, trainData');
    phi = exp(-r.^2/(2*stddev^2));
    testPred = phi * w;
    trainPred = phi_train * w;
    loss = mse(testPred, testLabel);
    loss_train = mse(trainPred, trainLabel);

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
    h4 = title(sprintf('lambda = %g', lambda));
    set([h1 h2 h3 h4], 'Interpreter', 'latex');
    fprintf('Lambda: %g\nLoss train: %f\nLoss test: %f\nAcc train max: %f\nAcc test max: %f\n\n', ...
            lambda, loss_train, loss, 100*max(trainAcc), 100*max(testAcc));
end