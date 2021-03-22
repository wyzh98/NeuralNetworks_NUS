clear;
load('characters10.mat');
max_epoch = 1000;
lr0 = 0.1;
n_2d = 10;
cell_size = 784;
output_gif = false;
if output_gif
    handle = figure;
    axis tight manual;
    filename = 'q3c-1000.gif';
end
trainIdx = find(train_label~=0 & train_label~=4);
trainLabel = train_label(trainIdx); 
trainData = train_data(trainIdx, :);
n_train = length(trainLabel);
som = zeros(cell_size, n_2d, n_2d);
sigma0 = sqrt(n_2d^2 + n_2d^2) / 2;
tau1 = max_epoch / log(sigma0);
tau2 = max_epoch;

for epoch = 1:max_epoch
    lr = lr0 * exp(- epoch/tau2); 
    sigma = sigma0 * exp(- epoch/tau1); 
    for i = 1:n_train
        distance = squeeze(sum((double(trainData(i,:))' - som).^2, 1))';
        [~, winner] = min(distance, [], 'all', 'linear'); 
        row = ceil(winner/n_2d);
        col = mod(winner, n_2d);
        if col == 0
            col = n_2d;
        end
        d0 = ([1:n_2d] - row).^2; 
        d1 = ([1:n_2d] - col).^2; 
        d = d1' + d0; 
        h0 = exp(-d ./ (2*sigma^2));
        h = permute(repmat(h0, [1 1 cell_size]), [3 2 1]);
        som = som + lr*h.*(double(trainData(i,:))' - som);
    end
    if output_gif
        if epoch == 1 || mod(epoch, 50) == 0
            gen_gif(som, epoch, n_2d, handle, filename);
        end
    end
end

%% Test
LABELMAP = [[6 6 5 5 5 7 7 7 7 7];
            [6 6 5 5 9 7 7 7 7 7];
            [6 6 6 5 9 9 7 2 2 2];
            [6 6 6 5 5 9 2 2 2 2];
            [6 6 6 9 1 1 1 2 2 2];
            [6 9 9 9 1 1 1 8 2 2];
            [1 9 9 9 1 1 1 8 8 8];
            [1 3 9 3 1 1 1 8 8 8];
            [3 3 3 3 1 1 1 8 8 8];
            [3 3 3 3 1 1 1 8 8 8]];
testIdx = find(test_label~=0 & test_label~=4);
testLabel = test_label(testIdx); 
testData = test_data(testIdx, :);
testPred = zeros(size(testLabel));
n_test = length(testLabel);
for i = 1:n_test
    distance = squeeze(sum((double(testData(i,:))' - som).^2, 1))';
    [~, winner] = min(distance, [], 'all', 'linear'); 
    row = ceil(winner/n_2d);
    col = mod(winner, n_2d);
    if col == 0
        col = n_2d;
    end
    testPred(i) = LABELMAP(row, col);
end
testAcc = sum(testLabel == testPred) / n_test;
fprintf('Test accuracy: %g\n', testAcc*100);

%% Utilities
function gen_gif(som, epoch, n_2d, handle, filename)
    hold on; axis equal;
    for i = 1:n_2d
        for j = 1:n_2d
            subplot(n_2d, n_2d, n_2d*(i-1)+j);
            imshow(reshape(som(:,i,j), [28 28]), [0 255], 'InitialMagnification', 'fit');
        end
    end
    h1 = suptitle(sprintf('SOM conceptual/semantic map. (Epoch: %g)', epoch));
    set([h1], 'Interpreter', 'latex'); hold off;
    drawnow
        frame = getframe(handle); 
        im = frame2im(frame); 
        [imind, cm] = rgb2ind(im, 256);
        if epoch == 1
            imwrite(imind, cm, filename, 'gif', 'Loopcount', inf); 
        else 
            imwrite(imind, cm,filename, 'gif', 'WriteMode','append'); 
        end 
end
