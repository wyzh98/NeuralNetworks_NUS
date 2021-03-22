clear;
max_epoch = 600;
lr0 = 0.1;
n_2d = 6;
n_slice = 800;
output_gif = true;
if output_gif
    handle = figure;
    axis tight manual;
    filename = 'q3b-600.gif';
end
X = randn(n_slice, 2);
s2 = sum(X.^2, 2);
trainX = (X.*repmat(1*(gammainc(s2/2, 1).^(1/2))./sqrt(s2), 1, 2))';
som = zeros(2, n_2d, n_2d);
sigma0 = sqrt(n_2d^2 + n_2d^2) / 2;
tau1 = max_epoch / log(sigma0);
tau2 = max_epoch;

for epoch = 1:max_epoch
    lr = lr0 * exp(- epoch/tau2); 
    sigma = sigma0 * exp(- epoch/tau1); 
    for i = 1:n_slice
        distance = squeeze(sum((trainX(:,i) - som).^2, 1))';
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
        h = permute(repmat(h0, [1 1 2]), [3 2 1]);
        som = som + lr*h.*(trainX(:,i) - som);
    end
    if output_gif
        gen_gif(trainX, som, epoch, n_2d, handle, filename);
    end
end

function gen_gif(trainX, som, epoch, n_2d, handle, filename)
    plot(trainX(1,:), trainX(2,:), '+r');
    hold on; grid on; axis equal;
    for i = 1:n_2d
        for j = 1:n_2d
            if i+1 <= n_2d
                plot([som(1,i,j),som(1,i+1,j)], [som(2,i,j),som(2,i+1,j)], 'bo-');
            end
            if j+1 <= n_2d
                plot([som(1,i,j),som(1,i,j+1)], [som(2,i,j),som(2,i,j+1)], 'bo-');
            end
        end
    end
    h1 = xlabel('$x$'); h2 = ylabel('$y$');
    h3 = legend('Train set', 'SOM', 'Location', 'northwest');
    h4 = title(sprintf('SOM function approximation. (Epoch: %g)', epoch));
    set([h1 h2 h3 h4], 'Interpreter', 'latex'); hold off;
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
