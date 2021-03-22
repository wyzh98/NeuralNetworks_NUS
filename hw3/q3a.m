clear;
max_epoch = 600;
lr0 = 0.1;
n_neuron = 36;
n_slice = 400;
output_gif = true;
if output_gif
    handle = figure;
    axis tight manual
    filename = 'q3a-600.gif';
end
x = linspace(-pi, pi, n_slice);
trainX = [x; 2*sin(x)];
som = zeros(1, n_neuron);
sigma0 = sqrt(1^2 + n_neuron^2) / 2;
tau1 = max_epoch / log(sigma0);
tau2 = max_epoch;

for epoch = 1:max_epoch
    lr = lr0 * exp(- epoch/tau2); 
    sigma = sigma0 * exp(- epoch/tau1); 
    for i = 1:n_slice
        distance = sum((trainX(:,i) - som).^2, 1);
        [~, winner] = min(distance, [], 2); 
        d = abs([1:n_neuron] - winner); 
        h = exp(-d.^2 / (2*sigma^2)); 
        som = som + lr*h.*(trainX(:,i) - som);
    end
    if output_gif
        gen_gif(trainX, som, epoch, handle, filename);
    end
end

function gen_gif(trainX, som, epoch, handle, filename)
    plot(trainX(1,:), trainX(2,:), '+r');
    hold on; grid on;
    plot(som(1,:), som(2,:), 'bo-');
    h1 = xlabel('$x$'); h2 = ylabel('$y$');
    h3 = legend('Train set', 'SOM', 'Location', 'northwest');
    h4 = title(sprintf('SOM function approximation. Epoch: %g/600', epoch));
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


