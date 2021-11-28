function X = LorenzFn(xrange, yrange, zrange, numICs, tSpan, seed, rho)

    % set random number generator seed
    rng(seed)

    % set lorenz parameters
    sigma = 10;
    beta = 8/3;
    %rho = 28;

    % define lorenz system
    % note that xyz = [x; y; z]
    lorenz_sys = @(t,xyz) [sigma*(xyz(2,:) - xyz(1,:)); 
                     xyz(1,:).*(rho - xyz(3,:)) - xyz(2,:); 
                     xyz(1,:).*xyz(2,:) - beta*xyz(3,:)];

    lenT = length(tSpan);
    X = zeros(numICs*lenT, 3);

    % generate data for random initial conditions
    for j = 1:numICs
        % choose random initial conditions in xrange x yrange x zrange
        x = (xrange(2) - xrange(1))*rand + xrange(1);
        y = (yrange(2) - yrange(1))*rand + yrange(1);
        z = (zrange(2) - zrange(1))*rand + zrange(1);

        ic = [x, y, z];

        % solve system and store data in X
        [~, temp] = ode45(lorenz_sys, tSpan, ic);
        X(1+(j-1)*lenT : lenT + (j-1)*lenT,:) = temp;
    end
end
