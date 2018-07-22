function plot_eer(FPR, FNR)


% plots the detection error tradeoff (DET) curve

fnr = icdf(FNR);
fpr = icdf(FPR);
plot(fpr, fnr);

xtick = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, ...
         0.1, 0.2, 0.4, 0.7];
xticklabel = num2str(xtick * 100, '%g\n');
xticklabel = textscan(xticklabel, '%s'); xticklabel = xticklabel{1};
set (gca, 'xtick', icdf(xtick));
set (gca, 'xticklabel', xticklabel);
xlim(icdf([0.00051 0.7]));
xlabel ('False Positive Rate (FPR) [%]');

ytick = xtick;         
yticklabel = num2str(ytick * 100, '%g\n');
yticklabel = textscan(yticklabel, '%s'); yticklabel = yticklabel{1};
set (gca, 'ytick', icdf(ytick));
set (gca, 'yticklabel', yticklabel);
ylim(icdf([0.00051 0.7]));
ylabel ('False Negative Rate (FNR) [%]')

grid on;
box on;
axis square;
axis manual;
hold;

function y = icdf(x)
% computes the inverse of cumulative distribution function in x
y = -sqrt(2).*erfcinv(2 * ( x + eps));
y(isinf(y)) = nan;
