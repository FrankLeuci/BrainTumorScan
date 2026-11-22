clear; clc; close all;
addpath('functions');
if ~exist('results', 'dir'), mkdir('results'); end

imds_yes = imageDatastore(fullfile('dataset', 'yes'));
imds_no  = imageDatastore(fullfile('dataset', 'no'));

files  = [imds_yes.Files; imds_no.Files];
labels = [true(numel(imds_yes.Files), 1); false(numel(imds_no.Files), 1)];
N      = numel(files);

target_size = [256, 256];
features    = zeros(N, 10);
pred_morph  = false(N, 1);

fprintf('Elaborazione di %d scansioni MRI in corso...\n', N);
for i = 1:N
    raw  = imread(files{i});
    proc = preprocessMRI(raw, target_size);
    [brain, m_brain] = skullStripping(proc);
    [m_tumor, ~]     = segmentTumor(brain, m_brain);

    pred_morph(i)  = any(m_tumor(:));
    features(i, :) = extractFeatures(brain, m_brain, m_tumor);
end

% Valutazione robusta: 5-Fold Stratified Cross-Validation con Weighted k-NN (k = 7)
rng(42);
K_folds = 5;
K_nn    = 7;

idx_yes = find(labels);
idx_no  = find(~labels);
idx_yes = idx_yes(randperm(numel(idx_yes)));
idx_no  = idx_no(randperm(numel(idx_no)));

fold_id = zeros(N, 1);
for i = 1:numel(idx_yes), fold_id(idx_yes(i)) = mod(i - 1, K_folds) + 1; end
for i = 1:numel(idx_no),  fold_id(idx_no(i))  = mod(i - 1, K_folds) + 1; end

pred_cv = false(N, 1);

for f = 1:K_folds
    test_idx  = find(fold_id == f);
    train_idx = find(fold_id ~= f);

    % Standardizzazione Z-score calcolata SOLO sul training fold
    mu_tr  = mean(features(train_idx, :), 1);
    sig_tr = std(features(train_idx, :), 0, 1) + 1e-6;

    X_train = (features(train_idx, :) - mu_tr) ./ sig_tr;
    X_test  = (features(test_idx,  :) - mu_tr) ./ sig_tr;
    y_train = labels(train_idx);

    for k = 1:numel(test_idx)
        x_q   = X_test(k, :);
        dists = sqrt(sum((X_train - x_q).^2, 2));
        [d_sorted, ord] = sort(dists, 'ascend');

        nn_d = d_sorted(1:K_nn);
        nn_y = y_train(ord(1:K_nn));
        w    = 1 ./ (nn_d + 1e-6);

        pred_cv(test_idx(k)) = sum(w(nn_y)) > sum(w(~nn_y));
    end
end

y_true = labels;
TP = sum( y_true &  pred_cv);
TN = sum(~y_true & ~pred_cv);
FP = sum(~y_true &  pred_cv);
FN = sum( y_true & ~pred_cv);

acc  = (TP + TN) / N;
sens = TP / max(1, TP + FN);
spec = TN / max(1, TN + FP);
prec = TP / max(1, TP + FP);
f1   = 2 * (prec * sens) / max(1e-6, prec + sens);

fprintf('\n--- RISULTATI 5-FOLD CROSS-VALIDATION (%d immagini) ---\n', N);
fprintf('TP: %d | TN: %d | FP: %d | FN: %d\n', TP, TN, FP, FN);
fprintf('Accuracy:    %.2f%%\n', acc * 100);
fprintf('Sensitivity: %.2f%%\n', sens * 100);
fprintf('Specificity: %.2f%%\n', spec * 100);
fprintf('F1-Score:    %.2f%%\n', f1 * 100);

% Esportazione CSV delle 10 feature e predizioni in results/
T = array2table(features, 'VariableNames', ...
    {'GlobalAsymP95', 'GlobalAsymMean', 'SkewSpread', 'Top5Ratio', 'GradP90', ...
     'AreaRatio', 'ContrastZ', 'AsymDiff', 'Solidity', 'NormThickness'});
T.File = files;
T.TrueLabel = labels;
T.PredMorph = pred_morph;
T.PredCV    = pred_cv;
writetable(T, fullfile('results', 'dataset_features_report.csv'));

% Visualizzazione e salvataggio della Matrice di Confusione
fig_cm = figure('Name', 'Confusion Matrix (5-Fold CV)', 'NumberTitle', 'off');
cm = [TP, FN; FP, TN];
imagesc(cm); colormap(parula); colorbar;
text(1, 1, sprintf('TP\n%d', TP), 'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold');
text(2, 1, sprintf('FN\n%d', FN), 'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold');
text(1, 2, sprintf('FP\n%d', FP), 'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold');
text(2, 2, sprintf('TN\n%d', TN), 'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold');
xticks([1 2]); xticklabels({'Pred: Yes', 'Pred: No'});
yticks([1 2]); yticklabels({'True: Yes', 'True: No'});
title(sprintf('5-Fold CV Confusion Matrix (Acc: %.1f%%, Sens: %.1f%%, Spec: %.1f%%)', acc*100, sens*100, spec*100));
saveas(fig_cm, fullfile('results', 'confusion_matrix.png'));
fprintf('Report e immagine salvati correttamente nella cartella results/.\n');