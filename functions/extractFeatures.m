function feat = extractFeatures(img_brain, brain_mask, tumor_mask)
% EXTRACTFEATURES Estrae 10 feature (5 globali sull'encefalo + 5 locali sulla lesione).

brain_vals = img_brain(brain_mask);
if isempty(brain_vals)
    feat = zeros(1, 10);
    return;
end

brain_area = numel(brain_vals);
props_b    = regionprops(brain_mask, 'Centroid');
shift_x    = round(size(img_brain, 2) / 2 - props_b(1).Centroid(1));

% --- A. 5 FEATURE GLOBALI (sempre calcolate, anche se tumor_mask è vuota) ---
img_c   = imtranslate(img_brain, [shift_x, 0]);
brain_c = imtranslate(brain_mask, [shift_x, 0]) & fliplr(imtranslate(brain_mask, [shift_x, 0]));
brain_c = imerode(brain_c, strel('disk', 4));

if any(brain_c(:))
    diff_map         = abs(img_c - fliplr(img_c));
    global_asym_p95  = prctile(diff_map(brain_c), 95);
    global_asym_mean = mean(diff_map(brain_c));
else
    global_asym_p95  = 0;
    global_asym_mean = 0;
end

med_b       = median(brain_vals);
std_b       = max(0.02, std(brain_vals));
skew_spread = (prctile(brain_vals, 95) - med_b) / std_b;
top5_vals   = brain_vals(brain_vals >= prctile(brain_vals, 95));
top5_ratio  = mean(top5_vals) / max(0.05, med_b);

safe_grad = imerode(brain_mask, strel('disk', 3));
if any(safe_grad(:))
    gmag     = imgradient(img_brain);
    grad_p90 = prctile(gmag(safe_grad), 90);
else
    grad_p90 = 0;
end

% --- B. 5 FEATURE LOCALI SULLA MASCHERA SEGMENTATA ---
tumor_area = sum(tumor_mask(:));
if tumor_area == 0
    feat = [global_asym_p95, global_asym_mean, skew_spread, top5_ratio, grad_p90, ...
        0, 0, 0, 0, 0];
    return;
end

area_ratio   = tumor_area / brain_area;
healthy_mask = brain_mask & ~imdilate(tumor_mask, strel('disk', 5));
if any(healthy_mask(:))
    h_med = median(img_brain(healthy_mask));
    h_std = max(0.02, std(img_brain(healthy_mask)));
else
    h_med = med_b;
    h_std = std_b;
end

tumor_vals = img_brain(tumor_mask);
contrast_z = (prctile(tumor_vals, 80) - h_med) / h_std;

tum_c       = imtranslate(tumor_mask, [shift_x, 0]);
contra_mask = fliplr(tum_c) & imtranslate(brain_mask, [shift_x, 0]);
if any(contra_mask(:))
    asym_diff = mean(tumor_vals) - median(img_c(contra_mask));
else
    asym_diff = mean(tumor_vals) - h_med;
end

cc    = bwconncomp(tumor_mask);
stats = regionprops(cc, 'Area', 'Solidity');
[~, idx_main] = max([stats.Area]);
solidity = stats(idx_main).Solidity;

d_map = bwdist(~tumor_mask);
norm_thickness = max(d_map(:)) / sqrt(brain_area);

feat = [global_asym_p95, global_asym_mean, skew_spread, top5_ratio, grad_p90, ...
    area_ratio, contrast_z, asym_diff, solidity, norm_thickness];
end