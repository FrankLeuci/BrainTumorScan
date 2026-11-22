function [tumor_mask, tumor_overlay] = segmentTumor(img_brain, brain_mask)
% SEGMENTTUMOR Segmenta tumori solidi, lobulati e a mezzaluna confrontando l'emisfero sano.

    tumor_mask = false(size(img_brain));
    brain_vals = img_brain(brain_mask);

    if isempty(brain_vals)
        tumor_overlay = repmat(img_brain, [1 1 3]);
        return;
    end

    % 1. Margine corticale di sicurezza ridotto a 2 px per non tagliare i meningiomi periferici
    safe_brain = imerode(brain_mask, strel('disk', 2));
    if ~any(safe_brain(:))
        safe_brain = brain_mask;
    end

    % 2. Geometria dell'encefalo e statistiche sull'emisfero più sano
    brain_props = regionprops(brain_mask, 'Centroid', 'BoundingBox');
    x_mid   = brain_props(1).Centroid(1);
    y_mid   = brain_props(1).Centroid(2);
    brain_w = brain_props(1).BoundingBox(3);
    brain_h = brain_props(1).BoundingBox(4);

    left_mask  = brain_mask; left_mask(:, round(x_mid):end) = false;
    right_mask = brain_mask; right_mask(:, 1:max(1, round(x_mid)-1)) = false;

    if sum(left_mask(:)) > 100 && sum(right_mask(:)) > 100
        l_vals = img_brain(left_mask);
        r_vals = img_brain(right_mask);
        if prctile(l_vals, 80) <= prctile(r_vals, 80)
            ref_vals = l_vals;
        else
            ref_vals = r_vals;
        end
    else
        ref_vals = brain_vals;
    end

    med_val     = median(ref_vals);
    p95_ref     = prctile(ref_vals, 95);
    robust_std  = max(0.03, 1.4826 * median(abs(ref_vals - med_val)));
    seed_thresh = min(0.78, med_val + 1.55 * robust_std);

    % 3. Estrazione e pulizia morfologica dei seed (solidi e ad anello)
    seed_raw     = (img_brain > seed_thresh) & safe_brain;
    seed_closed  = imclose(seed_raw, strel('disk', 6));
    seed_filled  = imfill(seed_closed, 'holes');
    seed_cleaned = imopen(seed_filled, strel('disk', 3));

    % 4. Mappa di confronto con l'emisfero controlaterale
    shift_x        = round(size(img_brain, 2) / 2 - x_mid);
    img_centered   = imtranslate(img_brain, [shift_x, 0]);
    brain_centered = imtranslate(brain_mask, [shift_x, 0]);
    asym_centered  = max(0, imgaussfilt(img_centered, 1.5) - imgaussfilt(fliplr(img_centered), 2.5));
    asym_map       = imtranslate(asym_centered, [-shift_x, 0]);

    % 5. Filtraggio geometrico (tollerante a forme lobulate/mezzaluna) e contrasto
    brain_area    = sum(brain_mask(:));
    min_seed_area = max(45, round(0.008 * brain_area));
    max_seed_area = round(0.45 * brain_area);

    cc = bwconncomp(seed_cleaned);
    valid_pixels = {};
    valid_areas  = [];

    if cc.NumObjects > 0
        stats = regionprops(cc, 'Area', 'Solidity', 'Eccentricity', ...
                            'MajorAxisLength', 'MinorAxisLength');
        for k = 1:cc.NumObjects
            idx_k = cc.PixelIdxList{k};
            [r_k, c_k] = ind2sub(size(img_brain), idx_k);

            cx = mean(c_k);
            cy = mean(r_k);
            max_lat_dist = max(abs(c_k - x_mid));
            elongation   = stats(k).MajorAxisLength / max(1, stats(k).MinorAxisLength);

            % Esclude ventricoli e bande periventricolari lungo la linea mediana
            is_midline_strip = (abs(cx - x_mid) < 0.15 * brain_w) && ...
                               (abs(cy - y_mid) < 0.28 * brain_h) && ...
                               (max_lat_dist < 0.27 * brain_w);

            % Spessore interno (scarta solchi corticali sottili)
            cand_mask = false(size(img_brain));
            cand_mask(idx_k) = true;
            d_map = bwdist(~cand_mask);
            max_thickness = max(d_map(:));

            % Confronto con la mediana della regione speculare nell'emisfero opposto
            cand_centered = imtranslate(cand_mask, [shift_x, 0]);
            contra_roi    = imdilate(fliplr(cand_centered), strel('disk', 8)) & brain_centered;

            cand_p80 = prctile(img_brain(idx_k), 80);
            if any(contra_roi(:))
                contra_med = median(img_centered(contra_roi));
            else
                contra_med = med_val;
            end

            contra_diff = cand_p80 - contra_med;
            z_score     = (cand_p80 - med_val) / robust_std;

            valid_area  = stats(k).Area >= min_seed_area && stats(k).Area <= max_seed_area;
            valid_shape = stats(k).Solidity > 0.48 && stats(k).Eccentricity < 0.93 && ...
                          elongation < 2.8 && max_thickness >= 3.2;

            % Ammette lesioni mediane solo se molto iperintense (es. meningiomi della falce)
            effective_midline = is_midline_strip && (z_score < 3.0);
            is_distinct = (z_score > 1.65) && (contra_diff > 0.13) && (cand_p80 > p95_ref + 0.03);

            if valid_area && valid_shape && is_distinct && ~effective_midline
                valid_pixels{end+1} = idx_k; %#ok<AGROW>
                valid_areas(end+1)  = stats(k).Area; %#ok<AGROW>
            end
        end
    end

    seed_mask = false(size(img_brain));
    if ~isempty(valid_areas)
        max_cand_area = max(valid_areas);
        for k = 1:numel(valid_areas)
            if valid_areas(k) >= 0.45 * max_cand_area
                seed_mask(valid_pixels{k}) = true;
            end
        end
    end

    % 6. Espansione guidata dall'asimmetria per inglobare lobi tumorali eterogenei
    if any(seed_mask(:))
        [X, Y] = meshgrid(1:size(img_brain, 2), 1:size(img_brain, 1));
        midline_guard = (abs(X - x_mid) < 0.11 * brain_w) & (abs(Y - y_mid) < 0.25 * brain_h);

        grow_thresh = med_val + 0.85 * robust_std;
        local_roi   = imdilate(seed_mask, strel('disk', 22)) & safe_brain & ~midline_guard;
        grow_zone   = ((img_brain > grow_thresh) & (asym_map > 0.09) & local_roi) | seed_mask;

        expanded   = imreconstruct(seed_mask, grow_zone);
        expanded   = imclose(expanded, strel('disk', 7));
        tumor_mask = imfill(expanded, 'holes') & safe_brain;
    end

    % 7. Sovrapposizione del contorno rosso
    contour_mask = imdilate(tumor_mask, strel('disk', 1)) & ~imerode(tumor_mask, strel('disk', 1));
    R = img_brain; G = img_brain; B = img_brain;
    R(contour_mask) = 1; G(contour_mask) = 0; B(contour_mask) = 0;
    tumor_overlay = cat(3, R, G, B);
end