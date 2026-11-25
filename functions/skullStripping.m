function [img_brain, brain_mask] = skullStripping(img_proc)
% SKULLSTRIPPING Isola l'encefalo in modo uniforme su scansioni T1, T2 e FLAIR.

% 1. Esclude le bande nere aggiunte dal padding
img_smooth = imgaussfilt(img_proc, 1.5);%sfoco per eliminare singoli px scuri dovuti al rumore all'interno del cuoio capelluto
non_pad = img_smooth > 0.015; %altrimenti i nero tirerebbe giù la soglia 1
if ~any(non_pad(:))
    disp("Imm nera dopo filtro gaussiano")
    brain_mask = false(size(img_proc));
    img_brain = img_proc;
    return;
end

%Soglie con metodo di Otsu (multithresh)
levels = multithresh(img_smooth(non_pad), 2);
valid_area = imerode(non_pad, strel('disk', 2));

head_mask = (img_smooth > levels(1)) & valid_area;
head_mask = bwareaopen(head_mask, 300);
head_mask = imclose(head_mask, strel('disk', 8));
head_mask = imfill(head_mask, 'holes');
head_mask = bwareafilt(head_mask, 1);

if sum(head_mask(:)) < 0.08 * numel(head_mask)
    head_mask = (img_smooth > (levels(1) * 0.5)) & valid_area;
    head_mask = imclose(head_mask, strel('disk', 8));
    head_mask = imfill(head_mask, 'holes');
    head_mask = bwareafilt(head_mask, 1);
end

if ~any(head_mask(:))
    brain_mask = false(size(img_proc));
    img_brain = img_proc;
    return;
end

% 3. Rimozione del collo nelle scansioni coronali/sagittali che toccano il bordo inferiore
bbox = regionprops(head_mask, 'BoundingBox');
aspect_ratio = bbox(1).BoundingBox(4) / max(1, bbox(1).BoundingBox(3));
bottom_head  = find(any(head_mask, 2), 1, 'last');
bottom_valid = find(any(valid_area, 2), 1, 'last');
row_widths   = sum(head_mask, 2);
[max_w, y_max_w] = max(row_widths);

if aspect_ratio > 1.22 && ~isempty(bottom_head) && ...
        (bottom_head >= bottom_valid - 2) && (row_widths(bottom_head) > 0.42 * max_w)
    search_start = min(size(head_mask, 1), y_max_w + round(0.35 * max_w));
    neck_rel = find(row_widths(search_start:end) < 0.68 * max_w, 1, 'first');
    if ~isempty(neck_rel)
        y_cut = search_start + neck_rel - 1;
        head_mask(y_cut:end, :) = false;
        head_mask = bwareafilt(head_mask, 1);
    end
end

head_stats  = regionprops(head_mask, 'EquivDiameter');
head_radius = head_stats.EquivDiameter / 2;

% 4. Pelatura geometrica uniforme dello strato esterno (cuoio capelluto + tavolato cranico)
%    Evita tagli basati sulla luminosità che scavano i solchi luminosi nelle T2
dist_from_skin = bwdist(~head_mask);
skull_thickness = max(4, 0.095 * head_radius);
brain_raw = dist_from_skin > skull_thickness;

if ~any(brain_raw(:))
    brain_raw = dist_from_skin > (0.06 * head_radius);
end

% 5. Regolarizzazione morfologica del contorno corticale
brain_raw  = imopen(brain_raw, strel('disk', 3));
se_close   = strel('disk', max(5, round(0.08 * head_radius)));
brain_mask = imclose(brain_raw, se_close);
brain_mask = imfill(brain_mask, 'holes');
brain_mask = bwareafilt(brain_mask, 1);

% 6. Applicazione della maschera
img_brain = img_proc .* double(brain_mask);
end