clear; clc; close all;
addpath('functions');

imds_yes = imageDatastore(fullfile('dataset', 'yes'));
imds_no  = imageDatastore(fullfile('dataset', 'no'));

% indici per testare diversi casi
idx_yes = 17;
idx_no  = 17;

raw_yes = readimage(imds_yes, idx_yes);
raw_no  = readimage(imds_no,  idx_no);

target_size = [256, 256];

% Pipeline su immagine si
proc_yes              = preprocessMRI(raw_yes, target_size);
[brain_yes, mbrain_y] = skullStripping(proc_yes);
[tum_yes, over_yes]   = segmentTumor(brain_yes, mbrain_y);

% Pipeline su immagine no
proc_no               = preprocessMRI(raw_no, target_size);
[brain_no, mbrain_n]  = skullStripping(proc_no);
[tum_no, over_no]     = segmentTumor(brain_no, mbrain_n);

% Visualizzazione a 5 colonne
fig = figure('Name', 'Pipeline Completa: Pre-processing, Skull Stripping e Segmentazione', ...
    'NumberTitle', 'off', ...
    'WindowStyle', 'normal', ...
    'Position', [60, 80, 1300, 560]);

subplot(2,5,1); imshow(raw_yes);   title('Tumore - Originale');
subplot(2,5,2); imshow(proc_yes);  title('1. Pre-processata');
subplot(2,5,3); imshow(brain_yes); title('2. Skull Stripped');
subplot(2,5,4); imshow(tum_yes);   title('3. Maschera Tumore');
subplot(2,5,5); imshow(over_yes);  title('4. Contorno Rilevato');

subplot(2,5,6);  imshow(raw_no);   title('Sano - Originale');
subplot(2,5,7);  imshow(proc_no);  title('1. Pre-processata');
subplot(2,5,8);  imshow(brain_no); title('2. Skull Stripped');
subplot(2,5,9);  imshow(tum_no);   title('3. Maschera Tumore');
subplot(2,5,10); imshow(over_no);  title('4. Contorno Rilevato');

figure(fig);