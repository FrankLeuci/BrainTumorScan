function img_proc = preprocessMRI(img_raw, target_size)

% Conversione in scala di grigi se a 3 canali
if size(img_raw, 3) == 3
    img_gray = rgb2gray(img_raw);
else
    img_gray = img_raw;
end


% Rimuove eventuali cornici di ritaglio di 2 pixel
if size(img_gray, 1) > 20 && size(img_gray, 2) > 20
    img_gray = img_gray(3:end-2, 3:end-2);
end

% Denoising mediano (preserva i bordi senza alterare la luminosità globale)
img_double   = im2double(img_gray);
img_denoised = medfilt2(img_double);

% Normalizzazione lineare della dinamica [0, 1] sul 99° percentile
fg_pixels = img_denoised(img_denoised > 0.02);
if ~isempty(fg_pixels)
    high_val = prctile(fg_pixels, 99);
    if high_val > 0.1
        img_denoised = min(1, img_denoised / high_val);
    end
end

% 5. Padding simmetrico a 0 (nero) e ridimensionamento
[r, c] = size(img_denoised);
max_dim = max(r, c);
pad_r = floor((max_dim - r) / 2);
pad_c = floor((max_dim - c) / 2);
img_square = padarray(img_denoised, [pad_r, pad_c], 0, 'both');

img_proc = imresize(img_square, target_size);
end