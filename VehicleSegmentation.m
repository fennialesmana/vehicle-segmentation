clear;clc;close all;
%% READ DATA
video = VideoReader('traffic.mj2');
nFrames = ceil(video.FrameRate * video.Duration);
videoWidth = video.Width;
videoHeight = video.Height;
Mov = struct('cdata',zeros(videoHeight,videoWidth,3,'uint8'), 'colormap', []);

frames = zeros(videoHeight,videoWidth,3,nFrames);
k = 1;
while hasFrame(video)
    frames(:,:,:,k) = readFrame(video);
    k = k+1;
end
% END OF READ DATA
%% RETRIEVE BACKGROUND IMAGE
R = squeeze(frames(:,:,1,:));
G = squeeze(frames(:,:,2,:));
B = squeeze(frames(:,:,3,:));
rBackground = uint8(mode(R,3));
gBackground = uint8(mode(G,3));
bBackground = uint8(mode(B,3));
backgroundImage = cat(3,rBackground,gBackground,bBackground);
% END OF RETRIEVE BACKGROUND IMAGE
%% ROAD SEGMENTATION USING K-MEANS CLUSTERING
cform = makecform('srgb2lab');
lab_he = applycform(backgroundImage, cform);

ab = double(lab_he(:,:,2:3));
nrows = size(ab,1);
ncols = size(ab,2);
ab = reshape(ab,nrows*ncols,2); % 19200 (row col each a and b -> col1a;col2a;col3a;) * 2 (a and b)

nCluster = 3;
[clusterIdx, clusterCenter] = ...
    kmeans(ab, nCluster, 'distance', 'sqEuclidean', 'Replicates', 3);
pixelLabels = reshape(clusterIdx, nrows, ncols);

% get the largest cluster
clusterFreq = histcounts(pixelLabels);
largestIdx = find(max(clusterFreq) == clusterFreq);
pixelLabels(pixelLabels ~= largestIdx) = 0;
pixelLabels(pixelLabels == largestIdx) = 1;

% get the largest area of chosen cluster
labels = bwlabel(pixelLabels);
areaVal = unique(labels);
areaFreq = zeros(size(areaVal, 1), 1);
for i=1:size(areaVal, 1)
    areaFreq(i, 1) = sum(sum(labels == areaVal(i)));
end
largestVal = areaVal(find(max(areaFreq) == areaFreq));
roadMask = labels == largestVal;
roadMask = imfill(roadMask, 'holes');
% END OF ROAD SEGMENTATION USING K-MEANS CLUSTERING
%% OBJECT DETECTION
isShowImg = 0;
nCars = zeros(nFrames, 1);
for x=71:nFrames
    mixImage = uint8(frames(:,:,:,x));
    
    % Convert backgroundImage and foregroundImage to grayscale
    backgroundGray = rgb2gray(backgroundImage);
    foregroundGray = rgb2gray(mixImage);
    if isShowImg
        subplot(3,4,1);imshow(backgroundGray);title('Background');
        subplot(3,4,2);imshow(foregroundGray);title('Foreground');
    end
    
    % Substract backgroundImage and foregroundImage
    substractedImage = (double(backgroundGray)-double(foregroundGray));
    minVal = min(substractedImage(:));
    maxVal = max(substractedImage(:));
    substractedImage = ((substractedImage-minVal)/(maxVal-minVal))*255;
    substractedImage = uint8(substractedImage);
    if isShowImg
        subplot(3,4,3);imshow(substractedImage);title('Bg Subs');
    end
    
    % Use gradient magnitude to enhance the image
    hy = fspecial('sobel');
    hx = hy';
    Iy = imfilter(double(substractedImage), hy, 'replicate');
    Ix = imfilter(double(substractedImage), hx, 'replicate');
    gradmag = sqrt(Ix.^2 + Iy.^2);    
    substractedImage = uint8(gradmag);
    if isShowImg
        subplot(3,4,4);imshow(substractedImage);title('Gradient Magnitude');
    end
    
    % Use imfill to fill holes
    substractedImage = imfill(substractedImage,'holes');
    if isShowImg
        subplot(3,4,5);imshow(substractedImage);title('imfill');
    end
    
    % Convert image to binary image
    substractedImage = imbinarize(substractedImage);
    if isShowImg
        subplot(3,4,6);imshow(substractedImage);title('Binarization');
    end
    
    % Remove small area less than 10 pixel
    substractedImage = bwareaopen(substractedImage,10);
    if isShowImg
        subplot(3,4,7);imshow(substractedImage);title('bwareaopen');
    end
    
    % Dilate image
    substractedImage = imdilate(substractedImage,strel('line',4,4));
    if isShowImg
        subplot(3,4,8);imshow(substractedImage);title('Dilation');    
    end
    
    % Use imfill to fill holes
    substractedImage = imfill(substractedImage,'holes');
    if isShowImg
        subplot(3,4,9);imshow(substractedImage);title('imfill');
    end
    
    % Remove region outside roadMask
    substractedImage = double(substractedImage).*double(roadMask);
    if isShowImg
        subplot(3,4,10);imshow(substractedImage);title('Non-road removal');
    end
    
    % Area labeling
    labels = bwlabel(substractedImage);
    % Count number of objects
    nCars(x) = max(max(labels));

    bw = labels;

    [a,b] = size(bw);
    mask = false(a,b);
    
    for i=1:nCars(x)
        [row,col] = find(bw==i);
        mask(min(row):max(row),min(col):max(col)) = 1;
    end
    mask =  bwperim(mask,8);
    mask = imdilate(mask,strel('square',3));
    
    R = mixImage(:,:,1);
    G = mixImage(:,:,2);
    B = mixImage(:,:,3);
    R(mask) = 255;
    G(mask) = 0;
    B(mask) = 0;
    
    RGB = cat(3,R,G,B);
    
    % Insert number of objects to the image
    RGB = insertText(RGB,[1 20],nCars(x),'AnchorPoint','LeftBottom');
    
    Mov(x).cdata = RGB;
end
% END OF OBJECT DETECTION
%% CALCULATE ACCURACY
correct = [1;1;1;1;1;1;1;1;1;1;1;2;2;2;2;2;2;3;3;3;3;3;3;3;3;3;4;3;3;3;...
           3;3;3;3;3;3;3;3;3;3;2;2;2;3;3;2;2;2;2;3;3;4;4;4;4;3;3;3;3;3;...
           3;3;3;3;3;3;4;4;4;4;4;4;4;4;4;5;4;4;4;3;3;3;2;2;2;2;2;2;2;2;...
           3;3;3;3;3;3;3;3;2;2;2;2;2;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;0];
a = nCars == correct;
[colon(1, nFrames)' a nCars correct]
sum(a==1)/nFrames
% END OF CALCULATE ACCURACY
%% PLAY THE RESULT
hf = figure;
set(hf,'position',[300 300 videoWidth videoHeight]);

movie(hf,Mov,1,video.FrameRate);
close
% END OF PLAY THE RESULT