% run ManakinTracker or ManakinTracker2 on video 'sequenceName'
sequenceName = 'video';
frames = dir([sequenceName,'/*.jpg']);
groundtruth = dlmread(strcat(sequenceName, '/groundtruth.txt'));
groundtruth = groundtruth(1:min(end,numel(frames)),:);

% using CNN 'birdClassificationNet'
% note: CNNtrainedOnFirstHalf.mat and CNNtrainedOnSecondHalf.mat are CNNs 
% that are based on AlexNet and were fine-tuned using the first and second 
% half, respectively, of a dataset of videos showing the golden-collared 
% manakin performing its courtship dance.
load('CNNtrainedOnFirstHalf.mat','birdClassificationNet')

% with kalmanFilterType 'ConstantVelocity'
kalmanFilterType = 'ConstantVelocity';

% show visual output if showTrackingOutput is true
showTrackingOutput = true;

%firstFrame is first frame that contains ground truth annotation
firstFrame = find(sum(groundtruth,2)>0,1);

% version of tracker described in paper 'Tracking Golden-Collared Manakins in the Wild'
ManakinTracker(sequenceName, firstFrame, birdClassificationNet,kalmanFilterType,showTrackingOutput);


% updated version of tracker: uses correlation to better recognize 
% sitting male bird and avoid confusion with female bird
%ManakinTracker2(sequenceName, firstFrame, birdClassificationNet,kalmanFilterType,showTrackingOutput);





