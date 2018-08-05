minimumArea = 2211; % minimum area for a blob
CNNWidthHeight = 227; % size of the CNN's input layer
birdLostSize = 500; % size of search window if bird is lost or out of frame
mainBlobMinScore = 0.9; % threshold for target score of first blob
addedBlobMinScore = 0.9; % threshold for target score of other blobs added to first blob
cnnOnlyThresh = 0.99; % target score has to be >= this threshold if classification with CNN only

% we assume the bird is sitting if two consecutive bounding boxes (not blobs) overlap by more than sittingThreshold
sittingThreshold = 0.8;


% ------  debug variables ---------
showSearchGrid = false;
showKalmanPrediction = true;
openInSeparateFigures = false;
showRespondingPositions = true;
showAllBlobs = true;
showForegroundMask = false;
isRecording = false;