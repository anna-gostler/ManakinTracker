function [predictedBoundingBoxes] = ManakinTracker(sequenceName, firstFrame, birdClassificationNet, kalmanFilterType, showFigure)
config;
frames = dir([sequenceName,'/*.jpg']);
groundtruth = dlmread(strcat(sequenceName, '/groundtruth.txt'));
groundtruth = groundtruth(1:end,:);

% ------  prepare blob detector ---------
detector = vision.ForegroundDetector(...
    'NumTrainingFrames', 100, ...
    'InitialVariance', 30*30);

blob = vision.BlobAnalysis(...
    'CentroidOutputPort', false, 'AreaOutputPort', false, ...
    'BoundingBoxOutputPort', true, ...
    'MinimumBlobAreaSource', 'Property', 'MinimumBlobArea', minimumArea);
% ---------------------------------------

close all
if showFigure
    figure
    hold on
end
skipFrames = 0;
initialized = false;
bird_out_of_frame = false;
bird_lost = false;
birdIsSitting = false;

%-------------store output here:-----------
% column 1-4 contains bounding boxes;
% column 5 type of prediction
% (0: initial bb, 1: based on multiple candidates; 2: based on blob; 3: based on one candidate; 4: based on kalman filter; 5: based on GT(init); 6: keep prev. bb)
% column 6 overlap with GT
% column 7 bird is sitting? yes:1; no:0
% column 8 bird is out of frame  or lost? yes:1; no:0
predictedBoundingBoxes = zeros(500,8);

tic
for i = firstFrame: min(numel(frames), size(groundtruth,1))
    
    % ------  look for first frame with ground truth and initialize tracker ---------
    while initialized==false
        startFrame = i + skipFrames;
        if sum(groundtruth(startFrame)) > 0
            predictedBB = groundtruth(startFrame,:);
            initialized = true;
            if strcmp(kalmanFilterType,'ConstantAcceleration')
                kalmanFilter = configureKalmanFilter('ConstantAcceleration',groundtruth(startFrame,:),[1 1 1 ]*1e5, [25, 10, 1], 25);
            elseif strcmp(kalmanFilterType,'ConstantVelocity')
                kalmanFilter = configureKalmanFilter('ConstantVelocity',groundtruth(startFrame,:),[1 1]*1e5, [25, 10], 25);
            else
                disp('ERROR: choose valid kalman filter type (ConstantAcceleration or ConstantVelocity)')
            end
            candidateRegionWidth = groundtruth(startFrame,3)*3;
            candidateRegionHeight = groundtruth(startFrame,4)*3;
            candidateRegion = [groundtruth(startFrame,1)+groundtruth(startFrame,3)/2-(candidateRegionWidth/2) groundtruth(startFrame,2)+groundtruth(startFrame,4)/2-(candidateRegionHeight/2) candidateRegionWidth candidateRegionHeight];
            
            predictedBoundingBoxes(startFrame,1:4)=groundtruth(startFrame,:);
            predictedBoundingBoxes(startFrame,5) = 0;
            predictedBoundingBoxes(startFrame,6) = 1;
            skipFrames = skipFrames+1;
        else
            skipFrames = skipFrames+1;
        end
    end
    i = i + skipFrames;

    if i < min(numel(frames), size(groundtruth,1))
        frame = imread([sequenceName,'/',frames(i).name]);
        
        if openInSeparateFigures
            figure
        end
        
        if showFigure
            clf
            title(num2str(i))
            imshow(frame)
            hold on
        end
        
        if sum(groundtruth(i)) > 0 && showFigure
            rectangle('Position', groundtruth(i,:), 'EdgeColor','g','LineWidth',2, 'LineStyle', '--')
        end
        
        predictedBoundingBoxes(i,8) = bird_out_of_frame || bird_lost;
        
        % --  generate candidate locations --------------------------------
        % --  if we lost the bird: fill entire frame with candidate locations
        % --  if bird is out of frame: fill line edges with candidate locations
        % --  otherwise: candidate locations are the shifted prev. bbox  
        positions = [];
        if bird_lost
            %fill frame with candidate positions
            candidateRegion(3) = birdLostSize;
            candidateRegion(4) = birdLostSize;
            for j = 0 : 1 : (ceil(size(frame,1)/candidateRegion(4))*2)
                candidateRegion(2) = (candidateRegion(4)/3)*j+1;
                for k = 0 : 1 : (ceil(size(frame,2)/candidateRegion(3)))
                    positions = [positions;generateCandidatePositions(candidateRegion,bird_lost,'top')];
                    candidateRegion(1) = candidateRegion(3)*k + 1;
                end
            end
            for j = 0 : 1 :(ceil(size(frame,2)/candidateRegion(3))*2)+2
                candidateRegion(1) = (candidateRegion(3)/3)*j+1;
                for k = 0 : 1 : ceil(size(frame,1)/candidateRegion(4))
                    positions = [positions;generateCandidatePositions(candidateRegion,bird_lost,'left')];
                    candidateRegion(2) = candidateRegion(4)*k + 1;
                end
            end
            
        elseif bird_out_of_frame
            %line edges of frame with candidate positions
            candidateRegion(1) = 1;
            candidateRegion(2) = 1;
            candidateRegion(3) = birdLostSize;
            candidateRegion(4) = birdLostSize;
            
            for j = 0 : 0.5 : (ceil(size(frame,2)/candidateRegion(3)))
                candidateRegion(1) = candidateRegion(3)*j + 1;
                positions = [positions;generateCandidatePositions(candidateRegion,bird_out_of_frame,'top')];
            end
            
            candidateRegion(1) = 1;
            candidateRegion(2) = 1;
            for j = 0 : 0.5 : ceil(size(frame,1)/candidateRegion(4))
                candidateRegion(2) = candidateRegion(4)*j+1;
                positions = [positions;generateCandidatePositions(candidateRegion,bird_out_of_frame,'left')];
            end
            
            candidateRegion(1) = size(frame,2)-(candidateRegion(4)/3);
            candidateRegion(2) = 1;
            for j = 0 : 0.5 : ceil(size(frame,1)/candidateRegion(4))
                candidateRegion(2) = candidateRegion(4)*j+1;
                positions = [positions;generateCandidatePositions(candidateRegion,bird_out_of_frame,'right')];
            end
            title('bird_out_of_frame: looking for bird')
        else
            % "stay"-search region; assumes the bird is at the same position as
            % previous frame (sitting)
            positions = generateCandidatePositions(candidateRegion, bird_out_of_frame);
        end
        
        if ~bird_out_of_frame  
            kalmanPredictedLoc = predict(kalmanFilter);
            if showKalmanPrediction && kalmanPredictedLoc(3)>0 && kalmanPredictedLoc(4)>0 && showFigure
                rectangle('Position', kalmanPredictedLoc, 'EdgeColor','k','LineWidth',2,'LineStyle', '--')
            end
        end
        
        %---------------------estimate current bounding box----------------
        %-------get blobs--------------------------------------------------
        fgMask  = step(detector, frame);
        blobs   = step(blob, fgMask);
        
        if(showForegroundMask)
            figure
            imshow(fgMask)
        end
        
        found_blob_based_prediction = false;
        includedBlobs = 0;
        
        %find blob with smallest distance to kalmanPredictedLoc 
        minDistanceToKalmanPredictedLoc = 1000;
        firstChosenBlob = 0;
        for b = 1: size(blobs,1)
            %----- classify blob -----
            I_crop = imcrop(frame, blobs(b,:));
            I_resized = imresize(I_crop, [CNNWidthHeight CNNWidthHeight]);
            [~,blob_score] = classify(birdClassificationNet,I_resized);
            if showFigure && showAllBlobs
                text(double(blobs(b,1))+10, double(blobs(b,2))+30,num2str(blob_score(2)), 'Color', 'white','FontSize',15)
                rectangle('Position', blobs(b,:), 'EdgeColor','b','LineWidth',1, 'LineStyle', '--')
            end
            if blob_score(2) > mainBlobMinScore
                if (bird_lost || bird_out_of_frame) || ( minDistBB(kalmanPredictedLoc, blobs(b,:)) < minDistanceToKalmanPredictedLoc)
                    minDistanceToKalmanPredictedLoc = minDistBB(kalmanPredictedLoc, blobs(b,:));
                    includedBlobs = 1;
                    minX = blobs(b,1);
                    minY = blobs(b,2);
                    maxX = blobs(b,1) + blobs(b,3);
                    maxY = blobs(b,2) + blobs(b,4);
                    firstChosenBlob = b;
                end
            end
        end
        if firstChosenBlob > 0 && showFigure
            rectangle('Position', blobs(firstChosenBlob,:), 'EdgeColor','b','LineWidth',3, 'LineStyle', '--')            
        end
        
        
        % add blobs that are close to first chosen blob
        if includedBlobs > 0 && birdIsSitting
            for b = 1: size(blobs,1)
                if b ~= firstChosenBlob
                    
                    %----- classify blob -----
                    I_crop = imcrop(frame, blobs(b,:));
                    I_resized = imresize(I_crop, [CNNWidthHeight CNNWidthHeight]);
                    [~,blob_score] = classify(birdClassificationNet,I_resized);

                    if blob_score(2) >  addedBlobMinScore
                        %an added blob must be close to the first chosen blob and
                        %other added blobs
                        %it should be not further away than the bounding box is large (max of width and height)
                        %max(maxX-minX, maxY-minY)
                        %minDistBB([minX minY maxX-minX maxY-minY], blobs(b,:))
                        if minDistBB([minX minY maxX-minX maxY-minY], blobs(b,:)) < max(max(maxX-minX, maxY-minY),max(blobs(b,3),blobs(b,4)))
                            includedBlobs = includedBlobs + 1;
                            if blobs(b,1) < minX
                                minX = blobs(b,1);
                            end
                            if blobs(b,2) < minY
                                minY = blobs(b,2);
                            end
                            if blobs(b,1) + blobs(b,3) > maxX
                                maxX = blobs(b,1) + blobs(b,3);
                            end
                            if blobs(b,2) + blobs(b,4) > maxY
                                maxY = blobs(b,2) + blobs(b,4);
                            end
                            if showFigure
                                rectangle('Position', blobs(b,:), 'EdgeColor','b','LineWidth',2, 'LineStyle', '--')
                            end
                        end
                    end
                end
            end
        end
        
        if includedBlobs > 0
            found_blob_based_prediction = true;
            bird_lost = false;
            bird_out_of_frame = false;
            
            predictedBoundingBoxes(i,5) = 2;
            predictedBB = double([minX minY maxX-minX maxY-minY]);
            
            kalmanPredictedLoc = correct(kalmanFilter, predictedBB);
            candidateRegionWidth = predictedBB(3)*3;
            candidateRegionHeight = predictedBB(4)*3;
            candidateRegion = [predictedBB(1)+predictedBB(3)/2-(candidateRegionWidth/2) predictedBB(2)+predictedBB(4)/2-(candidateRegionHeight/2) candidateRegionWidth candidateRegionHeight];
            if showFigure
                rectangle('Position', predictedBB, 'EdgeColor','b','LineWidth',3, 'LineStyle', '-')
            end
        end
        
        %if prev bbox was blob and this bbox is blob and this bbox is
        %contained inside prev. bbox -> bird is sitting
        if predictedBoundingBoxes(i-1,5) == 2 && predictedBoundingBoxes(i,5) == 2 && isBox1InsideBox2(predictedBB, predictedBoundingBoxes(i-1,1:4),10)
            birdIsSitting = true;
        end
            
        
        if (found_blob_based_prediction == false && ~bird_out_of_frame)
            birdIsSitting = false;
            
            %----- get scores for all candidate locations ----- 
            %put candidates cropped at candidate position into 4D image array
            candidates = zeros(CNNWidthHeight, CNNWidthHeight, 3, size(positions,1));
            for p=1:size(positions,1)
                position = positions(p,:);
                I_crop = imcrop(frame,position);
                if p == 1 && (size(I_crop,1) < candidateRegion(4)/5 || size(I_crop,2) < candidateRegion(3)/5) %p==1 is the center position
                    bird_out_of_frame = true;
                    predictedBoundingBoxes(i,8) = bird_out_of_frame;
                end
                if ~isempty(I_crop)
                    I_resized = imresize(I_crop, [CNNWidthHeight CNNWidthHeight]);
                    candidates(:,:,:,p) = I_resized;
                else
                    candidates(:,:,:,p) = nan;
                end
                
                if (bird_out_of_frame || showSearchGrid || bird_lost) && showFigure
                    rectangle('Position', position,'EdgeColor', 'y','LineWidth',1,'LineStyle','--')
                end
            end
            [~,scores] = classify(birdClassificationNet,candidates);
            candidateScores = scores;
            
            if bird_out_of_frame
                disp(strcat('it seems the bird has left the frame ',num2str(i)))
                if showFigure
                    title(strcat('it seems the bird has left the frame ',num2str(i)))
                end
            end
            %---------------------END: get scores for all candidate locations ---------------------
            
            %get max score for bird (second element of scores)
            [maxScore,maxI] = max(candidateScores(:,2));
            
            % if center box contains a bird and is same as initial
            % groundtruth -> choose as target bounding box
            if candidateScores(1,2) >= cnnOnlyThresh && sum(groundtruth(startFrame,:) == positions(1,:)) == 4 
                predictedBB = positions(1,:);
                predictedBoundingBoxes(i,5) = 0;
                rectangle('Position',kalmanPredictedLoc,'EdgeColor','g','LineWidth',1)
                birdIsSitting = true;
                predictedBoundingBoxes(i,7) = 1;
                
            elseif sum(candidateScores(:,2) >= cnnOnlyThresh) > 1 % decision not conclusive: target found in multiple candidate locations
                aboveThreshCandidates = find(candidateScores(:,2) >= cnnOnlyThresh);
                
                %---- find set of overlapping candidate locations above target threshold
                
                %puts all aboveThreshCandidates in a matrix defining
                %overlapping groups
                %overlapping elements (position-index) are in one row; 
                %zero entries must be ignored
                %example: positions 1 and 13 overlap, 7 does not overlap another;
                %there are 4 aboveThreshCandidates in total
                %position -> row 1 = [1, 13, 0, 0] row 3 = [7, 0, 0, 0]
                addedElem = false;
                for aboveThreshCandidate = 1:numel(aboveThreshCandidates)
                    addedElem = false;
                    if aboveThreshCandidate == 1
                        overlappingGroups = zeros(1,numel(aboveThreshCandidates));
                        overlappingGroups(1,1) = aboveThreshCandidates(aboveThreshCandidate);
                    else
                        for overlappingGroup = 1: size(overlappingGroups,1)
                            for overlappingGroupElement = 1: numel(aboveThreshCandidates)
                                %for all elements
                                if overlappingGroups(overlappingGroup,overlappingGroupElement) ~= 0 && bboxOverlapRatio(positions(overlappingGroups(overlappingGroup,overlappingGroupElement),:),positions(aboveThreshCandidates(aboveThreshCandidate),:)) > 0
                                    %put new element in overlapping-group
                                    freeSpaces = find(~overlappingGroups(overlappingGroup,:));
                                    overlappingGroups(overlappingGroup,freeSpaces(1)) = aboveThreshCandidates(aboveThreshCandidate);
                                    addedElem = true;
                                    
                                end
                                if(addedElem)
                                    break;
                                end
                            end
                            if(addedElem)
                                break;
                            end
                        end
                        if ~addedElem
                            overlappingGroups = [overlappingGroups; zeros(1,numel(aboveThreshCandidates))];
                            overlappingGroups(size(overlappingGroups,1),1) = aboveThreshCandidates(aboveThreshCandidate);
                        end
                    end
                end
                %find largest group of overlapping positions
                [~, maxoverlappingGroup] = max(sum(overlappingGroups ~= 0,2));
                
                if(nnz(overlappingGroups(maxoverlappingGroup,:)) > 0)
                    
                    %show only largest overlapping group
                    averageOverlappingBox = [0, 0, 0, 0];
                    for j = 1:nnz(overlappingGroups(maxoverlappingGroup,:))
                        if showRespondingPositions && showFigure
                            rectangle('Position', positions(overlappingGroups(maxoverlappingGroup,j),:),'EdgeColor','r','LineWidth',1)
                        end
                        averageOverlappingBox = averageOverlappingBox + positions(overlappingGroups(maxoverlappingGroup,j),:);
                    end
                    
                    averageOverlappingBox = averageOverlappingBox ./ (nnz(overlappingGroups(maxoverlappingGroup,:)));
    
                    for j = 1:nnz(overlappingGroups(maxoverlappingGroup,:))
                        currentPosition = positions(overlappingGroups(maxoverlappingGroup,j),:);
                        if j == 1
                            minX = currentPosition(1);
                            minY = currentPosition(2);
                            maxX = currentPosition(1) + currentPosition(3);
                            maxY = currentPosition(2) + currentPosition(4);
                        else
                            if currentPosition(1) < minX
                                minX = currentPosition(1);
                            end
                            if currentPosition(2) < minY
                                minY = currentPosition(2);
                            end
                            if currentPosition(1) + currentPosition(3) > maxX
                                maxX = currentPosition(1) + currentPosition(3);
                            end
                            if currentPosition(2) + currentPosition(4) > maxY
                                maxY = currentPosition(2) + currentPosition(4);
                            end
                        end
                    end
                    
                    if exist('maximumOverlappingBox')
                        maximumOverlappingBox_prev = maximumOverlappingBox;
                    end
                    
                    maximumOverlappingBox = [minX minY maxX-minX maxY-minY];
                    
                    if exist('maximumOverlappingBox_prev') && bboxOverlapRatio(maximumOverlappingBox, maximumOverlappingBox_prev) > sittingThreshold
                        birdIsSitting = true;
                        predictedBoundingBoxes(i,7) = 1;
                    else
                        birdIsSitting = false;
                    end
                    
                    %predicted bounding box is average of all positions in the largest overlapping group
                    predictedBB = averageOverlappingBox;
                    
                    if birdIsSitting
                        if strcmp(kalmanFilterType,'ConstantAcceleration')
                            kalmanFilter = configureKalmanFilter('ConstantAcceleration',predictedBB,[1 1 1 ]*1e5, [25, 10, 1], 25);
                        elseif strcmp(kalmanFilterType,'ConstantVelocity')
                            kalmanFilter = configureKalmanFilter('ConstantVelocity',predictedBB,[1 1]*1e5, [25, 10], 25);
                        end
                    end
                    
                    if birdIsSitting
                        bird_lost = false;
                    end
                    
                    if kalmanPredictedLoc(3)>0 && kalmanPredictedLoc(4)>0 && showFigure
                        rectangle('Position',kalmanPredictedLoc,'EdgeColor','k','LineWidth',1)
                    end
                    
                end

                candidateRegionWidth  = min(500,predictedBB(3)*3);
                candidateRegionHeight = min(500,predictedBB(4)*3);
                candidateRegionWidth  = max(100,candidateRegionWidth);
                candidateRegionHeight = max(100,candidateRegionHeight);
                
                kalmanPredictedLoc = correct(kalmanFilter, predictedBB);
                candidateRegion = [predictedBB(1)+predictedBB(3)/2-(candidateRegionWidth/2) predictedBB(2)+predictedBB(4)/2-(candidateRegionHeight/2) candidateRegionWidth candidateRegionHeight];
                
            elseif maxScore >= cnnOnlyThresh
                    predictedBB = positions(maxI,:);
                    candidateRegionWidth  = min(500,predictedBB(3)*3);
                    candidateRegionHeight = min(500,predictedBB(4)*3);
                    candidateRegionWidth  = max(100,candidateRegionWidth);
                    candidateRegionHeight = max(100,candidateRegionHeight);
                    
                    candidateRegion = [predictedBB(1)+predictedBB(3)/2-(candidateRegionWidth/2) predictedBB(2)+predictedBB(4)/2-(candidateRegionHeight/2) candidateRegionWidth candidateRegionHeight];
                    
                    correct(kalmanFilter, predictedBB);
                    
                    if kalmanPredictedLoc(3) > 0 && kalmanPredictedLoc(4) > 0 && bboxOverlapRatio(kalmanPredictedLoc, predictedBB) < 0.01
                        predictedBB = kalmanPredictedLoc;
                        if showFigure
                            
                            rectangle('Position', kalmanPredictedLoc,'EdgeColor','k','LineWidth',3,'LineStyle','--')
                        end
                    else
                        if showFigure
                            rectangle('Position', predictedBB,'EdgeColor',[0.4 0 0],'LineWidth',3, 'LineStyle','-')
                        end
                    end
                    predictedBoundingBoxes(i,5) = 3;
                
            elseif ~bird_lost && ~ birdIsSitting %use kalman predicted location if no reliable prediction by cnn
                predictedBB = kalmanPredictedLoc;
                candidateRegionWidth  = min(500,candidateRegion(3)*3); 
                candidateRegionHeight = min(500,candidateRegion(4)*3);
                candidateRegionWidth  = max(100,candidateRegionWidth);
                candidateRegionHeight = max(100,candidateRegionHeight);
                candidateRegion = [predictedBB(1)+predictedBB(3)/2-(candidateRegionWidth/2) predictedBB(2)+predictedBB(4)/2-(candidateRegionHeight/2) candidateRegionWidth candidateRegionHeight];
                
                if kalmanPredictedLoc(3) > 0 && kalmanPredictedLoc(4) > 0 && showFigure
                    rectangle('Position', kalmanPredictedLoc,'EdgeColor','k','LineWidth',3)
                end
                predictedBoundingBoxes(i,5) = 4;
            else %if no other cues are available and bird is sitting-> keep previous bounding box
                predictedBB = predictedBoundingBoxes(i-1,1:4);
                predictedBoundingBoxes(i,5) = 6;
                correct(kalmanFilter, predictedBB); 
                candidateRegion = [predictedBB(1)+predictedBB(3)/2-(candidateRegionWidth/2) predictedBB(2)+predictedBB(4)/2-(candidateRegionHeight/2) candidateRegionWidth candidateRegionHeight];
                if predictedBB(3) > 0 && predictedBB(4) > 0 && showFigure
                    rectangle('Position', predictedBB,'EdgeColor','y','LineWidth',3)
                end
            end
            
        end
        if birdIsSitting
            %text(50,130,'bird is sitting', 'Color', 'white','FontSize',20)
        end
        
        if ~bird_out_of_frame
            if predictedBB(3) * predictedBB(4) < minimumArea
                if showFigure && predictedBB(3) > 0 && predictedBB(4) > 0
                    rectangle('Position', predictedBB,'EdgeColor','r','LineWidth',6, 'LineStyle','-')
                end
                disp('too small for a bird')
                bird_lost = true;
                predictedBoundingBoxes(i,8) = 1;
            end
            
            predictedBoundingBoxes(i,1:4) = predictedBB;
            if showFigure && predictedBB(3)>0 && predictedBB(4)>0
                rectangle('Position', predictedBB,'EdgeColor','w','LineWidth',3, 'LineStyle','--')
            end
        end
        if sum(groundtruth(i)) > 0
            if predictedBoundingBoxes(i,3) > 0 && predictedBoundingBoxes(i,4) > 0 %check if width or height of predicted bbox is 0
                predictedBoundingBoxes(i,6) = bboxOverlapRatio(groundtruth(i,:), predictedBoundingBoxes(i,1:4));
            else
                predictedBoundingBoxes(i,6) = 0;
            end
            
            %re-initialize if overlap with gt == 0; re-initialize on next available gt bbox
            if predictedBoundingBoxes(i,6) == 0
                lookingForValidGT = true;
                skip = 0;
                disp(['re-init from ', num2str(i)]);
                while lookingForValidGT && (i + skip) < min(numel(frames), size(groundtruth,1))
                    skip = skip + 1;
                    
                    if sum(groundtruth(i+skip)) > 0 % valid ground truth
                        lookingForValidGT = false;
                        disp(['re-init to ', num2str(i+skip)]);
                        predictedBoundingBoxes(i+skip,1:4) = groundtruth(i+skip,:);
                        predictedBoundingBoxes(i+skip,5) = 5; %init based on gt
                        predictedBoundingBoxes(i+skip,6) = 1; %overlap
                        predictedBB = groundtruth(i+skip,:);
                        candidateRegion = [predictedBB(1)+predictedBB(3)/2-(candidateRegionWidth/2) predictedBB(2)+predictedBB(4)/2-(candidateRegionHeight/2) candidateRegionWidth candidateRegionHeight];
                        %reset kalman filter
                        if strcmp(kalmanFilterType,'ConstantAcceleration')
                            kalmanFilter = configureKalmanFilter('ConstantAcceleration',predictedBB,[1 1 1 ]*1e5, [25, 10, 1], 25);
                        elseif strcmp(kalmanFilterType,'ConstantVelocity')
                            kalmanFilter = configureKalmanFilter('ConstantVelocity',predictedBB,[1 1]*1e5, [25, 10], 25);
                        end
                        bird_out_of_frame = false;
                        skipFrames = skipFrames + skip;
                        if showFigure
                            %openInSeparateFigures = true;
                            %text(50,130,'re-init', 'Color', 'white','FontSize',20)
                            pause(4)
                        end
                    end
                end
            end
            
        end
        if showFigure
            %if(sum(groundtruth(i+skipFrames,:))>0 && sum(groundtruth(i+skipFrames-2,:))>0 && bboxOverlapRatio(groundtruth(i+skipFrames,:), groundtruth(i+skipFrames-2,:)) ~=0)
            pause(0.01)
            %end
        end
    end
end
toc
end