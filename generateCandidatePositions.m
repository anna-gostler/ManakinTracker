function [ positions ] = generateCandidatePositions(candidateRegion, bird_out_of_frame, edge)
%edge is only used when bird_out_of_frame==true; it is the edge of the
%frame where new candidate positions are to be placed

% generate candidate positions
width = candidateRegion(3);
height = candidateRegion(4);

basicPositions =   [...
    %middle
    [candidateRegion(1)+width/3 candidateRegion(2)+height/3 width/3 height/3]
    %left upper
    [candidateRegion(1) candidateRegion(2) width/3 height/3]
    %left lower
    [candidateRegion(1) candidateRegion(2)+height*2/3 width/3 height/3]
    %left middle
    [candidateRegion(1) candidateRegion(2)+height/3 width/3 height/3]
    %middle upper
    [candidateRegion(1)+width/3 candidateRegion(2) width/3 height/3]
    %middle lower
    [candidateRegion(1)+width/3 candidateRegion(2)+height*2/3 width/3 height/3]
    %right upper
    [candidateRegion(1)+width*2/3 candidateRegion(2) width/3 height/3]
    %right middle
    [candidateRegion(1)+width*2/3 candidateRegion(2)+height/3 width/3 height/3]
    %right lower
    [candidateRegion(1)+width*2/3 candidateRegion(2)+height*2/3 width/3 height/3]
    ];

if bird_out_of_frame == 0
    
    %------------------shift sideways-----------------------------------------
    %positionsShiftedLeft = basicPositions(1:6,:);
    positionsShiftedRight = basicPositions([1,5,6],:);
    positionsShiftedRight2 = positionsShiftedRight;
    positionsShiftedRight3 = positionsShiftedRight;
    
    positionsShiftedRight(:,1) = positionsShiftedRight(:,1) + width*(1/12);
    positionsShiftedRight2(:,1) = positionsShiftedRight2(:,1) + width*(2/12);
    positionsShiftedRight3(:,1) = positionsShiftedRight3(:,1) + width*(3/12);
    
    positionsShiftedLeft = basicPositions([1,5,6],:);
    positionsShiftedLeft2 = positionsShiftedLeft;
    positionsShiftedLeft3 = positionsShiftedLeft;
    
    positionsShiftedLeft(:,1) = positionsShiftedLeft(:,1) - width*(1/12);
    positionsShiftedLeft2(:,1) = positionsShiftedLeft2(:,1) - width*(2/12);
    positionsShiftedLeft3(:,1) = positionsShiftedLeft3(:,1) - width*(3/12);
    
    %------------------shift down and up-----------------------------------------
    
    %positionsShiftedDown = basicPositions([1,2,4,5,7,8],:);
    positionsShiftedDown = basicPositions([1,4,8],:);
    positionsShiftedDown2 = positionsShiftedDown;
    positionsShiftedDown3 = positionsShiftedDown;
    
    positionsShiftedDown(:,2) = positionsShiftedDown(:,2) + height*(1/12);
    positionsShiftedDown2(:,2) = positionsShiftedDown2(:,2) + height*(2/12);
    positionsShiftedDown3(:,2) = positionsShiftedDown3(:,2) + height*(3/12);
    
    positionsShiftedUp = basicPositions([1,4,8],:);
    positionsShiftedUp2 = positionsShiftedUp;
    positionsShiftedUp3 = positionsShiftedUp;
    
    positionsShiftedUp(:,2) = positionsShiftedUp(:,2) - height*(1/12);
    positionsShiftedUp2(:,2) = positionsShiftedUp2(:,2) - height*(2/12);
    positionsShiftedUp3(:,2) = positionsShiftedUp3(:,2) - height*(3/12);
    
    positions = [basicPositions([1,4,5,6,8],:); positionsShiftedRight; positionsShiftedRight2; positionsShiftedRight3; positionsShiftedLeft; positionsShiftedLeft2; positionsShiftedLeft3; positionsShiftedDown; positionsShiftedDown2; positionsShiftedDown3; positionsShiftedUp; positionsShiftedUp2; positionsShiftedUp3];
    %positions = [basicPositions; positionsShiftedLeft;positionsShiftedLeft2; positionsShiftedLeft3; positionsShiftedDown; positionsShiftedDown2; positionsShiftedDown3];
  
    
    centerPosition = basicPositions(1,:);
    
    %shift diagonally
    for i = 1:4
        shiftedPositionLeftUp = centerPosition;
        shiftedPositionLeftUp(:,1) = shiftedPositionLeftUp(:,1)+width*(i/12);
        shiftedPositionLeftUp(:,2) = shiftedPositionLeftUp(:,2)-height*(i/12);
        
        shiftedPositionLeftDown = centerPosition;
        shiftedPositionLeftDown(:,1) = shiftedPositionLeftDown(:,1)+width*(i/12);
        shiftedPositionLeftDown(:,2) = shiftedPositionLeftDown(:,2)+height*(i/12);
        
        shiftedPositionRightUp = centerPosition;
        shiftedPositionRightUp(:,1) = shiftedPositionRightUp(:,1)-width*(i/12);
        shiftedPositionRightUp(:,2) = shiftedPositionRightUp(:,2)-height*(i/12);
        
        shiftedPositionRightDown = centerPosition;
        shiftedPositionRightDown(:,1) = shiftedPositionRightDown(:,1)-width*(i/12);
        shiftedPositionRightDown(:,2) = shiftedPositionRightDown(:,2)+height*(i/12);
        
        positions = [positions; shiftedPositionLeftUp; shiftedPositionLeftDown; shiftedPositionRightUp; shiftedPositionRightDown];
    end
elseif bird_out_of_frame == 1
    if strcmp(edge, 'top') == 1
        positions = basicPositions([2,5,7],1:4);
        for i = 1:2
            shiftedPosition = basicPositions([2,5,7],1:4);
            shiftedPosition(:,2) = shiftedPosition(:,2) + height*(i/9);
            positions = [positions; shiftedPosition];
        end
    elseif strcmp(edge,'left') == 1
        positions = basicPositions([2,3,4],:);
        for i = 1:2
            shiftedPosition = basicPositions([2,3,4],:);
            shiftedPosition(:,1) = shiftedPosition(:,1) + width*(i/9);
            positions = [positions; shiftedPosition];
        end    

    elseif strcmp(edge,'right') == 1
        positions = basicPositions([2,3,4],:);
        for i = 1:2
            shiftedPosition = basicPositions([2,3,4],:);
            shiftedPosition(:,1) = shiftedPosition(:,1) - width*(i/9);
            positions = [positions; shiftedPosition];
        end  

    end
end


