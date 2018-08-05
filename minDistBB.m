function [ distance ] = minDistBB( bb1, bb2)
% calcuates the minimum distance between two bounding boxes
x1 = double(bb1(1));
y1 = double(bb1(2));
bb1width = double(bb1(3));
bb1height = double(bb1(4));

x2 = double(bb2(1));
y2 = double(bb2(2));
bb2width = double(bb2(3));
bb2height = double(bb2(4));

bb2LeftOfbb1  = false;
bb2RightOfbb1 = false;
bb2Belowbb1 = false;
bb2Abovebb1 = false;
DEBUG = false;

if (x2 + bb2width) < x1
    bb2LeftOfbb1 = true;
    if DEBUG
        disp('bb2LeftOfbb1')
    end
end
if (x1 + bb1width) < x2
    bb2RightOfbb1 = true;
    if DEBUG
        disp('bb2RightOfbb1') 
    end
end
if y2 > (y1 + bb1height)
    bb2Belowbb1 = true;
    if DEBUG
        disp('bb2Belowbb1') 
    end
end
if (y2+bb2height) < y1 
    bb2Abovebb1 = true;
    if DEBUG
        disp('bb2Abovebb1') 
    end
end


if bb2Abovebb1 && bb2LeftOfbb1
    distance = pdist([x1+bb1width, y1+bb1height; x2, y2],'euclidean');
elseif bb2LeftOfbb1 && bb2Belowbb1
    distance = pdist([x2+bb2width, y2; x1, y1+bb1height],'euclidean');
elseif bb2Belowbb1 && bb2RightOfbb1
    distance = pdist([x1+bb1width, y1+bb1height; x2, y2],'euclidean');
elseif bb2RightOfbb1 && bb2Abovebb1
    distance = pdist([x1+bb1width, y1; x2, y2+bb2height],'euclidean');
elseif bb2LeftOfbb1
    distance = x1 - (x2 + bb2width);
    if DEBUG
        distance
    end
elseif bb2RightOfbb1
    distance = x2 - (x1+bb1width);
elseif bb2Belowbb1
    distance = y2 - (y1+bb1height);
elseif bb2Abovebb1
    distance = y1 - (y2+bb2height);
else % rectangles intersect
    distance = 0;
end

text(10,10, num2str(distance))
end

