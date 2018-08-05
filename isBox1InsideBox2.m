function [ box2ContainsBox1 ] = isBox1InsideBox2(box1, box2, tolerance)
%box 1 and box2 have to be bounding boxes
%box 1 is allowed to be 'tolerance' pixels outside of box 2 in every direction 

%if box1(1) >= box2(1) && box1(2) >= box2(2) && ( box1(1) + box1(3) <= box2(1) + box2(3) ) && ( box1(2) + box1(4) <= box2(2) + box2(4) )
if (box2(1) - box1(1)) <= tolerance && ...
   (box2(2) - box1(2)) <= tolerance && ...
   ((box1(1) + box1(3)) - (box2(1) + box2(3))) <= tolerance && ...
   ((box1(2) + box1(4)) - (box2(2) + box2(4))) <= tolerance

    box2ContainsBox1 = true;
else
    box2ContainsBox1 = false;
end
    

end
