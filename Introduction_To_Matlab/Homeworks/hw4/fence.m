function [nums , nump] = fence(lng,seg)
    nums = ceil(lng/seg);
    nump = nums+1;
end