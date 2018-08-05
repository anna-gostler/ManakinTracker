# ManakinTracker

Male golden-collared manakins are tropical birds that perform an elaborate courtship display
which determines their mating success. Biologists recorded the birds’ displays in the
jungle with high-speed cameras. To analyze what constitutes a good courtship performance
the biologists use the bird’s trajectory, which they currently obtain by manually annotating
the videos frame by frame. Automatically tracking the bird can save a lot of time.
The videos of the courtship displays are challenging for a tracker: the bird is susceptible
to motion blur, quickly changes its appearance and often leaves the frame. The cluttered
background contains elements that visually resemble or occlude the bird. We present an
online visual tracking algorithm, which combines a Mixture of Gaussians model to detect
moving objects, a Convolutional Neural Network trained to recognize the male goldencollared
manakin, and a Kalman Filter as a motion model. Our tracker achieves better
accuracy and robustness on a dataset of videos of courtship displays than state-of-the-art
trackers.

## Example Output

This figure shows the output of the tracker (bounding box around the bird).

![Example of ManakinTracker](https://github.com/anna-gostler/ManakinTracker/blob/master/output.gif)


## Getting Started

Download CNN 'CNNtrainedOnFirstHalf' from https://www.dropbox.com/s/x0fb4rs5a8osc4h/CNNtrainedOnFirstHalf.mat?dl=0
and put in the same folder as runTracker

Run the file runTracker to start ManakinTracker on the demo video stored in the folder 'video'.
 
When running the tracker on a Windows machine change path seperators '/' to '\\' in the files runTracker and ManakinTracker.
