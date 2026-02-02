## What I did
Analyzed a side-view video of a cricket match using pose estimation to analyze the player’s movement.

## How it works
- Employed MediaPipe Pose to identify keypoints
- Zeroed in on hip, knee, and ankle keypoints
- Determined the knee angle for each frame
- Monitored changes in the knee angle over time

## Metrics
- Knee angle over time
- Range of motion (max - min angle)
- Movement stability (variance)

## Observations
- Fast motion introduces a slight jitter effect
- Missing keypoints in some frames due to occlusion

## Output
- Skeleton overlay video
- Knee angle statistics
