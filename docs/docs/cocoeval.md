# COCOeval

Interface for evaluating detection on the Microsoft COCO dataset.

The usage for CocoEval is as follows:
```python
    cocoGt=..., cocoDt=...       # load dataset and results
    E = CocoEval(cocoGt,cocoDt); # initialize CocoEval object
    E.params.recThrs = ...;      # set parameters as desired
    E.evaluate();                # run per image evaluation
    E.accumulate();              # accumulate per image results
    E.summarize();               # display summary metrics of results
```
For example usage see evalDemo.m and https://cocodataset.org/.

The evaluation parameters are as follows (defaults in brackets):
- imgIds     - [all] N img ids to use for evaluation
- catIds     - [all] K cat ids to use for evaluation
- iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
- recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
- areaRng    - [...] A=4 object area ranges for evaluation
- maxDets    - [1 10 100] M=3 thresholds on max detections per image
- iouType    - ['segm'] set iouType to 'segm', 'bbox' or 'keypoints'
- iouType replaced the now DEPRECATED useSegm parameter.
- useCats    - [1] if true use category labels for evaluation

Note: if useCats=0 category labels are ignored as in proposal scoring.

Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.

`evaluate()`: evaluates detections on every image and every category and concats the results into the "evalImgs" with fields:
- dtIds      - [1xD] id for each of the D detections (dt)
- gtIds      - [1xG] id for each of the G ground truths (gt)
- dtMatches  - [TxD] matching gt id at each IoU or 0
- gtMatches  - [TxG] matching dt id at each IoU or 0
- dtScores   - [1xD] confidence of each dt
- gtIgnore   - [1xG] ignore flag for each gt
- dtIgnore   - [TxD] ignore flag for each dt at each IoU

`accumulate()`: accumulates the per-image, per-category evaluation results in "evalImgs" into the dictionary "eval" with fields:
- params     - parameters used for evaluation
- date       - date evaluation was performed
- counts     - [T,R,K,A,M] parameter dimensions (see above)
- precision  - [TxRxKxAxM] precision for every evaluation setting
- recall     - [TxKxAxM] max recall for every evaluation setting

Note: precision and recall==-1 for settings with no gt objects.