import os
import json
import numpy as np
import matplotlib.pyplot as plt


def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''
    top1, left1, bot1, right1 = box_1
    top2, left2, bot2, right2 = box_2
    
    def get_overlap(tl1, br1, tl2, br2):
        if tl2 <= tl1 and tl1 <= br1 and br1 <= br2:
            overlap = br1 - tl1
        elif tl2 <= tl1 and tl1 <= br2 and br2 <= br1:
            overlap = br2 - tl1
        elif tl1 <= tl2 and tl2 <= br1 and br1 <= br2:
            overlap = br1 - tl2
        elif tl1 <= tl2 and tl2 <= br2 and br2 <= br1:
            overlap = br2 - tl2
        else:
            overlap = 0
        return overlap

    overlap_h = get_overlap(top1, bot1, top2, bot2)
    overlap_w = get_overlap(left1, right1, left2, right2)
    intersection = overlap_h *  overlap_w
    
    height1 = bot1 - top1
    width1 = right1 - left1
    height2 = bot2 - top2
    width2 = right2 - left2
    union = height1*width1 + height2*width2 - intersection
    
    iou = intersection / union
    
    assert (iou >= 0) and (iou <= 1.0)

    return iou


def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.) 
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives. 
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0

    '''
    BEGIN YOUR CODE
    '''
    for pred_file in preds:
        pred = preds[pred_file]
        gt = gts[pred_file]
        n = 0
        for i in range(len(gt)):
            for j in range(len(pred)):
                if pred[j][4] > conf_thr:
                    iou = compute_iou(pred[j][:4], gt[i])
                    if iou > iou_thr:
                        n += 1
        TP += n
        FP += len(pred) - n
        FN += len(gt) - n

    '''
    END YOUR CODE
    '''

    return TP, FP, FN

user_profile = os.environ['HOME']

# set a path for predictions and annotations:
preds_path = '%s/data/EE148/hw02_preds' % user_profile
gts_path = '%s/data/EE148/hw02_annotations' % user_profile

# load splits:
split_path = '%s/data/EE148/hw02_splits'% user_profile
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = False
weakened = True

'''
Load training data. 
'''

if done_tweaking:
    preds_filename = 'preds_test.json'
    gts_filename = 'annotations_test.json'
    title = 'PR Curve across IOU Thresholds on Testing Data'
    img_name = 'pr-test.jpg'
elif weakened:
    preds_filename = 'preds_train-weakened.json'
    gts_filename = 'annotations_train.json'
    title = 'PR Curve across IOU Thresholds on Training Data (weakened)'
    img_name = 'pr-train-weakened.jpg'
else:
    preds_filename = 'preds_train.json'
    gts_filename = 'annotations_train.json'
    title = 'PR Curve across IOU Thresholds on Training Data'
    img_name = 'pr-train.jpg'

with open(os.path.join(preds_path, preds_filename),'r') as f:
    preds = json.load(f)
    
with open(os.path.join(gts_path, gts_filename),'r') as f:
    gts = json.load(f)


# For a fixed IoU threshold, vary the confidence thresholds.

def get_pr_vals(iou_thr):

    thresholds = []
    for fname in preds:
        if len(preds[fname]) > 0:
            for pred in preds[fname]:
                thresholds.append(pred[4])

    confidence_thrs = np.sort(thresholds)
    #confidence_thrs = np.sort(np.array([preds_train[fname][4] for fname in preds_train if len(preds_train[fname]) == 5],dtype=float)) # using (ascending) list of confidence scores as thresholds
    tp = np.zeros(len(confidence_thrs))
    fp = np.zeros(len(confidence_thrs))
    fn = np.zeros(len(confidence_thrs))
    for i, conf_thr in enumerate(confidence_thrs):
        tp[i], fp[i], fn[i] = compute_counts(preds, gts, iou_thr=iou_thr, conf_thr=conf_thr)

    recall = [tp[i] / (tp[i]+fn[i]) for i in range(len(confidence_thrs))]

    precision = [tp[i] / (tp[i]+fp[i]) for i in range(len(confidence_thrs))]
    return recall, precision


fig = plt.figure()
ax = fig.add_subplot(111)
for iou_thr in [0.25, 0.5, 0.75]:
    recall, precision = get_pr_vals(iou_thr)
    ax.scatter(recall, precision, label='%0.2f' % iou_thr)


plt.title(title)
plt.legend()
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.savefig(img_name)
plt.show()
