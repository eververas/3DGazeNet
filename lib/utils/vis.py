
def print_losses(epoch, i, len_train_load, speed, losses_out, 
                 writer_dict, logger, print_progress=True):
    # print progress
    if print_progress:
        msg = f"\n - Epoch: [{epoch}][{i}/{len_train_load}] - Speed: {speed:.1f} samples/s"
        logger.info(msg)
    # print losses message
    loss_string = ' '.join([f"{loss_name}: {loss.item():.4f}" for loss_name, loss in losses_out.items()])
    msg = f"   Losses Opt:  {loss_string}"
    logger.info(msg)
    # fill writer_dict
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    for loss_name, l in losses_out.items():
        writer.add_scalar('training_loss/' + str(loss_name), l.item(), global_steps)
    writer.add_scalar('training_loss/total_loss', sum(losses_out.values()), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1


def show_result(img, bboxes=None, keypoints=None,gaze=None, title=None):
    import copy
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    """Draw `result` over `img` and plot it on local Pycharm.
        This function is used for internal debugging purposes only.
    Args:
        img (str or Numpy): The image to be displayed.
        bboxes (Numpy or tuple): The bboxes to draw over `img`.
        keypoints (Numpy): The available keypoints to draw over `img`.
    Returns:
        None
    """
    if isinstance(img, str):
        # img = np.asarray(Image.open(img))
        img = cv2.imread(img, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    else:
        img = copy.deepcopy(img)
    if title is None:
        title = 'test input img with bboxes and keypoints (if available)'
    # draw bounding boxes
    if bboxes is not None:
        for j, _bboxes in enumerate(bboxes):
            left_top = (bboxes[j, 0], bboxes[j, 1])
            right_bottom = (bboxes[j, 2], bboxes[j, 3])
            cv2.rectangle(img, left_top, right_bottom, (0, 255, 0), thickness=1)
    # draw keypoints
    if keypoints is not None:
        for annot in keypoints:
            cor_x, cor_y = int(annot[0]), int(annot[1])
            cv2.circle(img, (cor_x, cor_y), 1, (255, 0, 0), 1)
    if gaze is not None:
        ry, rx = gaze[0]
        eye_center = gaze[1]
        dx = 25 * np.sin(-rx)
        dy = 25 * np.sin(-ry)
        pt2 = np.array((eye_center[0] + dx, eye_center[1] + dy)).astype(np.int32)
        cv2.arrowedLine(img, eye_center.astype(np.int32), pt2, (255, 0, 0), 2)
    # plot the result
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255)  # We expect image to be bgr and to 0-255
    plt.title(title)
    plt.tight_layout()
    plt.show(block=True)
