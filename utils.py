import numpy as np
import cv2


def read_img(img_path, dsize, need_trasform=True, to_tensor=True, need_normalize=False):

    cvimg = cv2.imread(img_path)

    if need_trasform:
        cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)

    cvimg = cv2.resize(cvimg, dsize=dsize)

    if need_normalize:
        cvimg = cv2.normalize(cvimg, 0, 1, norm_type=cv2.NORM_MINMAX)

    if to_tensor:
        tenimg = np.array(cvimg, dtype=np.float32).transpose([2, 0, 1])
        return cvimg, tenimg

    return cvimg


if __name__ == '__main__':
    pass
