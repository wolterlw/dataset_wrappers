import random
import numpy as np
import cv2
import torch
from scipy.io import loadmat
from imageio import get_reader
import torchvision.transforms.functional as tf

from datasets.abstract import Loader, Modifier, ParameterGenerator

class RandFramesLoader(Loader):
    r"""loads random sequential {n_frames} from the
    {sample['video_file']} into {sample['imgs']}
    and stores their indices into {sample['frame_idx']}
    """

    def __init__(self, num_frames=5, stride=1):
        super(RandFramesLoader, self).__init__(
            adds=('frames','frame_idx'),
            requires=('video_file'))
        self.n_fr = num_frames
        self.stride = stride
        self.fr_needed = self.n_fr * self.stride

    def __call__(self, sample):
        sample['frames'] = []

        with get_reader(sample['video_file']) as reader:
            n_total = reader.count_frames()
            assert n_total > self.fr_needed,\
                f"need {self.fr_needed} frames, but only have {n_total}"
                        
            start_from = random.randint(0, n_total - self.fr_needed)
            frame_idx = list(
                range(start_from, start_from+self.fr_needed, self.stride))

            for i, frame in enumerate(reader):
                if i in frame_idx:
                    sample['frame'].append(frame)
            sample['frame_idx'] = frame_idx
        return sample

class NormalizeImgBatch(Modifier):
    """Expected input sample:
    {'img': torch.Tensor(bsize,C,H,W)}
    image values in range [0,255]
    """

    def __init__(self, mean, std, requires=('img'), modifies=('img')):
        super(NormalizeImgBatch, self).__init__(
            name='norm',
            requires=requires,
            modifies=modifies
        )
        self.mean = torch.as_tensor(mean, dtype=torch.float32).view(1,-1,1,1)
        self.std = torch.as_tensor(std, dtype=torch.float32).view(1,-1,1,1)
    
    def __call__(self, sample):
        new_name = self._rename(sample,'img')
        sample[new_name].div_(255).sub_(self.mean).div_(self.std)
        return sample

class LoadSURREALAnnoByIdx(Loader):
    """Loads annotations from {sample['anno_file']}
    for frames in {sample['frame_idx']}"""
    def __init__(self, adds, requires):
        super(LoadSURREALAnnoByIds, self).__init__(
            adds=('zrot','joints2d','joints3d','pose','shape'),
            requires=('anno_file', 'frame_idx')   
        )

    def __call__(self, sample):
        data = loadmat(sample['anno_file'])
        frame_idx = sample['frame_idx']

        sample['zrot'] = data['zrot'][frame_idx]
        
        sample['joints2d'] = np.transpose(
            data['joints2d'][:, :, frame_idx], (2, 1, 0)
        ).astype('int')

        sample['joints3d'] = np.transpose(
            data['joints3d'][:, :, frame_idx], (2, 1, 0)
        ).astype('float32')
        sample['pose'] = data['pose'][:, frame_idx].T
        sample['shape'] = data['shape'][:, frame_idx].T
        
        return sample


class KeypointPainter():
    """Draws limbs and spine on the original image"""

    def __init__(self, line_width=1):
        self.f_lenormat = {
            'left_arm': ([12, 14, 17, 19, 21, 23], (0, 255, 128)),
            'left_leg': ([0, 2, 5, 8, 11], (0, 255, 128)),
            'right_arm': ([12, 13, 16, 18, 20, 22], (0, 128, 255)),
            'right_leg': ([0, 1, 4, 7, 10], (0, 128, 255)),
            'spine': ([15, 12, 9, 6, 3, 0], (255, 0, 0)),
        }
        self.lw = line_width

    def __call__(self, img, keypoints):
        assert len(img.shape) == 3

        for joints, color in self.f_lenormat.values():
            for beg, end in zip(joints, joints[1:]):
                k_b = tuple(keypoints[beg])
                k_e = tuple(keypoints[end])
                cv2.line(img, k_b, k_e, color, self.lw)
        return img


class BboxPainter():
    """Draw bbox in format [[ymin, xmin], [ymax,xmax]]"""

    def __init__(self, color=(0, 255, 0)):
        assert type(color) is tuple
        self.c = color

    def __call__(self, img, bbox):
        #         ymn,xmn -> ymn,xmx -> ymx,xmx -> ymx,xmn
        bbox = bbox.ravel()
        points = bbox[[0, 1]], bbox[[0, 3]], bbox[[2, 3]], bbox[[2, 1]]
        for i in range(4):
            cv2.line(img,
                     tuple(points[i]),
                     tuple(points[(i+1) % 4]),
                     self.c, 1
                     )


class DrawAnno(Modifier):
    r"""Draws on top of images using Painter classes
    painters should be fed as {anno_fieldname: PainerObject}
    """

    def __init__(self, painters):
        super(DrawAnno, self).__init__(
            name = 'with_anno',
            requires = ('frame', *painters.keys()),
            modifies = ('img')
        )
        self.p = painters

    def __call__(self, sample):
        new_name = self._rename(sample, 'img')

        for k, v in self.p.items():
            v(sample['img'], sample[k])
        return sample


class PermuteImgChannels(Modifier):
    def __init__(self, keep_originals=True):
        super(PermuteImgChannels, self).__init__(
            requires = ('img'),
            modifies = ('img'),
            keep_originals = keep_originals
        )

    def __call__(self, sample):
        new_name = self._rename(sample, 'img')
        sample[new_name] = sample[new_name].permute(0, 3, 1, 2)
        return sample



# ================ TODO: all below =============================


class PointsToBBox():
    """computes a bounding box in
    [[y_min, x_min], [y_max, x_max]] format
    from 2d joint keypoints and applies padding
    (padding in % of width and height respectively)
    """

    def __init__(self, pad=((0, 0), (0, 0)), smooth_coef=0):
        self.pad = np.c_[pad]
        assert 0 <= smooth_coef < 1
        self.c = smooth_coef

    def __call__(self, sample):
        kps = sample['joints2D']
        bboxes = np.c_[
            [kps.min(axis=1),
             kps.max(axis=1)]
        ].transpose(1, 0, 2)
        wh = bboxes[:, 1, :] - bboxes[:, 0, :]
        bboxes[:, 0] = bboxes[:, 0] - self.pad[0] / 100 * wh
        bboxes[:, 1] = bboxes[:, 1] + self.pad[1] / 100 * wh

        if self.c > 0:
            bboxes[1:] = (1-self.c) * bboxes[1:] + self.c * bboxes[:-1]
        sample['bboxes'] = bboxes.astype('int')
        return sample


class GetCropMatrix():
    """Generates an affine transform matrix for cv2.warpAffine
    that both warps and crops the images according to sample['bbox']
    """
    def __init__(self, out_size, sequence=False):
        """
        out_size: side of the output image
        sequence: whether samples are sequences of images
        """
        self.s_out = out_size
        self.seq = sequence
        self.T = np.eye(3)

    @staticmethod
    def _bbox_to_center_side(bbox):
        """TODO: add to functional"""
        size = (bbox[1, :] - bbox[0, :]).max()
        center = (bbox[1, :] + bbox[0, :]) / 2
        return center, size

    def _comp_matrix(self, bbox):
        cnt, sz = self._bbox_to_center_side(bbox)
        tr = self.s_out / 2 - cnt
        scale = self.s_out / sz
        self.T[:2, 2:] = tr[:, None] / scale
        R = cv2.getRotationMatrix2D(tuple(cnt), 0, scale)
        M = R @ self.T
        return M

    def __call__(self, sample):
        if self.seq:
            sample['Ms'] = [
                self._comp_matrix(bbox) for
                bbox in sample['bboxes']
            ]
        else:
            sample['M'] = self._comp_matrix(sample['bbox'])
        return sample

class AffineTransformNCrop():
    def __init__(self, out_size, sequence=False):
        self.out_size = (out_size, out_size)
        self.seq = sequence

    def _warp_img(self, img, M):
        return cv2.warpAffine(img, M, self.out_size)

    @staticmethod
    def _warp_coords(crd, M):

        return crd @ M[:, :2] + M[:, 2]

    def __call__(self, sample):
        if self.seq:
            for i, M in enumerate(sample['Ms']):
                sample['imgs'][i] = self._warp_img(sample['imgs'][i], M)
                sample['bboxes'][i] = self._warp_coords(sample['bboxes'][i], M)
                if 'joints2D' in sample:
                    sample['joints2D'][i] = self._warp_coords(
                        sample['joints2D'][i], M)
        else:
            M = sample['M']
            sample['img_orig'] = sample['img']
            sample['img'] = self._warp_img(sample['img'], M)
            sample['bboxes'] = self._warp_coords(sample['bbox'], M)
            if 'joints2D' in sample:
                sample['joints2D'] = self._warp_coords(sample['joints2D'], M)
        return sample

class ToTensor():
    """converts numpy arrays into float tensors
    if there is a list of numpy arrays - stacks them and does the same"""
    @staticmethod
    def __call__(sample):
        for k, v in sample.items():
            if type(v) is np.ndarray:
                sample[k] = torch.from_numpy(v.astype('float32'))
            if isinstance(v, list) and isinstance(v[0], np.ndarray):
                sample[k] = torch.from_numpy(
                    np.stack(v, axis=0).astype('float32')
                )
        return sample

class ReprojectKeypoints():
    @staticmethod
    def __call__(sample):
        M = sample['M']
        Aff = M[:, :2]
        T = M[:, 2]
        Inv = np.linalg.inv(Aff)
        sample['joints2D_pred'] = (sample['joints2D_pred'] - T) @ Inv
        return sample


class SurrealPointsAdjust():
    """deals with the fact that SURREAL 3D joints
    are rotated, shifted from (0,0) and inverted along x axis.
    Also subsamples those points that have corresponding ones
    among SMPL predicted
    """
    def __init__(self, ):
        self.idx = [15, 12, 16, 18, 20, 17, 19, 21, 6, 0, 1, 4, 7, 2, 5, 8]
        self.R = np.rot90(np.diag([1,1,-1])).reshape(1, 3, 3)  # for batch operations

    def __call__(self, sample):
        # they always come as a batch because we grab several frames
        assert sample['joints3D'].shape[1:] == (
            24, 3) and isinstance(sample['joints3D'], np.ndarray)
        j3d = sample['joints3D'][:, self.idx, :]
        j2d = sample['joints2D'][:, self.idx, :]

        j3d -= j3d.mean(axis=1).reshape(-1, 1, 3)
        j3d = j3d @ self.R
        j3d[:, :, 0] *= -1
        sample['joints3D'] = j3d
        sample['joints2D'] = j2d
        return sample


class AddCameraTranslation():
    """Computes camera transition for perspective projection
    only uses the middle sample assuming that the camera is supposed
    to be the similar between adjacent frames"""
    def __init__(self, focal_length=5000, center=(112, 112)):
        self.f_len = np.r_[[focal_length, focal_length]]
        self.cnt = np.r_[center] if type(center != int) else np.r_[
            [center, center]]

    def _get_tr(self, j2d, j3d, weights=None):
        num_joints = len(j3d)

        # transformations
        Z = np.reshape(np.tile(j3d[:, 2], (2, 1)).T, -1)
        XY = np.reshape(j3d[:, 0:2], -1)

        O = np.tile(self.cnt, num_joints)
        F = np.tile(self.f_len, num_joints)

        # least squares
        Q = np.array([
            F * np.tile(np.array([1, 0]), num_joints),
            F * np.tile(np.array([0, 1]), num_joints),
            O - np.reshape(j2d, -1)
        ]).T
        c = (np.reshape(j2d, -1)-O)*Z - F*XY

        if not weights is None:
            raise NotImplementedError()

        # square matrix
        A = np.dot(Q.T, Q)
        b = np.dot(Q.T, c)

        # solution
        trans = np.linalg.solve(A, b)
        return trans.astype('float32')

    def __call__(self, sample):
        if sample['dataset'] != 'surreal':
            raise NotImplementedError('only SURREAL is supported so far')
        bsize = len(sample['joints2D'])
        tr = self._get_tr(sample['joints2D'][bsize//2],
                          sample['joints3D'][bsize//2])
        sample['cam_t'] = tr.reshape(1, 3)  # unsqueeze
        return sample

class RemapSample():
    """Using a key->new_key map produces a new sample
    that includes only the keys mentioned in the map,
    but with new names.
    """
    def __init__(self, map_=None):
        self._map = map_

    def __call__(self, sample):
        new_sample = {
            nk: sample[k] for k, nk in self._map
        }
        return new_sample
