"""Course protocol tests with known answers, not model-generated benchmarks."""
import unittest
import numpy as np
from protocol import evaluate_depth,aggregate,course_mask,resize_prediction
from nyu_data import convert_sample,load_splits
from pathlib import Path


class ProtocolTests(unittest.TestCase):
    def test_perfect_prediction(self):
        gt=np.full((480,640),2,np.float32)
        r=evaluate_depth(gt,gt)
        self.assertEqual(r['absrel'],0);self.assertEqual(r['rmse_m'],0);self.assertEqual(r['delta1'],1)
        self.assertEqual(r['valid_pixels'],426*560)

    def test_scale_error_is_not_aligned_away(self):
        gt=np.full((480,640),2,np.float32)
        r=evaluate_depth(gt*2,gt)
        self.assertEqual(r['absrel'],1);self.assertEqual(r['rmse_m'],2);self.assertEqual(r['delta1'],0)

    def test_crop_half_open_and_gt_mask(self):
        gt=np.full((480,640),2,np.float32)
        gt[45,41]=0;gt[45,42]=np.nan;gt[45,43]=10;gt[45,44]=.001
        m=course_mask(gt)
        self.assertFalse(m[44,41]);self.assertTrue(m[470,600]);self.assertFalse(m[471,600]);self.assertFalse(m[470,601])
        self.assertEqual(int(m.sum()),426*560-4)

    def test_predictions_clipped_without_excluding_bad_values(self):
        gt=np.full((480,640),2,np.float32)
        r=evaluate_depth(np.full(gt.shape,20,np.float32),gt)
        self.assertEqual(r['absrel'],4);self.assertEqual(r['rmse_m'],8)
        self.assertEqual(r['prediction_clipped_pixels'],r['valid_pixels'])
        with self.assertRaises(ValueError): evaluate_depth(np.full(gt.shape,np.nan),gt)

    def test_delta_strict_threshold(self):
        gt=np.full((480,640),4,np.float32)
        self.assertEqual(evaluate_depth(np.full(gt.shape,5,np.float32),gt)['delta1'],0)

    def test_equal_image_average_not_pixel_weighted(self):
        rows=[{'absrel':0.,'rmse_m':0.,'delta1':1.,'valid_pixels':1},
              {'absrel':1.,'rmse_m':2.,'delta1':0.,'valid_pixels':1000}]
        self.assertEqual(aggregate(rows),{'absrel':.5,'rmse_m':1.,'delta1':.5})

    def test_matlab_layout_axis_order(self):
        rgb=np.zeros((3,640,480),np.uint8);depth=np.zeros((640,480),np.float32)
        rgb[:,123,45]=[11,22,33];depth[123,45]=7.5
        image,gt=convert_sample(rgb,depth)
        np.testing.assert_array_equal(image[45,123],[11,22,33]);self.assertEqual(gt[45,123],7.5)

    def test_resize_keeps_metric_scale(self):
        result=resize_prediction(np.full((240,320),3.0,np.float32))
        self.assertEqual(result.shape,(480,640));np.testing.assert_allclose(result,3)

    def test_official_split_is_disjoint_and_zero_based(self):
        p=Path(__file__).parent/'data/splits.mat'
        if not p.is_file(): self.skipTest('Official split not downloaded')
        s=load_splits(p)
        self.assertEqual(len(s['test']),654);self.assertEqual(len(s['train']),795)
        self.assertFalse(set(s['test'])&set(s['train']))
        self.assertEqual(set(s['test'])|set(s['train']),set(range(1449)))


if __name__=='__main__':unittest.main(verbosity=2)
