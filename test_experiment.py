import unittest
import numpy as np
from experiment import collapse, design_cache, gains, normalize, choose, evaluate


class DesignTests(unittest.TestCase):
    def setUp(self):
        self.mu = np.array([[0,0,5],[0,4,5],[10,0,5],[10,4,5]],float)
        self.p = np.full((1,4),.25)
        self.cache = design_cache(self.mu,np.array([1,2,1]),64)

    def test_query_relevance_and_stop(self):
        z1 = np.array([1,1,-.2,-.8])
        z2 = np.array([-.2,-.8,-.2,-.8])
        for q,z,action in [(0,z1,0),(1,z2,1),(2,np.ones(4),-1)]:
            actual=choose(self.p,z,self.cache,'evsi',q,0,.005,np.array([.5]))
            self.assertEqual(actual[0],action)
        self.assertLess(abs(gains(self.p,z1,self.cache,'evsi')[0,2]),1e-12)

    def test_plugin_miss_overconfidence(self):
        plug=collapse(self.p,0)
        np.testing.assert_allclose(plug,[[.5,.5,0,0]])
        self.assertEqual(choose(plug,np.array([1,1,-.2,-.8]),self.cache,'evsi',0,0,.005,np.array([.5]))[0],-1)

    def test_quadrature_convergence(self):
        z=np.array([1,1,-.2,-.8])
        low=gains(self.p,z,design_cache(self.mu,np.array([1,2,1]),32),'evsi')
        high=gains(self.p,z,self.cache,'evsi')
        np.testing.assert_allclose(low,high,atol=2e-5)

    def test_refit_can_recover_geometry(self):
        full=normalize(np.log(self.p)-.5*((10-self.mu[:,0])/1)**2)
        p=collapse(full,0)
        self.assertAlmostEqual(p[0,2]+p[0,3],1)

    def test_proper_scores_degenerate_correct(self):
        z=np.array([1,1,-.2,-.8]); truth=np.arange(4)
        result=evaluate(np.eye(4),z,np.array([0,0,1,1]),np.arange(24).reshape(4,2,3),truth)
        for key in ['mse','brier','crps','trajectory_mse','energy','width90']:
            np.testing.assert_allclose(result[key],0,atol=1e-12)
        np.testing.assert_allclose(result['coverage90'],1)


if __name__=='__main__':
    unittest.main()
