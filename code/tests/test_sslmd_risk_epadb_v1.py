import unittest
import numpy as np
from vpa.sslmd_risk_epadb_v1 import to_crc,split_crc_feasible,empirical_threshold,loso
from vpa.public_risk import warning_rows


def rows(seed=0,people=5,n=60):
    rng=np.random.default_rng(seed);out=[]
    for p in range(people):
        for i in range(n):
            y=int(rng.random()<.3);out.append({'speaker':f's{p}','y':y,'score':float(rng.normal(1.5*y,1))})
    return out


class RiskEpadbTests(unittest.TestCase):
    def test_label_mapping(self):
        g,p,s=to_crc([{'speaker':'a','y':0,'score':0.},{'speaker':'a','y':1,'score':800.}])
        self.assertEqual(g.tolist(),[2,0]);self.assertAlmostEqual(p[0],.5);self.assertEqual(p[1],1.)

    def test_two_calibration_speakers_infeasible(self):
        self.assertFalse(split_crc_feasible(2,.2));self.assertTrue(split_crc_feasible(9,.1))

    def test_empirical_threshold_is_smallest_passing(self):
        g,p,s=to_crc(rows());t=empirical_threshold(g,p,s,.1)
        loss=lambda t:np.mean([r['crc_loss'] for r in warning_rows(g,p,s,t)])
        self.assertLessEqual(loss(t),.1)
        below=np.unique(p)[np.unique(p)<t]
        if len(below):self.assertGreater(loss(float(below.max())),.1)

    def test_crc_is_at_least_as_conservative_as_empirical(self):
        r=rows(1,people=8)
        crc=loso(r,.2,'crc');emp=loso(r,.2,'empirical')
        self.assertTrue(all(a['threshold']>=b['threshold'] for a,b in zip(crc,emp)))


if __name__=='__main__':
    unittest.main()
