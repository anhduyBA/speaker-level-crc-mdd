import unittest
import numpy as np
from vpa.sslmd_training_v2 import check_folds,fit_counts,threshold,pick_epoch,ensemble,attach_thresholds,posterior_qc


class TrainingContractTests(unittest.TestCase):
    def test_QC_tolerance_with_unchanged_alignment(self):
        old=np.array([[.1,.9],[.1,.9]])
        new=old+np.array([[9e-6,-9e-6],[0,0]])
        m={'phones':[0],'ranges':[[0,2]]}
        self.assertTrue(posterior_qc(new,old,m)['canonical_ranges_identical'])
        with self.assertRaises(ValueError):posterior_qc(old+np.array([[2e-5,-2e-5],[0,0]]),old,m)

    def test_QC_rejects_changed_alignment_even_with_tiny_error(self):
        from vpa.phone_alignment import posterior_align
        old=np.tile([.5000001,.4999999],(3,1));new=np.tile([.4999999,.5000001],(3,1))
        m={'phones':[0],'ranges':posterior_align(old,[1])['phone_frame_ranges']}
        with self.assertRaises(ValueError):posterior_qc(new,old,m)

    def test_overlap_rejected(self):
        with self.assertRaises(ValueError):
            check_folds([{'fit':list('abcdef'),'calibration':list('gh'),'evaluation':list('ha')}],list('abcdefghij'))

    def test_counts_exclude_calibration_and_held(self):
        rows=[{'speaker':'fit','eligible_frames':4},{'speaker':'fit','eligible_frames':6},
              {'speaker':'cal','eligible_frames':900},{'speaker':'held','eligible_frames':800}]
        self.assertEqual(fit_counts(rows,['fit']),{'fit':10})

    def test_first_epoch_on_ties(self):
        self.assertEqual(pick_epoch([{'calibration_pauc':.5},{'calibration_pauc':.8},{'calibration_pauc':.8}]),1)

    def test_threshold_exhaustive_against_all_possible_decisions(self):
        from itertools import product
        for y in product((0,1),repeat=4):
            if len(set(y))<2:continue
            for q in ([0,0,0,0],[-5,-4,-3,-2],[2,3,4,5],[2,1,2,3]):
                t=threshold(y,q);ng=y.count(0);nb=y.count(1)
                grid=np.unique(np.r_[np.nextafter(min(q),-np.inf),q])
                costs=[sum(a==1 and b<=g for a,b in zip(y,q))/nb+2*sum(a==0 and b>g for a,b in zip(y,q))/ng for g in grid]
                self.assertAlmostEqual(min(costs),costs[list(grid).index(t)])

    def test_equal_speaker_group_estimator(self):
        # Group loss must retain per-speaker denominators; normalizing weights
        # within each utterance would incorrectly erase them.
        losses=[2.,8.,3.];speakers=['a','a','b'];frames={'a':10,'b':3}
        objective=sum(losses[i]/(2*frames[speakers[i]]) for i in range(3))
        self.assertEqual(objective,1.)
        order=[0,2,1];g1=order[:2];g2=order[2:]
        estimates=[sum(losses[i]/(2*frames[speakers[i]]) for i in g)*3/len(g) for g in (g1,g2)]
        self.assertAlmostEqual(sum(e*len(g)/3 for e,g in zip(estimates,(g1,g2))),objective)

    def test_ensemble_rejects_missing_seed_position(self):
        row={'row_id':'u','slot':0,'speaker':'a','phone':0,'y':1,'score':2.}
        with self.assertRaises(AssertionError):ensemble([[row],[]])

    def test_held_labels_do_not_choose_threshold(self):
        cal=[{'speaker':'c','phone':0,'y':y,'score':q} for y,q in [(0,-3.),(1,2.),(0,-1.),(1,4.)]]
        a=[{'speaker':'h','phone':0,'y':0,'score':0.}]
        b=[{'speaker':'h','phone':0,'y':1,'score':0.}]
        aa,ta=attach_thresholds(cal,a);bb,tb=attach_thresholds(cal,b)
        self.assertEqual(ta,tb);self.assertEqual(aa[0]['flag'],bb[0]['flag'])

    def test_tiny_synthetic_optimizer_updates_only_head(self):
        import torch
        from vpa.sslmd_smoke import make_head,frame_map,masked_loss
        torch.manual_seed(1729);head=make_head(True,8);features=torch.randn(12,6,8)
        m=frame_map([0,1],[0,1],[[0,3],[3,6]],6)
        before={k:v.detach().clone() for k,v in head.state_dict().items()}
        opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01)
        opt.zero_grad();loss=masked_loss(head(features),m);loss.backward();opt.step()
        self.assertTrue(all(torch.isfinite(v).all() for v in head.parameters()))
        self.assertTrue(any(not torch.equal(before[k],v) for k,v in head.state_dict().items()))


if __name__=='__main__':unittest.main()
