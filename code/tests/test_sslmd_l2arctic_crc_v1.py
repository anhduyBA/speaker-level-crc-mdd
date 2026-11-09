import unittest
import numpy as np
from vpa.sslmd_l2arctic_crc_v1 import positions,gop_priority,f1_threshold,prf,loso


def iv(kind,canonical=None):return {'kind':kind,'canonical_phone':canonical}


class L2ArcticTests(unittest.TestCase):
    def test_positions_skip_silence_and_insertions(self):
        got,why=positions([iv('silence'),iv('acceptable','F'),iv('insertion'),iv('substitution','DH'),iv('deletion','T')])
        self.assertIsNone(why);phones,labels=got
        self.assertEqual(labels,[0,1,1]);self.assertEqual(len(phones),3)
        self.assertTrue(all(0<=p<39 for p in phones))

    def test_phone_ids_match_head_order(self):
        from vpa.external_hubert import PHONEMES
        (phones,_),_=positions([iv('acceptable','AA'),iv('acceptable','ZH')])
        self.assertEqual(phones,[0,38]);self.assertEqual(PHONEMES[phones[1]+1],'ZH')

    def test_unparsed_and_ax_exclude_utterance(self):
        self.assertEqual(positions([iv('acceptable','F'),iv('unparsed')])[1],'unparsed_or_unscorable')
        self.assertEqual(positions([iv('unscorable')])[1],'unparsed_or_unscorable')
        self.assertEqual(positions([iv('acceptable','AX')])[1],'canonical_AX')

    def test_gop_priority(self):
        prob=np.array([[.1,.9],[.5,.5],[.8,.2]])
        self.assertEqual([round(x,6) for x in gop_priority(prob,[1,1],[[0,2],[2,3]])],[.3,.8])

    def test_f1_threshold_flags_best_prefix(self):
        gold=np.array([0,0,2,2,0,2]);pr=np.array([.9,.8,.7,.2,.6,.1])
        t=f1_threshold(gold,pr);flag=pr>t
        self.assertEqual(flag.tolist(),[True,True,True,False,True,False])

    def test_prf(self):
        self.assertAlmostEqual(prf(3,1,1)['f1'],.75)
        self.assertIsNone(prf(0,0,4)['f1'])

    def test_loso_crc_keeps_each_speaker_once(self):
        rng=np.random.default_rng(0);n=24*40
        speakers=np.repeat([f's{i}' for i in range(24)],40);err=rng.random(n)<.3
        gold=np.where(err,0,2);pr=np.clip(rng.normal(.5+.2*err,.15),0,1)
        per,t=loso(gold,pr,speakers,.15,'crc')
        self.assertEqual(sorted(r['speaker'] for r in per),sorted(set(speakers.tolist())));self.assertEqual(len(t),24)


if __name__=='__main__':
    unittest.main()
