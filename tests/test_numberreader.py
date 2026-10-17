import json
import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from numberreader.data import parse,partitions
from numberreader.model import validate_pixels,export,predict,fit,conformal_cutoff,prediction_sets,evaluate,number_benchmark
from numberreader.report import render

class DataTests(unittest.TestCase):
    def test_parse_preserves_counts_and_labels(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'sample';p.write_text(','.join(['16']*64+['9']))
            X,y=parse(p);self.assertEqual(X.shape,(1,64));self.assertEqual(y.tolist(),[9]);self.assertEqual(X.max(),16)

    def test_bad_source_rows_rejected(self):
        for row in [['0']*64,['17']*64+['0'],['1']*64+['10'],['0.5']*64+['1'],['nan']*64+['1']]:
            with self.subTest(row=row[:2]),tempfile.TemporaryDirectory() as d:
                p=Path(d)/'sample';p.write_text(','.join(row))
                with self.assertRaises(ValueError):parse(p)

    def test_partitions_disjoint_complete_reproducible(self):
        y=np.arange(200)%10;a=partitions(y);b=partitions(y)
        merged=np.concatenate(list(a.values()));self.assertEqual(len(set(merged)),200)
        for key in a:
            np.testing.assert_array_equal(a[key],b[key]);self.assertEqual(set(y[a[key]]),set(range(10)))
        self.assertEqual([len(a[k]) for k in ['train','validation','calibration']],[120,40,40])

    def test_pixel_validation(self):
        for x in [np.zeros((1,63)),np.full((1,64),np.nan),np.full((1,64),17),np.full((1,64),-1)]:
            with self.assertRaises(ValueError):validate_pixels(x)
        np.testing.assert_array_equal(validate_pixels(np.full((2,64),16)),np.ones((2,64)))

class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(4);cls.y=np.arange(200)%10
        cls.X=rng.uniform(0,1,(200,64))
        cls.X[np.arange(200),cls.y]+=10

    def test_logistic_json_matches_native(self):
        estimator=LogisticRegression(max_iter=1000).fit(self.X/16,self.y)
        model=json.loads(json.dumps(export(estimator,'test')))
        np.testing.assert_allclose(predict(model,self.X),estimator.predict_proba(self.X/16),atol=1e-12)

    def test_neural_json_matches_native(self):
        estimator=MLPClassifier(hidden_layer_sizes=(12,),solver='lbfgs',max_iter=500,random_state=2).fit(self.X/16,self.y)
        model=json.loads(json.dumps(export(estimator,'test')))
        np.testing.assert_allclose(predict(model,self.X),estimator.predict_proba(self.X/16),atol=1e-12)

    def test_calibration_data_cannot_affect_model_selection(self):
        parts=partitions(self.y);first,c1=fit(self.X,self.y,parts)
        x=self.X.copy();y=self.y.copy();ids=parts['calibration'];x[ids]=16;y[ids]=(y[ids]+1)%10
        second,c2=fit(x,y,parts);self.assertEqual(first,second);self.assertEqual(c1,c2)

    def test_softmax_stable_at_extreme_logits(self):
        model={'version':1,'classes':list(range(10)),'weights':[np.zeros((64,10)).tolist()],
               'biases':[[10000]+[-10000]*9]}
        p=predict(model,np.zeros((1,64)));self.assertTrue(np.isfinite(p).all());self.assertEqual(p[0,0],1)

    def test_invalid_version_rejected(self):
        with self.assertRaises(ValueError):predict({'version':2},self.X)

    def test_corrupt_models_never_return_predictions(self):
        valid={'version':1,'classes':list(range(10)),
               'weights':[np.zeros((64,10)).tolist()], 'biases':[[0]*10]}
        bad=[]
        for key,value in [('biases',[]),('weights',[]),('weights',[np.zeros((64,9)).tolist()]),
                          ('biases',[[0]]),('weights',[np.full((64,10),np.nan).tolist()]),
                          ('biases',[[float('inf')]*10])]:
            model=copy.deepcopy(valid);model[key]=value;bad.append(model)
        model=copy.deepcopy(valid);model['weights'].append(np.zeros((10,10)).tolist());bad.append(model)
        for model in bad:
            with self.subTest(model=str(model)[:80]), self.assertRaises(ValueError):
                predict(model,self.X)

class CalibrationTests(unittest.TestCase):
    def test_finite_sample_rank(self):
        scores=np.linspace(.1,.9,9);p=np.zeros((9,10));p[:,0]=scores;p[:,1]=1-scores
        c=conformal_cutoff(p,np.zeros(9,dtype=int),.2)
        self.assertEqual(c['rank'],8);self.assertAlmostEqual(c['qhat'],.8)

    def test_rank_above_sample_includes_all_labels(self):
        p=np.full((3,10),.1);c=conformal_cutoff(p,np.array([0,1,2]),.01)
        self.assertEqual(c['minimum_score'],0);self.assertTrue(prediction_sets(p,c).all())

    def test_ties_included(self):
        p=np.array([[.5,.5]+[0]*8]);sets=prediction_sets(p,{'minimum_score':.5})
        self.assertEqual(np.flatnonzero(sets[0]).tolist(),[0,1])

    def test_cautious_policy_is_superset(self):
        rng=np.random.default_rng(5);p=rng.dirichlet(np.ones(10),100);y=np.arange(100)%10
        standard=conformal_cutoff(p,y,.05);cautious=conformal_cutoff(p,y,.0125)
        self.assertLessEqual(cautious['minimum_score'],standard['minimum_score'])
        self.assertTrue(np.all(prediction_sets(p,standard)<=prediction_sets(p,cautious)))

    def test_invalid_probability_rows(self):
        for p in [np.full((3,10),.2),np.full((3,10),np.nan),np.full((3,10),-.1)]:
            with self.assertRaises(ValueError):conformal_cutoff(p,np.array([0,1,2]))

    def test_invalid_labels_and_alpha(self):
        p=np.full((3,10),.1)
        for y in [np.array([0,1,10]),np.array([0.,1.,2.])]:
            with self.assertRaises(ValueError):conformal_cutoff(p,y)
        with self.assertRaises(ValueError):conformal_cutoff(p,np.array([0,1,2]),0)

class EvaluationTests(unittest.TestCase):
    def test_empty_multi_and_single_sets_distinct(self):
        p=np.array([[.8,.2]+[0]*8,[.5,.5]+[0]*8,[.1]*10]);y=np.array([0,1,3])
        m=evaluate(y,p,{'minimum_score':.4})
        self.assertEqual(m['singleton_count'],1);self.assertEqual(m['review_count'],2)
        self.assertEqual(m['empty_sets'],1);self.assertAlmostEqual(m['set_coverage'],2/3)
        self.assertEqual(m['singleton_accuracy'],1);self.assertEqual(sum(map(sum,m['confusion'])),3)

    def test_no_singletons_has_null_accuracy(self):
        m=evaluate(np.array([0,1]),np.full((2,10),.1),{'minimum_score':.5})
        self.assertIsNone(m['singleton_accuracy']);self.assertEqual(m['review_count'],2)

    def test_confident_wrong_still_possible(self):
        m=evaluate(np.array([1]),np.array([[1]+[0]*9]),{'minimum_score':.8})
        self.assertEqual(m['singleton_count'],1);self.assertEqual(m['singleton_accuracy'],0)

    def test_number_reproducibility_and_exact_match(self):
        y=np.arange(10);p=np.eye(10);c={'minimum_score':.5}
        a=number_benchmark(y,p,c,count=20);b=number_benchmark(y,p,c,count=20)
        self.assertEqual(a,b);self.assertEqual(a['exact_match'],1);self.assertEqual(a['read_count'],20)

    def test_number_all_digits_must_be_singletons(self):
        y=np.array([0]);p=np.full((1,10),.1)
        m=number_benchmark(y,p,{'minimum_score':.05},count=5)
        self.assertEqual(m['exact_match'],1);self.assertEqual(m['read_count'],0)
        self.assertIsNone(m['read_accuracy']);self.assertEqual(m['whole_set_coverage'],1)

    def test_labels_do_not_change_frozen_sets(self):
        p=np.eye(10);c={'minimum_score':.8};before=prediction_sets(p,c).copy()
        evaluate(np.arange(10)[::-1],p,c)
        np.testing.assert_array_equal(before,prediction_sets(p,c))

class ReportTests(unittest.TestCase):
    def test_embedded_script_payload_escaped(self):
        result=render({'text':'</script><script>bad</script>'})
        self.assertNotIn('</script><script>bad',result);self.assertIn('\\u003c/script\\u003e',result)

if __name__=='__main__':unittest.main()
