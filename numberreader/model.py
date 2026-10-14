import math
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix

def validate_pixels(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 64 or not np.isfinite(X).all() or X.min() < 0 or X.max() > 16:
        raise ValueError('Expected a finite N x 64 matrix in the range 0..16')
    return X/16

def export(estimator, name):
    if isinstance(estimator, LogisticRegression):
        weights, biases = [estimator.coef_.T], [estimator.intercept_]
    else:
        weights, biases = estimator.coefs_, estimator.intercepts_
    return {'version': 1, 'name': name, 'classes': estimator.classes_.tolist(),
            'weights': [w.tolist() for w in weights], 'biases': [b.tolist() for b in biases]}

def predict(model, X):
    if model['version'] != 1 or model['classes'] != list(range(10)):
        raise ValueError('Unsupported model schema')
    z = validate_pixels(X)
    for index, (weights,bias) in enumerate(zip(model['weights'],model['biases'])):
        z = z @ np.asarray(weights) + np.asarray(bias)
        if index < len(model['weights'])-1: z = np.maximum(z,0)
    z = z-z.max(axis=1,keepdims=True)
    e = np.exp(z)
    return e/e.sum(axis=1,keepdims=True)

def fit(X,y,parts):
    x = validate_pixels(X); train, val = parts['train'], parts['validation']
    specifications = [('Logistic C=1',LogisticRegression(C=1,max_iter=2000)),
                      ('Logistic C=10',LogisticRegression(C=10,max_iter=2000)),
                      ('Neural net 32',MLPClassifier(hidden_layer_sizes=(32,),alpha=.01,max_iter=500,
                            early_stopping=True,n_iter_no_change=20,random_state=42)),
                      ('Neural net 64',MLPClassifier(hidden_layer_sizes=(64,),alpha=.01,max_iter=500,
                            early_stopping=True,n_iter_no_change=20,random_state=42))]
    fitted = []
    for name, estimator in specifications:
        estimator.fit(x[train],y[train]); p = estimator.predict_proba(x[val])
        fitted.append((accuracy_score(y[val],p.argmax(axis=1)),log_loss(y[val],p),name,estimator))
    best = max(fitted,key=lambda r:(r[0],-r[1]))
    return export(best[3],best[2]), [{'name':r[2],'validation_accuracy':float(r[0]),
             'validation_log_loss':float(r[1])} for r in fitted]

def conformal_cutoff(probabilities, labels, alpha=.05):
    p = np.asarray(probabilities); y = np.asarray(labels)
    if p.ndim != 2 or p.shape[1] != 10 or len(p) != len(y) or not len(y) or not 0 < alpha < 1:
        raise ValueError('Invalid calibration inputs')
    if not np.isfinite(p).all() or np.any(p<0) or np.any(p>1) or not np.allclose(p.sum(axis=1),1):
        raise ValueError('Invalid probability rows')
    if not np.issubdtype(y.dtype,np.integer) or y.min()<0 or y.max()>9: raise ValueError('Invalid digit labels')
    scores = 1-p[np.arange(len(y)),y]
    rank = math.ceil((len(y)+1)*(1-alpha))
    # A rank beyond the sample uses the maximal bounded score: include all classes.
    q = 1.0 if rank>len(y) else float(np.sort(scores)[rank-1])
    return {'alpha':alpha,'n':len(y),'rank':rank,'qhat':q,'minimum_score':1-q}

def prediction_sets(p,cutoff):
    return np.asarray(p) >= cutoff['minimum_score']

def evaluate(y,p,cutoff):
    y=np.asarray(y);p=np.asarray(p);sets=prediction_sets(p,cutoff);sizes=sets.sum(axis=1)
    singleton=sizes==1; top=p.argmax(axis=1); correct=top==y
    return {'n':len(y),'accuracy':float(correct.mean()),'correct':int(correct.sum()),
            'set_coverage':float(sets[np.arange(len(y)),y].mean()),
            'singleton_count':int(singleton.sum()),'review_count':int((~singleton).sum()),
            'empty_sets':int((sizes==0).sum()),'mean_set_size':float(sizes.mean()),
            'singleton_accuracy':float(correct[singleton].mean()) if singleton.any() else None,
            'confusion':confusion_matrix(y,top,labels=range(10)).tolist()}

def number_benchmark(y,p,cutoff,count=500,seed=71):
    rng=np.random.default_rng(seed);idx=rng.integers(0,len(y),(count,4));sets=prediction_sets(p,cutoff)
    correct=(p.argmax(axis=1)[idx]==np.asarray(y)[idx]).all(axis=1)
    read=(sets.sum(axis=1)[idx]==1).all(axis=1)
    covered=sets[np.arange(len(y)),y][idx].all(axis=1)
    return {'count':count,'exact_match':float(correct.mean()),'whole_set_coverage':float(covered.mean()),
            'read_count':int(read.sum()),'review_count':int((~read).sum()),
            'read_accuracy':float(correct[read].mean()) if read.any() else None,
            'indices':idx.tolist()}
