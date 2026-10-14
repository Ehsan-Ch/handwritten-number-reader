from pathlib import Path
import csv
import importlib.metadata
import json
import platform
import numpy as np
from .data import load,partitions
from .model import fit,predict,conformal_cutoff,evaluate,number_benchmark,prediction_sets

def write_csv(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def run(folder,output):
    out=Path(output)
    if out.exists() and any(out.iterdir()):raise ValueError('Use an empty output folder')
    out.mkdir(parents=True,exist_ok=True)
    X,y,Xt,yt,source,audit=load(folder);parts=partitions(y)
    print('Training four candidates; selecting on validation only...',flush=True)
    model,selection=fit(X,y,parts)
    calibration=parts['calibration'];pc=predict(model,X[calibration])
    cutoffs={'standard':conformal_cutoff(pc,y[calibration],.05),
             'four_digit':conformal_cutoff(pc,y[calibration],.05/4)}
    model['cutoffs']=cutoffs
    p=predict(model,Xt)
    results={key:evaluate(yt,p,c) for key,c in cutoffs.items()}
    numbers={key:number_benchmark(yt,p,c) for key,c in cutoffs.items()}
    # One-pixel translation is deliberately severe for an 8 x 8 representation.
    shifted=np.zeros_like(Xt.reshape(-1,8,8));shifted[:,:,1:]=Xt.reshape(-1,8,8)[:,:,:-1]
    stress=evaluate(yt,predict(model,shifted.reshape(-1,64)),cutoffs['standard'])
    source['duplicate_audit']=audit
    summary={'source':source,'seed':42,'splits':{**{k:len(v) for k,v in parts.items()},'test':len(yt)},
             'selected':model['name'],'selection':selection,'cutoffs':cutoffs,'test':results,
             'numbers':{key:{k:v for k,v in n.items() if k!='indices'} for key,n in numbers.items()},
             'stress_right_shift':stress,
             'per_digit':{str(k):evaluate(yt[yt==k],p[yt==k],cutoffs['standard']) for k in range(10)},
             'environment':{'python':platform.python_version(),**{k:importlib.metadata.version(k) for k in ['numpy','scikit-learn']}},
             'limitations':['Internal validation and calibration are image splits; writer IDs are unavailable.',
                'Official test writers differ from development writers; calibration coverage need not transfer.',
                'Four-digit examples are synthetic groups of isolated cells, not document OCR.',
                'Drawing preprocessing approximates the dataset; its accuracy is not measured.']}
    sets=prediction_sets(p,cutoffs['standard'])
    rows=[{'id':f'test-{i:04d}','label':int(yt[i]),'prediction':int(p[i].argmax()),
           'top_score':float(p[i].max()),'candidates':' '.join(map(str,np.flatnonzero(sets[i]))),
           'status':'SINGLE_CANDIDATE' if sets[i].sum()==1 else 'REVIEW'} for i in range(len(yt))]
    write_csv(out/'predictions.csv',rows)
    split_rows=[{'source':'optdigits.tra','row':int(i),'split':k} for k,ids in parts.items() for i in ids]
    split_rows += [{'source':'optdigits.tes','row':i,'split':'test'} for i in range(len(yt))]
    write_csv(out/'splits.csv',split_rows)
    for name,value in [('metrics',summary),('model',model)]:
        (out/f'{name}.json').write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    (out/'sample-input.json').write_text(json.dumps(Xt[0].tolist()),encoding='utf-8')
    # Include fixed examples spanning labels; the full held-out pixel pool supports number experiments.
    examples=[]
    for k in range(10):
        ids=np.flatnonzero(yt==k)
        errors=ids[p[ids].argmax(axis=1)!=k]
        correct=ids[p[ids].argmax(axis=1)==k]
        examples.extend(errors[:3].tolist()+correct[:3].tolist())
    data={'summary':summary,'model':model,'pixels':Xt.tolist(),'labels':yt.tolist(),
          'examples':examples,'number_indices':numbers['standard']['indices'][:30],
          'checks':[{'index':i,'scores':p[i].tolist()} for i in range(6)]}
    from .report import render
    (out/'report.html').write_text(render(data),encoding='utf-8')
    return {'selected':summary['selected'],'splits':summary['splits'],'test':results,'numbers':summary['numbers']}
