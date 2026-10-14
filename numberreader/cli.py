import argparse
import json
from pathlib import Path
import numpy as np
from .data import download
from .pipeline import run
from .model import predict,prediction_sets

def main():
    p=argparse.ArgumentParser(description='Handwritten digit OCR with prediction sets')
    sub=p.add_subparsers(dest='command',required=True)
    d=sub.add_parser('download');d.add_argument('--data',default='data')
    r=sub.add_parser('run');r.add_argument('--data',default='data');r.add_argument('--out',default='reports/run')
    i=sub.add_parser('predict');i.add_argument('pixels',help='JSON list of 64 counts or list of such rows')
    i.add_argument('--model',default='reports/benchmark/model.json')
    args=p.parse_args()
    if args.command=='download': result=download(args.data)
    elif args.command=='run': result=run(args.data,args.out)
    else:
        model=json.loads(Path(args.model).read_text());X=np.asarray(json.loads(Path(args.pixels).read_text()))
        if X.ndim==1:X=X[None,:]
        prob=predict(model,X);sets=prediction_sets(prob,model['cutoffs']['standard'])
        result=[{'digit':int(row.argmax()),'set':np.flatnonzero(s).tolist(),
                 'status':'SINGLE_CANDIDATE' if s.sum()==1 else 'REVIEW'} for row,s in zip(prob,sets)]
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
