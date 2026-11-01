"""Replot existing Part 3 predictions; no model refits or ranking changes."""
from pathlib import Path
import argparse,json,sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from part3_auxiliary_plots import rerender_saved_cases
parser=argparse.ArgumentParser()
parser.add_argument('--smoke',action='store_true')
parser.add_argument('--near-days',type=float,default=5.)
parser.add_argument('--output',type=Path)
parser.add_argument('--primary-metric',choices=['crps','ncrps'],default='crps')
parser.add_argument('--panel-labels',action='store_true')
args=parser.parse_args()
if not args.smoke:
    assert json.loads((HERE/'common_time_reference_smoke/validation.json').read_text())['validation_passed']
result=rerender_saved_cases(args.output or HERE/('common_time_reference_smoke' if args.smoke else 'common_time_reference'),near_days=args.near_days,smoke=args.smoke,primary_metric=args.primary_metric,panel_labels=args.panel_labels)
print(json.dumps(result,indent=2),flush=True)
