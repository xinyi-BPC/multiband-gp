"""Standardized evaluation only: all existing GP scores/predictions are read-only."""
import argparse,json
from pathlib import Path
from unittest.mock import patch
from contextlib import ExitStack
from sklearn.gaussian_process import GaussianProcessRegressor
from normalized_evaluation_analysis import run_normalized_evaluation
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'normalized_evaluation')
parser.add_argument('--validate-only',action='store_true')
args=parser.parse_args()
with ExitStack() as stack:
    for name in ('MOGP_model.fit_mogp_gp','MOGP_model.fit_basic_gp','singleGP_model.fit_basic_gp'):
        stack.enter_context(patch(name,side_effect=AssertionError('GP fitting forbidden')))
    stack.enter_context(patch.object(GaussianProcessRegressor,'fit',side_effect=AssertionError('GP fitting forbidden')))
    stack.enter_context(patch.object(GaussianProcessRegressor,'predict',side_effect=AssertionError('New GP predictions forbidden')))
    result=run_normalized_evaluation(args.output,args.validate_only)
print(json.dumps(result,indent=2),flush=True)
