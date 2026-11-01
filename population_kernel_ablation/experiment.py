"""Matched population fitting and audit; all sampling comes from saved CSV indices."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import warnings

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.gaussian_process import GaussianProcessRegressor
from MOGP_model import fit_mogp_gp, DEFAULT_BAND_TO_WAVELENGTH
from singleGP_model import fit_basic_gp
from evaluation_metrics import evaluate_heldout_metrics, evaluate_mogp_heldout_metrics, gaussian_crps
from data_processing import _extract_valid_band_observations
from population_kernel_ablation.saved_splits import load_sources, reconstruct_saved_case, validate_nested_saved_rows
from band_amplitude_ablation.kernel import BandAmplitudeMatern, initialize_amplitudes
from part3_analysis import protection_manifest

MODELS=('A_single','B_independent','C_shared_original_bounds','D_shared_matched_bounds','E_band_amplitudes')
LABELS=dict(zip(MODELS,['Single-band','Independent-band','Shared · original bounds','Shared · matched bounds','Band amplitudes']))
COLORS=dict(zip(MODELS,['#555555','#4477aa','#cc6677','#ddaa33','#228833']))
HIST_MAP=dict(zip(('single_band_gp','mogp_independent_band_control','mogp_real_wavelength'),MODELS[:3]))
CONFIG=dict(target_band='i',auxiliary_bands=['u','g','r','z'],aux_train_fraction=.5,
    target_train_fractions=[.2,.8],seeds=[0,1,2],n_restarts_optimizer=5,time_length_scale=.3,
    time_length_scale_bounds=[.05,10.],wavelength_length_scale_A=6000.,sigma_bounds=[.01,100.],
    original_C_bounds=[.01,100.],matched_C_bounds=[.0001,10000.],yerr_scale=1.,noise_floor=0.,jitter=1e-8,
    optimizer='fmin_l_bfgs_b (unmodified sklearn defaults)',optimizer_objective='training log marginal likelihood only',
    tie_atol=1e-8,tie_rtol=1e-6,bootstrap_resamples=2000,bootstrap_seed=10931,
    catastrophic_thresholds=dict(ncrps=1.,nrmse=1.),object_seed_aggregation='arithmetic mean of all three seeds',
    primary_matched_subset='all five models successful, with all three seeds for object-level summaries',
    initialization='same MAD/RMS strategy as validated diagnostic; shared C = pooled training robust sigma squared',
    initialization_projection='if initial estimate outside bounds, project to nearest bound interior (1.001 factor); record every projection; parameter bounds unchanged',
    score_normalization='max(P95(abs(all valid complete object raw flux)),1e-8), evaluation only',
    variance_definition='sigma is SD; C is variance; D and E marginal variance bounds are [1e-4,1e4]')


def dump(path,value):
    def encode(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value,indent=2,default=encode,allow_nan=True)+'\n')


def manifest():
    before=protection_manifest()
    for directory in ('band_amplitude_ablation',):
        for path in (ROOT/directory).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                before[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return before


def project(value,bounds):
    chosen=float(min(max(value,bounds[0]*1.001),bounds[1]/1.001))
    return chosen,not np.isclose(chosen,value,rtol=1e-14,atol=0)


def pooled_initialization(data,bounds):
    synthetic=dict(band=np.repeat('pooled',len(data['y'])),y=data['y'],yerr=data['yerr'])
    values,details=initialize_amplitudes(synthetic,('pooled',),(np.finfo(float).tiny,np.inf))
    estimate=values[0]**2
    chosen,projected=project(estimate,bounds)
    return chosen,[dict(parameter='C',raw_estimate=estimate,initial_value=chosen,projected=projected,method=details[0]['method'])]


def new_kernel_and_fit(model,data,seed, historical=False):
    settings=dict(time_length_scale=.3,time_length_scale_bounds=(.05,10.),constant_value=1.,
        constant_value_bounds=(.01,100.),yerr_scale=1.,noise_floor=0.,jitter=1e-8,
        n_restarts_optimizer=0 if historical else 5,random_state=int(seed))
    initial=[]
    if model=='E_band_amplitudes':
        bands=tuple(sorted(set(data['band'])))
        estimates,detail=initialize_amplitudes(data,bands,(np.finfo(float).tiny,np.inf))
        values=[]
        for band,value,info in zip(bands,estimates,detail):
            chosen,projected=project(value,(.01,100.));values.append(chosen)
            initial.append(dict(parameter=f'sigma_{band}',raw_estimate=value,initial_value=chosen,projected=projected,method=info['method']))
        k=BandAmplitudeMatern(bands=bands,wavelengths=tuple(DEFAULT_BAND_TO_WAVELENGTH[b] for b in bands),sigma=tuple(values))
        gp=GaussianProcessRegressor(kernel=k,alpha=data['yerr']**2+1e-8,normalize_y=False,
            n_restarts_optimizer=5,random_state=int(seed))
        gp.fit(data['X'],data['y'])
    else:
        bounds=(.0001,10000.) if model=='D_shared_matched_bounds' else (.01,100.)
        settings['constant_value_bounds']=bounds
        if not historical:
            settings['constant_value'],initial=pooled_initialization(data,bounds)
        if model=='A_single':
            settings['length_scale']=settings.pop('time_length_scale')
            settings['length_scale_bounds']=settings.pop('time_length_scale_bounds')
            gp=fit_basic_gp(data,**settings)
        else:
            settings['wavelength_length_scale']=1e-5 if model=='B_independent' else 6000.
            gp=fit_mogp_gp(data,**settings)
    return gp,initial


def params(gp,model):
    kernel=gp.kernel_;rows=[];index=0
    for h in kernel.hyperparameters:
        if h.fixed:continue
        for j in range(h.n_elements):
            name=h.name
            if model=='E_band_amplitudes' and name=='sigma':name=f'sigma_{kernel.bands[j]}'
            theta=kernel.theta[index];lo,hi=kernel.bounds[index];index+=1
            rows.append(dict(model=model,parameter=name,value=float(np.exp(theta)),lower=float(np.exp(lo)),upper=float(np.exp(hi)),
                at_lower=bool(np.isclose(theta,lo,atol=1e-5,rtol=0)),at_upper=bool(np.isclose(theta,hi,atol=1e-5,rtol=0))))
    return rows


def evaluate(gp,model,data,scale):
    single=model=='A_single';train=data['single_train' if single else 'multi_train'];test=data['single_test' if single else 'multi_test']
    evaluator=evaluate_heldout_metrics if single else evaluate_mogp_heldout_metrics
    result=evaluator(gp,test,train_data=train,object_flux_scale=scale)
    # Check formulas and raw/normalized invariance independently of aggregate utility.
    mean,latent=gp.predict(test['X'],return_std=True)
    predictive_std=np.sqrt(latent**2+test['yerr']**2)
    raw_mean=mean*train['flux_scale'];raw_std=predictive_std*train['flux_scale']
    raw_crps=gaussian_crps(test['y_raw'],raw_mean,raw_std)
    np.testing.assert_allclose(raw_crps.mean(),result['mean_crps'],rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(result['ncrps'],raw_crps.mean()/scale,rtol=1e-12)
    np.testing.assert_allclose(result['nrmse'],np.sqrt(np.mean((test['y_raw']-raw_mean)**2))/scale,rtol=1e-12)
    z=(test['y_raw']-raw_mean)/raw_std
    np.testing.assert_allclose(z,(test['y']-mean)/predictive_std,rtol=1e-9,atol=1e-9)
    row=dict(raw_crps=result['mean_crps'],ncrps=result['ncrps'],nrmse=result['nrmse'],nlpd=result['mean_nlpd'],
        ks_pit=result['ks_pit_object'],coverage_1sigma=result['coverage']['coverage_1sigma'],coverage_2sigma=result['coverage']['coverage_2sigma'],
        standardized_residual_mean=float(z.mean()),standardized_residual_std=float(z.std(ddof=0)),
        rmse=result['rmse'],n_evaluation=len(z),object_flux_scale=scale)
    row.update(log_marginal_likelihood=float(gp.log_marginal_likelihood_value_))
    k=gp.kernel_
    temporal=k.time_length_scale if model=='E_band_amplitudes' else (k.k2.length_scale if single else k.k1.k2.length_scale)
    row.update(time_length_scale=float(temporal),time_length_scale_days=float(temporal*train['t_scale']))
    if model=='E_band_amplitudes':
        amplitudes=dict(zip(k.bands,np.atleast_1d(k.sigma)))
        row.update({f'sigma_{b}':float(v) for b,v in amplitudes.items()})
        row.update({f'ratio_sigma_i_{b}':float(amplitudes['i']/v) for b,v in amplitudes.items() if b!='i'})
    else:row['shared_covariance_amplitude']=float(k.k1.constant_value if single else k.k1.k1.constant_value)
    return row


def choose_pilot(rows):
    h=rows.copy();h['model']=h.model.map(HIST_MAP)
    means=h.groupby(['object_id','target_train_fraction','model']).ncrps.mean().unstack('model')
    low=means.xs(.2,level='target_train_fraction').copy();low['gain']=low.A_single-low.C_shared_original_bounds
    cases=h[h.target_train_fraction==.2].pivot(index=['object_id','random_state'],columns='model',values='ncrps')
    votes=(cases.A_single>cases.C_shared_original_bounds).groupby('object_id').sum()
    count=h[(h.model=='A_single')&(h.target_train_fraction==.2)].groupby('object_id').n_train_target_band.min()
    aux=h[(h.model=='C_shared_original_bounds')&(h.target_train_fraction==.2)].groupby('object_id').n_train_other_bands.min()
    threshold=low.C_shared_original_bounds.quantile(.95)
    eligible=low[(low.gain>0)&(votes>=2)&(count>=4)&(aux>=20)&(low.C_shared_original_bounds<=threshold)].copy()
    assert len(eligible)>0
    median=float(eligible.gain.median())
    eligible['distance']=(eligible.gain-median).abs()
    representative=str(eligible.sort_values(['distance','gain'],kind='stable').index[0])
    candidate=cases.loc[representative].copy();candidate['gain']=candidate.A_single-candidate.C_shared_original_bounds
    representative_seed=int((candidate.gain-candidate.gain.median()).abs().sort_values(kind='stable').index[0])
    selected=[];reasons={}
    def add(oid,reason):
        oid=str(oid)
        if oid not in selected:selected.append(oid);reasons[oid]=reason
    add('9858227','predeclared extreme failure')
    add(representative,'representative historical success closest to median positive eligible object gain')
    for oid in low.sort_values('gain',ascending=False,kind='stable').index:
        if len(selected)>=4:break
        add(oid,'strong historical success')
    for oid in low.sort_values('gain',ascending=True,kind='stable').index:
        if len(selected)>=7:break
        add(oid,'strong historical failure')
    for oid in (low.gain-low.gain.median()).abs().sort_values(kind='stable').index:
        if len(selected)>=10:break
        add(oid,'ordinary historical gain near cohort median')
    return dict(object_ids=selected,reasons=reasons,representative_object_id=representative,
        representative_seed=representative_seed,representative_fraction=.2,eligible_success_count=len(eligible),
        selection_uses='historical nCRPS only, plus historical training counts; no new model scores',
        representative_rule='at least 2/3 positive historical seeds, minimum four target and twenty auxiliary training rows, C score <= historical P95; closest to median positive eligible object-mean gain; seed closest to within-object median historical gain',
        eligible_median_gain=median)


def capture_curves(example,data,models,selection):
    t,_,_=_extract_valid_band_observations(example,'i')
    grid=np.linspace(t.min(),t.max(),1600)
    curves={}
    for model,gp in models.items():
        tr=data['single_train' if model=='A_single' else 'multi_train']
        normalized=(grid-tr['alignment_peak_time'])/tr['t_scale']
        X=normalized[:,None] if model=='A_single' else np.column_stack([normalized,np.full(len(grid),DEFAULT_BAND_TO_WAVELENGTH['i'])])
        mean,std=gp.predict(X,return_std=True)
        curves[model]=dict(mean=mean*tr['flux_scale'],latent_std=std*tr['flux_scale'])
    observations={}
    for band in ('u','g','r','i','z'):
        times,y,err=_extract_valid_band_observations(example,band)
        observations[band]=dict(mjd=times,flux=y,error=err)
    tr=data['multi_train'];te=data['multi_test']
    return dict(grid_mjd=grid,curves=curves,observations=observations,
        training_band=tr['band'],training_mjd=tr['X'][:,0]*tr['t_scale']+tr['alignment_peak_time'],
        training_flux=tr['y_raw'],training_error=tr['yerr_raw'],
        evaluation_mjd=te['X'][:,0]*tr['t_scale']+tr['alignment_peak_time'],evaluation_flux=te['y_raw'],evaluation_error=te['yerr_raw'])


def fit_case(example,saved,output,selection,pilot):
    first=saved.iloc[0];oid=str(first.object_id);seed=int(first.random_state);fraction=float(first.target_train_fraction)
    filename=f'{oid}_s{seed}_f{fraction:.1f}.json';path=output/'checkpoints'/filename
    if path.exists():return json.loads(path.read_text())
    data=reconstruct_saved_case(example,saved);scores=[];diagnostics=[];parameters=[];initializations=[];historical_checks=[];models={}
    # Pilot zero-restart reproduction with the original initial C=1; new fits below use robust initialization.
    if pilot:
        for old_model,model in HIST_MAP.items():
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                gp,_=new_kernel_and_fit(model,data['single_train' if model=='A_single' else 'multi_train'],seed,historical=True)
            score=evaluate(gp,model,data,data['object_flux_scale'])
            old=saved[saved.model==old_model].iloc[0]
            for metric,new_name in [('crps','raw_crps'),('ncrps','ncrps'),('nrmse','nrmse')]:
                np.testing.assert_allclose(score[new_name],old[metric],rtol=1e-6,atol=1e-8,err_msg=f'historical {oid} {model} {metric}')
                historical_checks.append(dict(model=model,metric=metric,saved=float(old[metric]),reproduced=score[new_name],absolute_difference=abs(score[new_name]-old[metric])))
    for model in MODELS:
        metadata=dict(object_id=oid,object_class=first['class'],seed=seed,target_fraction=fraction,model=model)
        status='success';error='';started=time.monotonic();recorded=[]
        try:
            tr=data['single_train' if model=='A_single' else 'multi_train']
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter('always')
                gp,initial=new_kernel_and_fit(model,tr,seed)
                score=evaluate(gp,model,data,data['object_flux_scale'])
                if model=='E_band_amplitudes':
                    K=gp.kernel_(tr['X']);np.testing.assert_allclose(K,K.T,atol=1e-10)
                    minimum=float(np.linalg.eigvalsh(K).min())
                    assert minimum>=-1e-7*max(1,float(np.max(np.diag(K))))
                    np.testing.assert_allclose(gp.kernel_.diag(tr['X']),np.diag(K))
                    score['covariance_min_eigenvalue']=minimum
            models[model]=gp
            score.update(metadata,fit_status='success',n_target_train=len(data['single_train']['y']),n_aux_train=len(data['multi_train']['y'])-len(data['single_train']['y']))
            scores.append(score)
            parameters.extend([{**metadata,**p} for p in params(gp,model)])
            initializations.extend([{**metadata,**p} for p in initial])
        except Exception as exc:
            status='failed';error=f'{type(exc).__name__}: {exc}'
            scores.append({**metadata,'fit_status':status,'error':error})
        diagnostics.append({**metadata,'fit_status':status,'error':error,'fit_runtime_seconds':time.monotonic()-started,
            'warnings':[dict(category=w.category.__name__,message=str(w.message)) for w in recorded]})
    content=dict(scores=scores,diagnostics=diagnostics,parameters=parameters,initializations=initializations,historical_checks=historical_checks,
        split_audit=dict(target_train_indices=first.target_train_indices,heldout_indices=first.heldout_indices,
            auxiliary_indices=first.aux_selected_indices_by_band,training_row_ids=data['multi_train']['row_id'],
            flux_scale=data['single_train']['flux_scale'],time_scale=data['single_train']['t_scale'],object_flux_scale=data['object_flux_scale']))
    if oid==selection['representative_object_id'] and seed==selection['representative_seed'] and fraction==.2:
        dump(output/'representative_success_curves.json',capture_curves(example,data,models,selection))
    if oid=='9858227' and seed==1 and fraction==.2:
        dump(output/'failure_all_aux_curves.json',capture_curves(example,data,models,selection))
    dump(path,content)
    return content


def prepare(output):
    examples,rows=load_sources()
    if not output.exists():
        output.mkdir();(output/'checkpoints').mkdir();(output/'figures').mkdir()
        dump(output/'preservation_manifest.json',manifest())
        dump(output/'configuration.json',{**CONFIG,'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
            'sklearn':sklearn.__version__,'pandas':pd.__version__,'wavelength_mapping_nm':{b:DEFAULT_BAND_TO_WAVELENGTH[b] for b in ('u','g','r','i','z')}})
        dump(output/'pilot_selection.json',choose_pilot(rows))
        scalar=['object_id','class','random_state','target_train_fraction','model','crps','ncrps','nrmse','nlpd','ks_pit','coverage_1sigma','coverage_2sigma',
            'z_score_mean','z_score_std','fit_n_restarts_optimizer','fit_jitter','fit_yerr_scale','fit_noise_floor']
        rows[scalar].to_csv(output/'historical_scores.csv',index=False)
        audit=rows.groupby('model')[['fit_n_restarts_optimizer','fit_yerr_scale','fit_noise_floor','fit_jitter']].agg(['min','max'])
        audit.to_csv(output/'historical_configuration_audit.csv')
    stored=json.loads((output/'configuration.json').read_text())
    assert all(stored[k]==v for k,v in CONFIG.items()),'Configuration changed after pilot'
    assert json.loads((output/'preservation_manifest.json').read_text())==manifest()
    return examples,rows,json.loads((output/'pilot_selection.json').read_text())


def run(output,stage,workers=1):
    examples,rows,selection=prepare(output)
    assert validate_nested_saved_rows(rows)==900
    if stage=='full':
        pilot=json.loads((output/'pilot_validation.json').read_text());assert pilot['validation_passed']
    subset=rows[rows.object_id.isin(selection['object_ids'])] if stage=='pilot' else rows
    grouped=list(subset.groupby(['object_id','random_state','target_train_fraction'],sort=True))
    started=time.monotonic();failures=0;checks=0
    if workers == 1:
        iterator=(fit_case(examples[str(key[0])],saved,output,selection,stage=='pilot') for key,saved in grouped)
        pool=None
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        pool=ProcessPoolExecutor(max_workers=workers)
        futures=[pool.submit(fit_case,examples[str(key[0])],saved,output,selection,False) for key,saved in grouped]
        iterator=(future.result() for future in as_completed(futures))
    for index,record in enumerate(iterator):
        failures+=sum(r['fit_status']!='success' for r in record['scores'])
        checks+=len(record['historical_checks'])
        if index%10==0 or index==len(grouped)-1:
            print(f'{stage}: {index+1}/{len(grouped)} cases, {time.monotonic()-started:.1f}s, failed fits {failures}',flush=True)
    if pool is not None:pool.shutdown()
    dump(output/f'{stage}_execution.json',dict(workers=workers,BLAS_threads_per_worker=1))
    assert json.loads((output/'preservation_manifest.json').read_text())==manifest()
    validation=dict(validation_passed=failures==0,n_cases=len(grouped),n_model_attempts=len(grouped)*5,failed_fits=failures,
        historical_metric_reproduction_checks=checks,elapsed_seconds=time.monotonic()-started,original_files_unchanged=True,
        sampling_permutations_generated=0,split_indices_verified=True)
    dump(output/f'{stage}_validation.json',validation)
    if stage=='pilot':assert validation['validation_passed'], 'Inspect recorded pilot failures before full fitting'
    if stage=='full':
        from population_kernel_ablation.analysis import analyze
        analyze(output)
    print(json.dumps(validation,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['pilot','full'],required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'band_amplitude_population_ablation');parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args();run(args.output,args.stage,args.workers)
