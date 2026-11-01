"""Refresh all four-model figures from saved results, without optimization."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scipy.linalg import cholesky,cho_solve,solve_triangular
from sklearn.gaussian_process.kernels import Matern
from band_amplitude_ablation.kernel import BandAmplitudeMatern
from data_processing import _extract_valid_band_observations
from MOGP_model import DEFAULT_BAND_TO_WAVELENGTH
from population_kernel_ablation import analysis
from population_kernel_ablation.experiment import MODELS,LABELS,COLORS
from population_kernel_ablation.saved_splits import load_sources,reconstruct_saved_case


def exact_failure_payload(output):
    """Restore the existing exact i+z posterior using SAVED optimized parameters.

    The older exact-control plot had no dense matched-bound cache. Cholesky
    conditioning uses its saved training rows, noise, and fitted parameters;
    test predictions must reproduce the existing CSV before any plot is saved.
    No GP fit, optimizer, new sampling or parameter selection is called.
    """
    examples,saved=load_sources();example=examples['9858227']
    rows=saved[(saved.object_id=='9858227')&(saved.random_state==1)&(saved.target_train_fraction==.2)]
    data=reconstruct_saved_case(example,rows)
    tr=data['multi_train'];mask=np.isin(tr['band'],['i','z']);n=len(mask)
    data['multi_train']={k:np.asarray(v)[mask] if isinstance(v,np.ndarray) and len(v)==n else v for k,v in tr.items()}
    tr=data['multi_train'];te=data['multi_test']
    prior=pd.read_csv(ROOT/'band_amplitude_ablation/results/training_observations.csv')
    np.testing.assert_array_equal(tr['row_id'],prior.row_id)
    parameters=pd.read_csv(output/'failure_9858227_parameters.csv',float_precision='round_trip')
    saved_predictions=pd.read_csv(output/'failure_9858227_test_point_diagnostics.csv',float_precision='round_trip')
    tz,yz,_=_extract_valid_band_observations(example,'z');peak=float(tz[np.argmax(yz)])
    grid=np.unique(np.r_[np.linspace(peak-60,peak+60,1300),peak])
    curves={};checks=[]
    for model in analysis.DISPLAY_MODELS:
        values=parameters[parameters.model==model].set_index('parameter').value.to_dict()
        train=data['single_train'] if model=='A_single' else tr
        test=data['single_test'] if model=='A_single' else te
        if model=='E_band_amplitudes':
            kernel=BandAmplitudeMatern(bands=('i','z'),wavelengths=tuple(DEFAULT_BAND_TO_WAVELENGTH[b] for b in ('i','z')),
                sigma=tuple(values[f'sigma_{b}'] for b in ('i','z')),time_length_scale=values['time_length_scale'])
            covariance=kernel;diagonal=kernel.diag
        else:
            single=model=='A_single'
            amplitude=values['k1__constant_value' if single else 'k1__k1__constant_value']
            temporal=Matern(values['k2__length_scale' if single else 'k1__k2__length_scale'],nu=1.5)
            wave=Matern(1e-5 if model=='B_independent' else 6000.,nu=1.5)
            def covariance(X,Y,amplitude=amplitude,temporal=temporal,wave=wave,single=single):
                result=amplitude*temporal(X[:,[0]],Y[:,[0]])
                if not single:result*=wave(X[:,[1]]*10.,Y[:,[1]]*10.)
                return result
            diagonal=lambda X,amplitude=amplitude:np.full(len(X),amplitude)
        X=np.asarray(train['X']);K=covariance(X,X)
        K[np.diag_indices_from(K)]+=np.asarray(train['yerr'])**2+1e-8
        L=cholesky(K,lower=True);alpha=cho_solve((L,True),train['y'])
        def predict(Z):
            cross=covariance(Z,X);v=solve_triangular(L,cross.T,lower=True)
            variance=diagonal(Z)-np.einsum('ij,ij->j',v,v)
            assert variance.min()>-1e-8
            return cross@alpha,np.sqrt(np.maximum(variance,0.))
        mean,std=predict(test['X']);scale=float(train['flux_scale'])
        raw_mean=mean*scale;raw_std=np.sqrt(std**2+test['yerr']**2)*scale
        saved_model=saved_predictions[saved_predictions.model==model]
        np.testing.assert_allclose(raw_mean,saved_model.predicted_mean,rtol=1e-8,atol=1e-7)
        np.testing.assert_allclose(raw_std,saved_model.predicted_std,rtol=1e-8,atol=1e-7)
        checks.append(dict(model=model,max_abs_test_mean_difference=float(np.max(abs(raw_mean-saved_model.predicted_mean))),
            max_abs_test_std_difference=float(np.max(abs(raw_std-saved_model.predicted_std)))))
        times=(grid-train['alignment_peak_time'])/train['t_scale']
        Z=times[:,None] if model=='A_single' else np.column_stack([times,np.full(len(grid),DEFAULT_BAND_TO_WAVELENGTH['i'])])
        mean,std=predict(Z);curves[model]=dict(mean=(mean*scale).tolist(),latent_std=(std*scale).tolist())
    observations={}
    for band in ('i','z'):
        t,y,e=_extract_valid_band_observations(example,band)
        observations[band]=dict(mjd=t.tolist(),flux=y.tolist(),error=e.tolist())
    return dict(grid_mjd=grid.tolist(),curves=curves,observations=observations,
        training_band=tr['band'].tolist(),training_mjd=(tr['X'][:,0]*tr['t_scale']+tr['alignment_peak_time']).tolist(),
        training_flux=tr['y_raw'].tolist(),training_error=tr['yerr_raw'].tolist(),
        evaluation_mjd=(te['X'][:,0]*tr['t_scale']+tr['alignment_peak_time']).tolist(),
        evaluation_flux=te['y_raw'].tolist(),evaluation_error=te['yerr_raw'].tolist()),checks


def exact_failure_plot(output,payload):
    import matplotlib.pyplot as plt
    p=payload;grid=np.asarray(p['grid_mjd']);obs=p['observations']['i'];tz=np.asarray(p['observations']['z']['mjd'])
    ti=np.asarray(obs['mjd']);yi=np.asarray(obs['flux']);ei=np.asarray(obs['error'])
    reference=float(min(ti.min(),tz.min()))
    peak=float(tz[np.argmax(p['observations']['z']['flux'])])
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    for ax in axes:
        for model in analysis.DISPLAY_MODELS:
            c=p['curves'][model];mean=np.asarray(c['mean']);std=np.asarray(c['latent_std']);color=COLORS[model]
            ax.plot(grid-reference,mean,color=color,label=LABELS[model])
            ax.fill_between(grid-reference,mean-2*std,mean+2*std,color=color,alpha=.07)
            ax.fill_between(grid-reference,mean-std,mean+std,color=color,alpha=.18)
        ax.scatter(ti-reference,yi,s=13,color='.6',label='Complete i')
        mask=np.asarray(p['training_band'])=='i'
        ax.errorbar(np.asarray(p['training_mjd'])[mask]-reference,np.asarray(p['training_flux'])[mask],
            yerr=np.asarray(p['training_error'])[mask],fmt='o',ms=4,color='#172b4d',label='i training')
        ax.errorbar(np.asarray(p['evaluation_mjd'])-reference,p['evaluation_flux'],yerr=p['evaluation_error'],
            fmt='x',color='#b54829',label='i evaluation')
        ax.axvline(peak-reference,color='.4',ls=':',label='z peak')
        ax.set_xlim(peak-reference-60,peak-reference+60);ax.set_xlabel(f'MJD − {reference:.5f} [days]');ax.grid(alpha=.15)
    axes[0].set_ylabel('i flux · raw units');axes[0].legend(fontsize=8);axes[0].set_title('Full posterior range')
    axes[1].set_ylim(yi.min()-2*ei.max(),yi.max()+2*ei.max());axes[1].set_title('Target flux zoom · uncertainty clipped')
    fig.suptitle('Object 9858227 · exact i+z diagnostic · target fraction 0.2')
    analysis.savefig(fig,output,'failure_9858227_gp_comparison')


def run(output):
    output=Path(output).resolve()
    before={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output.rglob('*') if p.is_file() and 'figures' not in p.relative_to(output).parts}
    stage=Path(tempfile.mkdtemp(prefix='.four-model-figures-',dir=output))
    records=[]
    def save(fig,unused,name):
        import matplotlib.pyplot as plt
        from matplotlib.text import Text
        texts=[t.get_text() for t in fig.findobj(match=Text)]
        joined='\n'.join(texts)
        assert LABELS['C_shared_original_bounds'] not in joined
        assert 'Historical C/A' not in joined
        assert 'seed 1' not in joined
        assert 'training/evaluation flux zoom' not in joined
        for extension in ('png','pdf'):fig.savefig(stage/f'{name}.{extension}',dpi=220)
        records.append(dict(figure=name,n_axes=len(fig.axes),text_original_bounds_present=False,
            model_labels=[LABELS[m] for m in analysis.DISPLAY_MODELS if LABELS[m] in joined],text=texts))
        plt.close(fig)
        print('Rendered:',name,flush=True)
    read=lambda name:pd.read_csv(output/name,dtype={'object_id':str},float_precision='round_trip')
    with patch('sklearn.gaussian_process.GaussianProcessRegressor.fit',side_effect=AssertionError('No GP refits allowed')),patch('sklearn.gaussian_process.GaussianProcessRegressor.predict',side_effect=AssertionError('Use saved curves / fixed-parameter conditioning')),patch.object(analysis,'savefig',side_effect=save):
        analysis.plots(output,read('object_level_scores.csv'),read('population_summary.csv'),read('object_paired_improvements.csv'),
            read('transition_table.csv'),read('tail_risk_summary.csv'),read('hyperparameters.csv'),
            read('model_scores.csv').query('fit_status == "success"'),read('amplitude_association_features.csv'))
        payload,checks=exact_failure_payload(output)
        exact_failure_plot(output,payload)
    assert len(records)==13
    changed=[key for key,h in before.items() if hashlib.sha256((output/key).read_bytes()).hexdigest()!=h]
    assert not changed,changed
    representative=next(r for r in records if r['figure']=='representative_success_gp_comparison_target_zoom')
    assert representative['n_axes']==8
    assert len(representative['model_labels'])==4
    old=output/'figures';backup=Path(tempfile.mkdtemp(prefix='old-population-figures-'))/'figures'
    old.rename(backup)
    try:stage.rename(old)
    except BaseException:
        backup.rename(old);raise
    shutil.rmtree(backup);backup.parent.rmdir()
    cache=output/'four_model_plot_cache';cache.mkdir(exist_ok=True)
    (cache/'exact_i_z_fixed_parameter_curves.json').write_text(json.dumps(payload)+'\n')
    validation=dict(validation_passed=True,display_models=list(analysis.DISPLAY_MODELS),n_figures=len(records),
        n_png=len(list(old.glob('*.png'))),n_pdf=len(list(old.glob('*.pdf'))),gp_refits=0,hyperparameter_optimizations=0,
        representative_zoom_auxiliary_bands=['u','g','r','z'],exact_failure_saved_prediction_checks=checks,
        original_data_files_checked=len(before),original_data_files_changed=changed,figures=records)
    (output/'four_model_plot_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    gallery=['# Four-model population ablation figures','',
        'Displayed models: Single-band, Independent-band, Shared matched bounds, Band amplitudes. Original-bound results are retained in numerical audit files only. All plots use saved results; no GP hyperparameters were fitted or optimized.','',
        'The representative target zoom includes complete auxiliary observations and the exact selected auxiliary training points, in separate raw-flux panels. Target zoom clipping is documented below the figure. Seeds are retained in metadata; figure titles omit seed and verbose zoom text.','']
    for r in records:
        gallery += [f"## {r['figure']}",'',f"![{r['figure']}]({old/r['figure']}.png)",'']
    (output/'FOUR_MODEL_FIGURES.md').write_text('\n'.join(gallery)+'\n')
    print(json.dumps({k:v for k,v in validation.items() if k!='figures'},indent=2),flush=True)
    return validation


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'band_amplitude_population_ablation')
    run(parser.parse_args().output)
