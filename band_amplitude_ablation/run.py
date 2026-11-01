"""Run only object 9858227; no historical file is written. See README.md."""
import argparse
import json
from pathlib import Path
import platform
import sys
from types import SimpleNamespace
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import scipy
import sklearn
from MOGP_model import (DEFAULT_BAND_TO_WAVELENGTH, THREE_MODEL_SPARSITY_MODELS,
    _make_target_sparsity_plan, run_target_band_ablation_study)
from data_processing import _extract_valid_band_observations
from evaluation_metrics import gaussian_crps, negative_log_predictive_density
from singleGP_model import object_level_empirical_flux_scale
from part3_analysis import validate_refit, protection_manifest, posterior_curves
from band_amplitude_ablation.kernel import fit_band_amplitude_gp

LABELS = dict(zip(THREE_MODEL_SPARSITY_MODELS, ['A · Single-band', 'B · Independent-band', 'C · Shared amplitude']))
D = 'band_specific_amplitude'
LABELS[D] = 'D · Band amplitudes'
COLORS = ['#375b9b', '#8c6bb1', '#d6604d', '#20877a']


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item()) + '\n')


def fit_capture(log, stage, function):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = function()
    log.extend(dict(stage=stage, category=w.category.__name__, message=str(w.message)) for w in caught)
    return result


def parameter_table(artifact, model):
    gp = artifact['gp']; k = gp.kernel_
    parameters = []
    names = []
    for h in k.hyperparameters:
        if not h.fixed:
            names.extend(f'{h.name}[{j}]' for j in range(h.n_elements))
    for name, theta, bounds in zip(names, k.theta, k.bounds):
        parameters.append(dict(model=model, parameter=name, value=float(np.exp(theta)),
            lower=float(np.exp(bounds[0])), upper=float(np.exp(bounds[1])),
            at_lower=bool(np.isclose(theta, bounds[0], atol=1e-5, rtol=0)),
            at_upper=bool(np.isclose(theta, bounds[1], atol=1e-5, rtol=0))))
    return parameters


def score(artifact, model, scale, raw_time):
    train, test = artifact['train_data'], artifact['heldout_data']
    _, _, mean, std = posterior_curves(artifact, raw_time, model)
    y = test['y_raw']; z = (y - mean) / std
    crps = gaussian_crps(y, mean, std)
    nlpd = negative_log_predictive_density(y, mean, std ** 2)
    # Confirm raw and normalized prediction/observation distributions agree.
    mn, latent = artifact['gp'].predict(test['X'], return_std=True)
    sn = np.sqrt(latent ** 2 + test['yerr'] ** 2)
    background = np.asarray(test.get('background_flux', 0))
    np.testing.assert_allclose(mean, mn * train['flux_scale'] + background)
    np.testing.assert_allclose(std, sn * train['flux_scale'])
    np.testing.assert_allclose(z, (test['y'] - mn) / sn, atol=1e-10)
    np.testing.assert_allclose(gaussian_crps(test['y'], mn, sn) * train['flux_scale'], crps)
    # Grid prediction at evaluation times must be the identical latent posterior.
    dense_mean, dense_std, _, _ = posterior_curves(artifact, raw_time, model)
    np.testing.assert_allclose(dense_mean, mean)
    np.testing.assert_allclose(dense_std ** 2 + test['yerr_raw'] ** 2, std ** 2, atol=1e-7)
    k = artifact['gp'].kernel_
    temporal = k.time_length_scale if model.startswith(D) else (k.k2.length_scale if model == 'single_band_gp' else k.k1.k2.length_scale)
    row = dict(model=model, n_evaluation=len(y), raw_crps=float(crps.mean()), ncrps=float(crps.mean()/scale),
        nrmse=float(np.sqrt(np.mean((y-mean)**2))/scale), nlpd=float(nlpd.mean()),
        coverage_1sigma=float(np.mean(np.abs(z)<=1)), coverage_2sigma=float(np.mean(np.abs(z)<=2)),
        standardized_residual_mean=float(z.mean()), standardized_residual_std=float(z.std(ddof=0)),
        time_length_scale=float(temporal), time_length_scale_days=float(temporal*train['t_scale']),
        object_flux_scale=scale, log_marginal_likelihood=float(artifact['gp'].log_marginal_likelihood_value_),
        n_restarts_optimizer=artifact['gp'].n_restarts_optimizer)
    if model.startswith(D):
        amplitudes = dict(zip(k.bands, np.asarray(k.sigma).tolist()))
        row.update(sigma_i=amplitudes['i'], sigma_z=amplitudes['z'], R_sigma=amplitudes['i']/amplitudes['z'])
    else:
        C = k.k1.constant_value if model == 'single_band_gp' else k.k1.k1.constant_value
        row.update(constant_covariance=C, shared_sigma=np.sqrt(C))
    table = pd.DataFrame(dict(object_id='34161124', evaluation_mjd=raw_time, observed_i_flux=y,
        model=model, predicted_mean=mean, predicted_std=std, standardized_residual=z,
        individual_crps=crps, individual_nlpd=nlpd, contribution_to_mean_crps=crps/len(y),
        contribution_to_ncrps=crps/len(y)/scale, fraction_of_total_crps=crps/crps.sum()))
    return row, table


def plots(example, artifacts, grid, peak, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False,
        'axes.spines.right':False, 'savefig.dpi':220, 'pdf.fonttype':42})
    ti, yi, ei = _extract_valid_band_observations(example, 'i')
    tz, yz, ez = _extract_valid_band_observations(example, 'z')
    reference = float(min(ti.min(), tz.min()))
    curves = {m: posterior_curves(a, grid, m)[:2] for m,a in artifacts.items()}
    def observations(ax, artifact):
        tr, te = artifact['train_data'], artifact['heldout_data']
        train_t = tr['X'][:,0]*tr['t_scale'] + tr['alignment_peak_time']
        bands = np.asarray(tr.get('band', 'i'))
        if bands.ndim == 0:
            bands = np.repeat(bands, len(train_t))
        mask = bands == 'i'
        test_t = te['X'][:,0]*tr['t_scale'] + tr['alignment_peak_time']
        ax.scatter(ti-reference, yi, color='.65', s=17, alpha=.45, label='Complete i', zorder=3)
        ax.errorbar(train_t[mask]-reference,tr['y_raw'][mask], yerr=tr['yerr_raw'][mask],fmt='o',ms=4,color='#172b4d',label='i training',zorder=5)
        ax.errorbar(test_t-reference,te['y_raw'], yerr=te['yerr_raw'],fmt='x',ms=5,color='#b54829',label='i evaluation',zorder=6)
    def prediction(ax, model, color):
        mean, std = curves[model]
        ax.fill_between(grid-reference,mean-2*std,mean+2*std,color=color,alpha=.10,label='Latent ±2σ')
        ax.fill_between(grid-reference,mean-std,mean+std,color=color,alpha=.23,label='Latent ±1σ')
        ax.plot(grid-reference,mean,color=color,lw=1.5,label='Posterior mean')
        ax.axvline(peak-reference,color='.4',ls=':',lw=1)
        observations(ax, artifacts[model])
        ax.set_xlim(grid.min()-reference,grid.max()-reference)
        ax.set_title(LABELS[model]);ax.grid(alpha=.15)
    fig=plt.figure(figsize=(18,8),layout='constrained')
    gs=fig.add_gridspec(2,4,height_ratios=[1.2,1])
    axes=[]
    for j,(m,color) in enumerate(zip(artifacts,COLORS)):
        ax=fig.add_subplot(gs[0,j],sharey=axes[0] if axes else None)
        prediction(ax,m,color);axes.append(ax)
        ax.set_xlabel(f'MJD − {reference:.5f} [days]')
    axes[0].set_ylabel('i flux [raw units]')
    low=min(np.min(curves[m][0]-2*curves[m][1]) for m in artifacts)
    high=max(np.max(curves[m][0]+2*curves[m][1]) for m in artifacts)
    axes[0].set_ylim(min(low,yi.min()-ei.max())*1.05,max(high,yi.max()+ei.max())*1.05)
    axes[0].legend(fontsize=7,loc='upper left')
    ax=fig.add_subplot(gs[1,:2]);observations(ax,artifacts['mogp_real_wavelength'])
    for model,color in zip(list(artifacts)[2:],COLORS[2:]):
        ax.plot(grid-reference,curves[model][0],color=color,lw=1.2,label=LABELS[model])
    ax.axvline(peak-reference,color='.4',ls=':');ax.set_title('Complete i observations and transferred posterior');ax.set_ylabel('i flux [raw units]');ax.legend(fontsize=7)
    ax=fig.add_subplot(gs[1,2:]);ax.errorbar(tz-reference,yz,yerr=ez,fmt='.',color='.55',label='Complete z')
    tr=artifacts[D]['train_data'];mask=tr['band']=='z';tt=tr['X'][:,0]*tr['t_scale']+tr['alignment_peak_time']
    ax.scatter(tt[mask]-reference,tr['y_raw'][mask],facecolors='none',edgecolors='#172b4d',s=50,label='Selected z training',zorder=5)
    j=np.argmax(yz);ax.annotate(f'Large z peak\nMJD {peak:.5f}',xy=(tz[j]-reference,yz[j]),xytext=(-130,-30),textcoords='offset points',arrowprops={'arrowstyle':'->'})
    ax.set_title('Complete z observations');ax.set_ylabel('z flux [raw units]');ax.legend(fontsize=8)
    for ax in fig.axes[4:]:ax.set_xlabel(f'MJD − {reference:.5f} [days]');ax.set_xlim(grid.min()-reference,grid.max()-reference);ax.grid(alpha=.15)
    fig.suptitle('PLAsTiCC 9858227 · M-dwarf · fixed i/z split · five optimizer restarts',fontsize=15)
    for ext in ('png','pdf'):fig.savefig(output/f'four_model_comparison.{ext}')
    plt.close(fig)
    fig,axs=plt.subplots(1,4,figsize=(18,4),sharex=True,sharey=True,layout='constrained')
    for ax,m,color in zip(axs,artifacts,COLORS):
        prediction(ax,m,color);ax.set_ylim(yi.min()-2*ei.max(), yi.max()+2*ei.max());ax.set_xlabel(f'MJD − {reference:.5f}')
    axs[0].set_ylabel('i flux [raw units]');fig.suptitle('Target-flux zoom · uncertainty outside the limits is clipped')
    for ext in ('png','pdf'):fig.savefig(output/f'target_flux_zoom.{ext}')
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    for ax in axs:
        for m,color in zip(list(artifacts)[2:],COLORS[2:]):
            mean,std=curves[m];ax.plot(grid-reference,mean,color=color,label=LABELS[m])
            ax.fill_between(grid-reference,mean-2*std,mean+2*std,color=color,alpha=.08)
            ax.fill_between(grid-reference,mean-std,mean+std,color=color,alpha=.18)
        observations(ax,artifacts[D]);ax.axvline(peak-reference,color='.3',ls=':',label='z peak time')
        ax.set_xlim(peak-reference-60,min(grid.max(),peak+60)-reference);ax.set_xlabel(f'MJD − {reference:.5f} [days]');ax.grid(alpha=.15)
    axs[0].set_title('z peak interval · full flux range');axs[1].set_title('Same interval · i flux zoom')
    axs[1].set_ylim(yi.min()-2*ei.max(),yi.max()+2*ei.max());axs[0].set_ylabel('i flux [raw units]');axs[0].legend(fontsize=8)
    for ext in ('png','pdf'):fig.savefig(output/f'peak_interval_zoom.{ext}')
    plt.close(fig)
    return reference,curves


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    protected=protection_manifest();dump(output/'preservation_before.json',protected)
    log=[]
    config=json.loads((ROOT/'three_model_experiment/full_configuration.json').read_text())
    example=next(e for e in json.loads((ROOT/'three_model_experiment/objects_300.json').read_text()) if str(e['object_id'])=='9858227')
    saved=pd.read_pickle(ROOT/'part2_experiment/full_300/part2_model_metrics_checkpoint.pkl')
    case=SimpleNamespace(object_id='9858227',target_band='i',seed=1,split_id='9858227:i:1',source_experiment='part2',aux_band='z')
    # Reconstruct original full-band RNG sequence, then retain z exactly as Part 2 did.
    plan=_make_target_sparsity_plan(example,'i',config['bands'],DEFAULT_BAND_TO_WAVELENGTH,.5,1)
    plan={**plan,'auxiliary':{'z':plan['auxiliary']['z']}}
    def baseline(settings):
        return run_target_band_ablation_study(example,target_band='i',bands=config['bands'],random_state=1,
            target_train_fraction=.2,aux_train_fraction=.5,models=THREE_MODEL_SPARSITY_MODELS,
            mogp_gp_kwargs={k:v for k,v in settings.items() if k!='random_state'},standardize_model_settings=True,_sampling_plan=plan)
    historical=fit_capture(log,'historical_0_restarts',lambda:baseline(config['mogp_gp_kwargs']))
    checks=validate_refit(historical,saved,case,.2)  # Must pass BEFORE new fits or plots.
    pd.DataFrame(checks).to_csv(output/'historical_reproduction.csv',index=False)
    settings={**config['mogp_gp_kwargs'],'n_restarts_optimizer':5,'random_state':1}
    matched=fit_capture(log,'matched_5_restarts',lambda:baseline(settings))
    for row in matched['rows']:
        old=next(r for r in historical['rows'] if r['model']==row['model'])
        for key in ('target_train_indices','heldout_indices','training_row_ids'):
            np.testing.assert_array_equal(row[key],old[key])
    artifacts={m:matched['artifacts'][m] for m in THREE_MODEL_SPARSITY_MODELS}
    real=artifacts['mogp_real_wavelength'];train=real['train_data'];test=real['heldout_data']
    initializations={}
    for name,bounds in [(D,(.01,100)),(D+'_wide',(.001,1000))]:
        (gp,init)=fit_capture(log,name,lambda:fit_band_amplitude_gp(train,sigma_bounds=bounds,**settings))
        artifacts[name]=dict(gp=gp,train_data=train,heldout_data=test);initializations[name]=init
    dump(output/'amplitude_initialization.json',initializations)
    raw_time=test['X'][:,0]*train['t_scale']+train['alignment_peak_time']
    ti,yi,ei=_extract_valid_band_observations(example,'i');tz,yz,ez=_extract_valid_band_observations(example,'z')
    np.testing.assert_allclose(raw_time,ti[matched['rows'][0]['heldout_indices']],atol=1e-10)
    scale=object_level_empirical_flux_scale(example=example)
    np.testing.assert_allclose(scale,matched['rows'][0]['object_flux_scale'],rtol=1e-14)
    rows=[];points=[];parameters=[]
    for m,a in artifacts.items():
        np.testing.assert_allclose(a['heldout_data']['y_raw'],test['y_raw'])
        np.testing.assert_allclose(a['heldout_data']['X'][:,0],test['X'][:,0])
        row,table=score(a,m,scale,raw_time);rows.append(row);points.append(table);parameters.extend(parameter_table(a,m))
    metrics=pd.DataFrame(rows);metrics.to_csv(output/'metrics.csv',index=False)
    pd.concat(points).to_csv(output/'test_point_diagnostics.csv',index=False)
    pd.DataFrame(parameters).to_csv(output/'parameters_and_bounds.csv',index=False)
    comparisons=[]
    for candidate in (D,D+'_wide'):
        for baseline_name in THREE_MODEL_SPARSITY_MODELS:
            new=metrics.set_index('model').loc[candidate];old=metrics.set_index('model').loc[baseline_name]
            comparisons.append(dict(candidate=candidate,baseline=baseline_name,delta_ncrps=old.ncrps-new.ncrps,delta_nrmse=old.nrmse-new.nrmse))
    pd.DataFrame(comparisons).to_csv(output/'paired_improvements.csv',index=False)
    split={key:matched['rows'][0][key] for key in ('target_train_indices','target_pool_indices','heldout_indices')}
    split.update(aux_selected_indices_by_band=matched['rows'][2]['aux_selected_indices_by_band'],
        training_row_ids=train['row_id'],preprocessing_flux_scale=train['flux_scale'],
        preprocessing_t_scale=train['t_scale'],alignment_peak_time=train['alignment_peak_time'],object_flux_scale=scale)
    dump(output/'split_and_normalization.json',split)
    pd.DataFrame(dict(band=train['band'],row_id=train['row_id'],source_index=train['source_indices'],
        mjd=train['X'][:,0]*train['t_scale']+train['alignment_peak_time'],flux_raw=train['y_raw'],
        flux_normalized=train['y'],error_raw=train['yerr_raw'])).to_csv(output/'training_observations.csv',index=False)
    covariance=artifacts[D]['gp'].kernel_(train['X'])
    np.testing.assert_allclose(covariance,covariance.T,atol=1e-12)
    minimum_eigenvalue=float(np.linalg.eigvalsh(covariance).min());assert minimum_eigenvalue>=-1e-8
    # No training/evaluation overlap; B/C/D share the very same dictionaries.
    assert not set(train['row_id']) & set(test['row_id'])
    for m in ('mogp_independent_band_control',D,D+'_wide'):
        for key in ('X','y','yerr','row_id'):
            np.testing.assert_array_equal(artifacts[m]['train_data'][key],train[key])
    peak=float(tz[np.argmax(yz)])
    grid=np.unique(np.r_[np.linspace(min(ti.min(),tz.min()),max(ti.max(),tz.max()),2500),
        np.linspace(peak-60,min(max(ti.max(),tz.max()),peak+60),700),raw_time,peak])
    # Validation above precedes figure generation.
    primary={m:artifacts[m] for m in LABELS}
    reference,curves=plots(example,primary,grid,peak,output)
    dense=[];peaks=[]
    for m,a in artifacts.items():
        mean,std=posterior_curves(a,grid,m)[:2]
        dense.append(pd.DataFrame(dict(model=m,mjd=grid,mean_raw=mean,latent_std_raw=std)))
        peak_mean,peak_std=posterior_curves(a,np.array([peak]),m)[:2]
        window=np.abs(grid-peak)<=60
        peaks.append(dict(model=m,z_peak_mjd=peak,i_mean_at_z_peak=peak_mean[0],i_latent_std_at_z_peak=peak_std[0],
            max_i_mean_within_60days=mean[window].max()))
    pd.concat(dense).to_csv(output/'dense_latent_predictions.csv',index=False)
    pd.DataFrame(peaks).to_csv(output/'peak_predictions.csv',index=False)
    dump(output/'warnings.json',log)
    dump(output/'configuration.json',dict(object_id='9858227',object_class='M-dwarf',target_band='i',auxiliary_bands=['z'],
        target_train_fraction=.2,aux_train_fraction=.5,seed=1,matched_fit_settings=settings,
        sigma_bounds=[.01,100],sensitivity_sigma_bounds=[.001,1000],wavelength_length_scale_A=6000,
        wavelength_mapping_nm={b:DEFAULT_BAND_TO_WAVELENGTH[b] for b in config['bands']},
        plot_mjd_reference=reference,python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__))
    after=protection_manifest();assert after==protected,'Protected original files changed'
    dump(output/'validation.json',dict(validation_passed=True,historical_metrics_exact=all(c['absolute_difference']==0 for c in checks),
        original_split_matches=True,identical_evaluation_points=True,raw_normalized_invariance=True,
        grid_evaluation_time_transform_matches=True,covariance_symmetric=True,covariance_minimum_eigenvalue=minimum_eigenvalue,
        shared_training_rows_B_C_D=True,no_training_evaluation_overlap=True,protected_files_checked=len(protected),protected_files_unchanged=True,
        kernel_tests='python -m unittest discover -s tests -p test_band_amplitude_ablation.py'))
    print(metrics.to_string(index=False),flush=True)
    print(f'Wrote {output}',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'band_amplitude_ablation/results2')
    run(parser.parse_args().output)
