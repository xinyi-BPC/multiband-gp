"""Exact earlier i+z diagnostic plus the new matched-bound shared control."""
from pathlib import Path
import json
import sys
import warnings
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from population_kernel_ablation.saved_splits import load_sources, reconstruct_saved_case
from population_kernel_ablation.experiment import evaluate, new_kernel_and_fit, MODELS, params, dump
from MOGP_model import fit_mogp_gp
from singleGP_model import fit_basic_gp
from band_amplitude_ablation.kernel import fit_band_amplitude_gp
from evaluation_metrics import gaussian_crps, negative_log_predictive_density
from data_processing import _extract_valid_band_observations


def run(output):
    examples,source=load_sources();example=examples['9858227']
    saved=source[(source.object_id=='9858227')&(source.random_state==1)&(source.target_train_fraction==.2)]
    data=reconstruct_saved_case(example,saved)
    for key in ('multi_train',):
        old=data[key];mask=np.isin(old['band'],['i','z']);n=len(mask)
        data[key]={k:np.asarray(v)[mask] if isinstance(v,np.ndarray) and len(v)==n else v for k,v in old.items()}
    train=data['multi_train'];test=data['multi_test']
    # Validate i+z indices directly against the prior single-object saved training table.
    previous=ROOT/'band_amplitude_ablation/results'
    np.testing.assert_array_equal(train['row_id'],pd.read_csv(previous/'training_observations.csv').row_id)
    old_scores=pd.read_csv(previous/'metrics.csv').set_index('model')
    baseline_settings=dict(n_restarts_optimizer=5,random_state=1,time_length_scale=.3,time_length_scale_bounds=(.05,10.),jitter=1e-8)
    models={};warnings_out=[];rows=[];point_rows=[];parameter_rows=[]
    for model in MODELS:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            if model=='A_single':gp=fit_basic_gp(data['single_train'],length_scale_bounds=(.05,10.),n_restarts_optimizer=5,random_state=1)
            elif model=='E_band_amplitudes':gp,_=fit_band_amplitude_gp(train,**baseline_settings)
            else:gp=fit_mogp_gp(train,constant_value_bounds=(.0001,10000.) if model==MODELS[3] else (.01,100.),wavelength_length_scale=1e-5 if model==MODELS[1] else 6000.,**baseline_settings)
        models[model]=gp
        warnings_out.extend(dict(model=model,category=w.category.__name__,message=str(w.message)) for w in caught)
        score=evaluate(gp,model,data,data['object_flux_scale']);score['model']=model;rows.append(score)
        parameter_rows.extend(params(gp,model))
        if model!=MODELS[3]:
            old_name=dict(zip([MODELS[0],MODELS[1],MODELS[2],MODELS[4]],['single_band_gp','mogp_independent_band_control','mogp_real_wavelength','band_specific_amplitude']))[model]
            for metric in ('raw_crps','ncrps','nrmse'):
                np.testing.assert_allclose(score[metric],old_scores.loc[old_name,metric],rtol=1e-6,atol=1e-8)
        te=data['single_test'] if model==MODELS[0] else test
        mean,latent=gp.predict(te['X'],return_std=True);scale=te['flux_scale'];mean*=scale;std=np.sqrt(latent**2+te['yerr']**2)*scale
        crps=gaussian_crps(te['y_raw'],mean,std)
        times=te['X'][:,0]*train['t_scale']+train['alignment_peak_time']
        point_rows.append(pd.DataFrame(dict(object_id='9858227',model=model,evaluation_mjd=times,observed_i_flux=te['y_raw'],predicted_mean=mean,predicted_std=std,standardized_residual=(te['y_raw']-mean)/std,individual_crps=crps,contribution_to_mean_crps=crps/len(crps),individual_nlpd=negative_log_predictive_density(te['y_raw'],mean,std**2))))
    tz,yz,_=_extract_valid_band_observations(example,'z');peak=tz[np.argmax(yz)];ti,yi,ei=_extract_valid_band_observations(example,'i')
    grid=np.unique(np.r_[np.linspace(peak-60,peak+60,1300),peak])
    curves={};peak_rows=[]
    for model,gp in models.items():
        times=(grid-train['alignment_peak_time'])/train['t_scale'];X=times[:,None] if model==MODELS[0] else np.column_stack([times,np.full(len(times),754.6)])
        mean,std=gp.predict(X,return_std=True);mean*=train['flux_scale'];std*=train['flux_scale']
        curves[model]=dict(mean=mean,latent_std=std)
        peak_rows.append(dict(model=model,i_mean_at_z_peak=mean[np.flatnonzero(grid==peak)[0]],max_i_mean_in_60day_window=mean.max(),z_peak_mjd=peak))
    scores=pd.DataFrame(rows);scores.to_csv(output/'failure_9858227_scores.csv',index=False)
    pd.concat(point_rows).to_csv(output/'failure_9858227_test_point_diagnostics.csv',index=False)
    pd.DataFrame(parameter_rows).to_csv(output/'failure_9858227_parameters.csv',index=False)
    pd.DataFrame(peak_rows).to_csv(output/'failure_9858227_peak_predictions.csv',index=False)
    dump(output/'failure_9858227_warnings.json',warnings_out)
    original=scores.set_index('model').loc[MODELS[2]];matched=scores.set_index('model').loc[MODELS[3]];new=scores.set_index('model').loc[MODELS[4]]
    decomposition=[]
    for metric in ('ncrps','nrmse'):
        total=original[metric]-new[metric];bound=original[metric]-matched[metric];amp=matched[metric]-new[metric]
        np.testing.assert_allclose(total,bound+amp)
        decomposition.append(dict(metric=metric,overall_improvement=total,bounds_improvement=bound,amplitude_improvement=amp,bounds_share_of_total=bound/total))
    pd.DataFrame(decomposition).to_csv(output/'failure_9858227_decomposition.csv',index=False)
    dump(output/'failure_9858227_validation.json',dict(previous_single_object_metrics_reproduced=True,training_i_z_row_ids_match=True,variance_bound_control_added=True))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from population_kernel_ablation.experiment import COLORS,LABELS
    from population_kernel_ablation.analysis import savefig, DISPLAY_MODELS
    reference=float(min(ti.min(),tz.min()));fig,axs=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    for ax in axs:
        for model in DISPLAY_MODELS:
            mean=curves[model]['mean'];std=curves[model]['latent_std'];color=COLORS[model]
            ax.plot(grid-reference,mean,color=color,label=LABELS[model]);ax.fill_between(grid-reference,mean-2*std,mean+2*std,color=color,alpha=.07);ax.fill_between(grid-reference,mean-std,mean+std,color=color,alpha=.18)
        ax.scatter(ti-reference,yi,s=13,color='.6',label='Complete i')
        tr=data['multi_train'];mask=tr['band']=='i';tt=tr['X'][:,0]*tr['t_scale']+tr['alignment_peak_time']
        ax.errorbar(tt[mask]-reference,tr['y_raw'][mask],yerr=tr['yerr_raw'][mask],fmt='o',ms=4,color='#172b4d',label='i training')
        test_t=test['X'][:,0]*tr['t_scale']+tr['alignment_peak_time']
        ax.errorbar(test_t-reference,test['y_raw'],yerr=test['yerr_raw'],fmt='x',color='#b54829',label='i evaluation')
        ax.axvline(peak-reference,color='.4',ls=':',label='z peak');ax.set_xlim(peak-reference-60,peak-reference+60);ax.set_xlabel(f'MJD − {reference:.5f} [days]');ax.grid(alpha=.15)
    axs[0].set_ylabel('i flux · raw units');axs[0].legend(fontsize=8);axs[0].set_title('Full posterior range');axs[1].set_ylim(yi.min()-2*ei.max(),yi.max()+2*ei.max());axs[1].set_title('Target flux zoom · uncertainty clipped')
    fig.suptitle('Object 9858227 · exact i+z diagnostic · fraction 0.2')
    savefig(fig,output,'failure_9858227_gp_comparison')
    print(scores[['model','ncrps','nrmse']].to_string(index=False));print(pd.DataFrame(decomposition).to_string(index=False))


if __name__=='__main__':run(ROOT/'band_amplitude_population_ablation')
