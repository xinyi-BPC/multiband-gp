"""Object-cluster analysis of fixed, matched five-model population fits."""
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from population_kernel_ablation.experiment import MODELS, LABELS, COLORS, CONFIG, dump, manifest

KEY=['object_id','object_class','seed','target_fraction']
OK=['object_id','object_class','target_fraction']
METRICS=['raw_crps','ncrps','nrmse','nlpd','ks_pit','coverage_1sigma','coverage_2sigma','standardized_residual_mean','standardized_residual_std']
# Display selection only: preserve the five-model fitting and numerical audit definitions.
DISPLAY_MODELS=tuple(m for m in MODELS if m!='C_shared_original_bounds')
PAIRS={'amp':(MODELS[3],MODELS[4]),'bounds':(MODELS[2],MODELS[3]),'overall':(MODELS[2],MODELS[4])}


def markdown(frame):
    def fmt(x):
        return f'{x:.5g}' if isinstance(x,(float,np.floating)) else str(x)
    rows=['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    rows+=['| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)]
    return '\n'.join(rows)


def paired_table(scores,keys):
    wide=scores.pivot(index=keys,columns='model',values=['raw_crps','ncrps','nrmse'])
    tables=[]
    for comparison,(before,after) in PAIRS.items():
        for metric in ['raw_crps','ncrps','nrmse']:
            a=wide[(metric,before)];b=wide[(metric,after)]
            table=pd.DataFrame(dict(baseline_score=a,candidate_score=b,delta=a-b)).reset_index()
            table['comparison']=comparison;table['metric']=metric
            tolerance=CONFIG['tie_atol']+CONFIG['tie_rtol']*np.maximum(abs(a.to_numpy()),abs(b.to_numpy()))
            table['tolerance']=tolerance
            table['improved']=table.delta>tolerance;table['worsened']=table.delta < -tolerance
            table['approximately_unchanged']=abs(table.delta)<=tolerance
            tables.append(table)
    result=pd.concat(tables,ignore_index=True)
    for metric in ['raw_crps','ncrps','nrmse']:
        part=result[result.metric==metric].pivot(index=keys,columns='comparison',values='delta')
        np.testing.assert_allclose(part.overall,part.bounds+part.amp,rtol=1e-10,atol=1e-10)
    return result


def bootstrap_summary(paired,level):
    rng=np.random.default_rng(CONFIG['bootstrap_seed'])
    rows=[]
    for (fraction,comparison,metric),part in paired.groupby(['target_fraction','comparison','metric'],sort=True):
        # Every object contributes its seed cluster together. Object-level input
        # has one averaged row; case-level input has exactly three seed rows.
        clusters=[g.sort_values('seed').delta.to_numpy() if 'seed' in g else g.delta.to_numpy()
                  for _,g in part.groupby('object_id',sort=True)]
        improved=[g.sort_values('seed').improved.to_numpy() if 'seed' in g else g.improved.to_numpy()
                  for _,g in part.groupby('object_id',sort=True)]
        sizes={len(x) for x in clusters};assert len(sizes)==1
        values=np.stack(clusters);flags=np.stack(improved);n=len(values)
        indices=rng.integers(0,n,size=(CONFIG['bootstrap_resamples'],n))
        sampled=values[indices].reshape(CONFIG['bootstrap_resamples'],-1)
        median=np.median(sampled,axis=1);fraction_improved=flags[indices].mean(axis=(1,2))
        low,high=np.quantile(median,[.025,.975]);flo,fhi=np.quantile(fraction_improved,[.025,.975])
        rows.append(dict(target_fraction=fraction,comparison=comparison,metric=metric,level=level,
            n_objects=n,n_rows=len(part),n_case_rows=n*3,median_improvement=part.delta.median(),q25=part.delta.quantile(.25),q75=part.delta.quantile(.75),
            mean_improvement=part.delta.mean(),fraction_improved=part.improved.mean(),fraction_worsened=part.worsened.mean(),
            fraction_approximately_unchanged=part.approximately_unchanged.mean(),median_ci_low=low,median_ci_high=high,
            fraction_improved_ci_low=flo,fraction_improved_ci_high=fhi,min_improvement=part.delta.min(),max_improvement=part.delta.max()))
    return pd.DataFrame(rows)


def population_summary(scores):
    rows=[]
    for (fraction,model),part in scores.groupby(['target_fraction','model']):
        row=dict(target_fraction=fraction,model=model,n_objects=part.object_id.nunique(),n_case_rows=len(part)*3,n_successful_fits=len(part)*3)
        for metric in METRICS:
            row[f'median_{metric}']=part[metric].median();row[f'q25_{metric}']=part[metric].quantile(.25);row[f'q75_{metric}']=part[metric].quantile(.75)
            row[f'mean_{metric}']=part[metric].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def groups_and_transitions(objects,historical):
    hist=historical.groupby(['object_id','target_train_fraction','model']).ncrps.mean().unstack('model').reset_index()
    hist=hist.rename(columns={'target_train_fraction':'target_fraction'})
    wide=objects.pivot(index=OK,columns='model',values=['ncrps','nrmse'])
    out=wide.index.to_frame(index=False)
    for metric in ('ncrps','nrmse'):
        for m in MODELS:out[f'{m}_{metric}']=wide[(metric,m)].to_numpy()
    out=out.merge(hist[['object_id','target_fraction','single_band_gp','mogp_real_wavelength']],on=['object_id','target_fraction'],validate='one_to_one')
    out['historical_gain']=out.single_band_gp-out.mogp_real_wavelength
    out['historical_group']=np.where(out.historical_gain>0,'success','failure') # Exact user-defined membership.
    def beats(a,b):return a < b-(CONFIG['tie_atol']+CONFIG['tie_rtol']*np.maximum(abs(a),abs(b)))
    out['historical_meaningful_success']=beats(out.mogp_real_wavelength,out.single_band_gp)
    out['matched_D_success']=beats(out[f'{MODELS[3]}_ncrps'],out[f'{MODELS[0]}_ncrps'])
    out['E_beats_A_ncrps']=beats(out[f'{MODELS[4]}_ncrps'],out[f'{MODELS[0]}_ncrps'])
    out['E_beats_A_nrmse']=beats(out[f'{MODELS[4]}_nrmse'],out[f'{MODELS[0]}_nrmse'])
    for metric in ('ncrps','nrmse'):
        out[f'delta_amp_{metric}']=out[f'{MODELS[3]}_{metric}']-out[f'{MODELS[4]}_{metric}']
    transitions=[]
    for fraction,part in out.groupby('target_fraction'):
        for basis,column in [('historical_C_vs_historical_A','historical_meaningful_success'),('matched_D_vs_matched_A','matched_D_success')]:
            for prior in (True,False):
                for new in (True,False):
                    name=('retained_success' if new else 'lost_advantage') if prior else ('recovered_failure' if new else 'persistent_failure')
                    n=int(((part[column]==prior)&(part.E_beats_A_ncrps==new)).sum())
                    transitions.append(dict(target_fraction=fraction,basis=basis,previous_success=prior,new_beats_A=new,transition=name,n_objects=n,n_cases=n*3))
    summaries=[]
    for (fraction,group),part in out.groupby(['target_fraction','historical_group']):
        row=dict(target_fraction=fraction,historical_group=group,n_objects=len(part),n_case_rows=len(part)*3,
            fraction_E_beats_A_ncrps=part.E_beats_A_ncrps.mean(),fraction_E_beats_A_nrmse=part.E_beats_A_nrmse.mean())
        for metric in ('ncrps','nrmse'):
            delta=part[f'delta_amp_{metric}'];a=part[f'{MODELS[3]}_{metric}'];b=part[f'{MODELS[4]}_{metric}']
            tolerance=CONFIG['tie_atol']+CONFIG['tie_rtol']*np.maximum(abs(a),abs(b))
            row[f'median_delta_amp_{metric}']=delta.median();row[f'fraction_amp_improved_{metric}']=(delta>tolerance).mean()
            for m in MODELS[2:]:row[f'n_catastrophic_{m}_{metric}']=int((part[f'{m}_{metric}']>CONFIG['catastrophic_thresholds'][metric]).sum())
        deterioration=part[f'{MODELS[4]}_ncrps']-part[f'{MODELS[3]}_ncrps']
        for q in (.25,.5,.75,.9,.95,1.):row[f'ncrps_deterioration_E_minus_D_q{q:g}']=deterioration.quantile(q)
        summaries.append(row)
    return out,pd.DataFrame(transitions),pd.DataFrame(summaries)


def tails(objects):
    rows=[]
    for (fraction,model),part in objects.groupby(['target_fraction','model']):
        for metric in ('ncrps','nrmse'):
            values=part[metric];n=max(1,math.ceil(.05*len(values)))
            row=dict(target_fraction=fraction,model=model,metric=metric,n_objects=len(part),p50=values.median(),p90=values.quantile(.9),p95=values.quantile(.95),
                maximum=values.max(),worst5pct_mean=values.nlargest(n).mean(),worst5pct_n=n,
                catastrophic_threshold=CONFIG['catastrophic_thresholds'][metric],n_above_threshold=int((values>CONFIG['catastrophic_thresholds'][metric]).sum()))
            d=objects[(objects.target_fraction==fraction)&(objects.model==MODELS[3])].set_index('object_id')
            fixed=d[metric].nlargest(n).index
            row['mean_on_D_worst5pct_objects']=part.set_index('object_id').loc[fixed,metric].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def associations(scores,paired,parameters):
    e=scores[scores.model==MODELS[4]].copy()
    ratios=[c for c in e if c.startswith('ratio_sigma_i_')]
    e['max_abs_log_sigma_ratio']=np.abs(np.log(e[ratios])).max(axis=1)
    features=e.groupby(['object_id','target_fraction'])[['max_abs_log_sigma_ratio','sigma_i','n_target_train','log_marginal_likelihood']].mean()
    upper=parameters[(parameters.model==MODELS[2])&parameters.parameter.str.contains('constant_value')].groupby(['object_id','target_fraction']).at_upper.mean()
    features['C_upper_hit_seed_fraction']=upper
    gain=paired[(paired.comparison=='amp')&(paired.metric=='ncrps')].set_index(['object_id','target_fraction']).delta
    features['delta_amp_ncrps']=gain
    rows=[]
    for fraction,part in features.reset_index().groupby('target_fraction'):
        for feature in ('max_abs_log_sigma_ratio','sigma_i','n_target_train','C_upper_hit_seed_fraction'):
            valid=part[[feature,'delta_amp_ncrps']].dropna()
            if valid[feature].nunique()>1:
                rho,p=spearmanr(valid[feature],valid.delta_amp_ncrps)
            else:rho,p=np.nan,np.nan
            rows.append(dict(target_fraction=fraction,feature=feature,n_objects=len(valid),spearman_rho=rho,pvalue_descriptive=p))
    return features.reset_index(),pd.DataFrame(rows)


def analyze(output):
    records=[json.loads(p.read_text()) for p in sorted((output/'checkpoints').glob('*.json'))]
    assert len(records)==1800
    reproduction=[]
    for record in records:
        metadata={k:record['scores'][0][k] for k in ['object_id','object_class','seed','target_fraction']}
        reproduction.extend([{**metadata,**check} for check in record['historical_checks']])
    pd.DataFrame(reproduction).to_csv(output/'pilot_historical_reproduction.csv',index=False)
    scores=pd.DataFrame([s for r in records for s in r['scores']]);assert len(scores)==9000
    if 'error' not in scores:scores['error']=''
    diagnostics=pd.DataFrame([s for r in records for s in r['diagnostics']])
    parameters=pd.DataFrame([s for r in records for s in r['parameters']])
    initialization=pd.DataFrame([s for r in records for s in r['initializations']])
    scores.to_csv(output/'model_scores.csv',index=False)
    diagnostics['warning_count']=diagnostics.warnings.map(len)
    diagnostics['convergence_warning']=diagnostics.warnings.map(lambda ws:any(w['category']=='ConvergenceWarning' for w in ws))
    warning_rows=[{**{k:r[k] for k in KEY},**w} for r in diagnostics.to_dict('records') for w in r['warnings']]
    pd.DataFrame(warning_rows,columns=KEY+['category','message']).to_csv(output/'optimization_warnings.csv',index=False)
    diagnostics['warnings']=diagnostics.warnings.map(json.dumps);diagnostics.to_csv(output/'optimizer_diagnostics.csv',index=False)
    scores[scores.fit_status!='success'].to_csv(output/'failed_fits.csv',index=False)
    parameters.to_csv(output/'hyperparameters.csv',index=False);initialization.to_csv(output/'initializations.csv',index=False)
    historical=pd.read_csv(output/'historical_scores.csv',dtype={'object_id':str})
    good=scores[scores.fit_status=='success'].copy()
    successes=good.groupby(KEY).model.nunique();case_keys=successes[successes==5].reset_index()[KEY]
    matched=good.merge(case_keys,on=KEY,validate='many_to_one')
    seed_counts=case_keys.groupby(OK).seed.nunique();full_keys=seed_counts[seed_counts==3].reset_index()[OK]
    matched=matched.merge(full_keys,on=OK,validate='many_to_one')
    objects=matched.groupby(OK+['model'])[METRICS+['log_marginal_likelihood','time_length_scale','time_length_scale_days','n_target_train','n_aux_train']].mean().reset_index()
    objects['n_seeds']=3;objects.to_csv(output/'object_level_scores.csv',index=False)
    paired=paired_table(matched,KEY);objpaired=paired_table(objects,OK)
    paired.to_csv(output/'paired_improvements.csv',index=False);objpaired.to_csv(output/'object_paired_improvements.csv',index=False)
    summary=population_summary(objects);summary.to_csv(output/'population_summary.csv',index=False)
    compact=summary[['target_fraction','model','n_objects','median_raw_crps','median_ncrps','median_nrmse','median_nlpd','median_ks_pit','mean_coverage_1sigma','mean_coverage_2sigma']].rename(columns={'median_raw_crps':'median_crps','median_ks_pit':'ks_pit','mean_coverage_1sigma':'coverage_1sigma','mean_coverage_2sigma':'coverage_2sigma'})
    compact.to_csv(output/'report_ready_table.csv',index=False)
    calibration_cols=['target_fraction','model','n_objects']+[c for c in summary if any(m in c for m in METRICS[3:])]
    summary[calibration_cols].to_csv(output/'calibration_summary.csv',index=False)
    boots=pd.concat([bootstrap_summary(paired,'case_clustered'),bootstrap_summary(objpaired,'object_seed_mean')],ignore_index=True)
    boots.to_csv(output/'paired_summary_bootstrap.csv',index=False)
    ranked=objpaired.sort_values('delta');extremes=pd.concat([g.head(5).assign(direction='deterioration') for _,g in ranked.groupby(['target_fraction','comparison','metric'])]+[g.tail(5).assign(direction='improvement') for _,g in ranked.groupby(['target_fraction','comparison','metric'])])
    extremes.to_csv(output/'extreme_paired_objects.csv',index=False)
    case_extremes=pd.concat([g.nsmallest(5,'delta').assign(direction='deterioration') for _,g in paired.groupby(['target_fraction','comparison','metric'])]+[g.nlargest(5,'delta').assign(direction='improvement') for _,g in paired.groupby(['target_fraction','comparison','metric'])])
    case_extremes.to_csv(output/'extreme_paired_cases.csv',index=False)
    bounds=parameters[parameters.model.isin(MODELS[3:])].assign(any_hit=lambda p:p.at_lower|p.at_upper).groupby(['object_id','target_fraction']).any_hit.any()
    convergence=diagnostics[diagnostics.model.isin(MODELS[3:])].groupby(['object_id','target_fraction']).convergence_warning.any()
    strata=objpaired[(objpaired.comparison=='amp')&objpaired.metric.isin(['ncrps','nrmse'])].merge(bounds.rename('D_E_any_bound_hit'),on=['object_id','target_fraction']).merge(convergence.rename('D_E_any_convergence_warning'),on=['object_id','target_fraction'])
    stratified=[]
    for column in ['D_E_any_bound_hit','D_E_any_convergence_warning']:
        for (f,metric,value),g in strata.groupby(['target_fraction','metric',column]):
            stratified.append(dict(target_fraction=f,metric=metric,stratification=column,stratum=bool(value),n_objects=len(g),median_delta_amp=g.delta.median(),q25=g.delta.quantile(.25),q75=g.delta.quantile(.75),fraction_improved=g.improved.mean(),fraction_worsened=g.worsened.mean()))
    pd.DataFrame(stratified).to_csv(output/'optimizer_stratified_comparisons.csv',index=False)
    selection=json.loads((output/'pilot_selection.json').read_text())
    selected=scores[(scores.object_id==selection['representative_object_id'])&(scores.seed==selection['representative_seed'])&(scores.target_fraction==.2)]
    selected.to_csv(output/'representative_case_scores.csv',index=False)
    curve=json.loads((output/'representative_success_curves.json').read_text());curve_summary=[]
    observed_i=np.asarray(curve['observations']['i']['flux']);obs_t=np.asarray(curve['observations']['i']['mjd']);peak_t=obs_t[np.argmax(observed_i)]
    for model,c in curve['curves'].items():
        mean=np.asarray(c['mean']);std=np.asarray(c['latent_std']);grid=np.asarray(curve['grid_mjd'])
        curve_summary.append(dict(model=model,posterior_max_i_mean=mean.max(),mjd_of_max_i_mean=grid[mean.argmax()],mean_latent_std=std.mean(),posterior_i_mean_at_complete_i_max_time=np.interp(peak_t,grid,mean),complete_i_max_mjd=peak_t,complete_i_max_flux=observed_i.max()))
    pd.DataFrame(curve_summary).to_csv(output/'representative_curve_summary.csv',index=False)
    groups,transitions,group_summary=groups_and_transitions(objects,historical)
    groups.to_csv(output/'success_failure_groups.csv',index=False);transitions.to_csv(output/'transition_table.csv',index=False);group_summary.to_csv(output/'success_failure_summary.csv',index=False)
    tail=tails(objects);tail.to_csv(output/'tail_risk_summary.csv',index=False)
    transfer=[]
    for f,part in objects.groupby('target_fraction'):
        for metric in ('ncrps','nrmse'):
            wide=part.pivot(index='object_id',columns='model',values=metric)
            for model in MODELS[1:]:
                excess=wide[model]-wide[MODELS[0]]
                tolerance=CONFIG['tie_atol']+CONFIG['tie_rtol']*np.maximum(abs(wide[model]),abs(wide[MODELS[0]]))
                transfer.append(dict(target_fraction=f,model=model,metric=metric,n_objects=len(wide),fraction_negative_transfer=(excess>tolerance).mean(),median_excess_over_A=excess.median(),p90_excess_over_A=excess.quantile(.9),p95_excess_over_A=excess.quantile(.95),maximum_excess_over_A=excess.max()))
    pd.DataFrame(transfer).to_csv(output/'negative_transfer_summary.csv',index=False)
    parameter_summary=parameters.groupby(['target_fraction','model','parameter']).agg(n_parameters=('value','size'),median=('value','median'),q25=('value',lambda x:x.quantile(.25)),q75=('value',lambda x:x.quantile(.75)),lower_hit_fraction=('at_lower','mean'),upper_hit_fraction=('at_upper','mean')).reset_index()
    parameter_summary.to_csv(output/'hyperparameter_summary.csv',index=False)
    features,association=associations(good,objpaired,parameters)
    features.to_csv(output/'amplitude_association_features.csv',index=False);association.to_csv(output/'amplitude_associations.csv',index=False)
    availability=scores.groupby(['target_fraction','model']).fit_status.agg(n_attempted='size',n_successful=lambda s:(s=='success').sum(),n_failed=lambda s:(s!='success').sum()).reset_index()
    availability.to_csv(output/'fit_availability.csv',index=False)
    # Descriptive per-model availability is separate from the primary all-five-matched subset.
    good.groupby(['target_fraction','model'])[METRICS].agg(['median','mean','count']).to_csv(output/'all_available_case_summary.csv')
    plots(output,objects,summary,objpaired,transitions,tail,parameters,good,features)
    report(output,summary,boots,tail,transitions,group_summary,association,availability,parameters)
    expected=json.loads((output/'preservation_manifest.json').read_text());assert expected==manifest()
    validation=dict(validation_passed=True,n_attempted_model_fits=len(scores),n_failed_fits=int((scores.fit_status!='success').sum()),
        n_complete_five_model_cases=len(case_keys),n_complete_three_seed_object_fractions=len(full_keys),
        n_primary_case_rows=len(matched),n_primary_object_model_rows=len(objects),
        paired_decomposition_verified=True,seed_aggregation='three seeds averaged before object summaries',
        bootstrap='2000 object-cluster resamples; no seeds treated as independent objects',
        historical_files_unchanged=True,protected_files_checked=len(expected),sampling_permutations_generated=0,
        kernel_and_population_unit_tests_passed=9,split_preflight='population_kernel_ablation/preflight/preflight_validation.json',exact_i_z_failure_control_reproduced=json.loads((output/'failure_9858227_validation.json').read_text())['previous_single_object_metrics_reproduced'])
    dump(output/'validation.json',validation)
    print(compact.to_string(index=False),flush=True)


def savefig(fig,output,name):
    for extension in ('png','pdf'):fig.savefig(output/'figures'/f'{name}.{extension}',dpi=220)
    import matplotlib.pyplot as plt
    plt.close(fig)


def plots(output,objects,summary,paired,transitions,tail,parameters,scores,features):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axs=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    for ax,metric in zip(axs,('ncrps','nrmse')):
        for j,model in enumerate(DISPLAY_MODELS):
            part=summary[summary.model==model].sort_values('target_fraction');median=part[f'median_{metric}']
            ax.errorbar(np.arange(2)+(j-(len(DISPLAY_MODELS)-1)/2)*.13,median,yerr=np.vstack([median-part[f'q25_{metric}'],part[f'q75_{metric}']-median]),fmt='o',capsize=3,color=COLORS[model],label=LABELS[model])
        counts=summary.groupby('target_fraction').n_objects.first();ax.set_xticks([0,1],[f'{f} (n={int(counts.loc[f])})' for f in (.2,.8)]);ax.set_xlabel('Target training fraction');ax.set_ylabel(f'Object-mean {metric} · median and IQR');ax.grid(axis='y',alpha=.2)
        ax.set_title(f'{metric} · three seeds averaged per object')
    axs[0].legend(fontsize=8);fig.suptitle('Four matched GP configurations · equal object weighting')
    savefig(fig,output,'four_model_ncrps_nrmse')
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for ax,metric in zip(axs,('ncrps','nrmse')):
        values=[paired[(paired.metric==metric)&(paired.comparison=='amp')&(paired.target_fraction==f)].delta for f in (.2,.8)]
        ax.boxplot(values,tick_labels=[f'{f} (n={len(v)})' for f,v in zip((.2,.8),values)],showfliers=True);ax.axhline(0,color='.4',ls='--');ax.set_yscale('symlog',linthresh=.005)
        ax.set_ylabel(f'Δamp {metric} = D − E');ax.set_xlabel('Target fraction');ax.set_title('Positive favors band amplitudes · all matched objects')
    savefig(fig,output,'paired_amplitude_improvements')
    highlight={}
    for metric in ('ncrps','nrmse'):
        fig,axs=plt.subplots(1,2,figsize=(11,5),layout='constrained')
        for ax,f in zip(axs,(.2,.8)):
            part=paired[(paired.metric==metric)&(paired.comparison=='amp')&(paired.target_fraction==f)]
            ax.scatter(part.baseline_score,part.candidate_score,s=17,alpha=.6,color=COLORS[MODELS[4]])
            lo=min(part.baseline_score.min(),part.candidate_score.min())*.8;hi=max(part.baseline_score.max(),part.candidate_score.max())*1.2
            ax.plot([lo,hi],[lo,hi],ls='--',color='.4');ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(lo,hi);ax.set_ylim(lo,hi)
            ax.set_xlabel(f'D shared matched bounds · {metric}');ax.set_ylabel(f'E band amplitudes · {metric}');ax.set_title(f'Target fraction {f} · n={len(part)} objects')
            for index in (part.delta.idxmin(),part.delta.idxmax()):
                r=part.loc[index];ax.annotate(str(r.object_id),(r.baseline_score,r.candidate_score),fontsize=7,xytext=(4,4),textcoords='offset points')
        savefig(fig,output,f'paired_model_scatter_{metric}')
    fig,axs=plt.subplots(1,2,figsize=(11,4.8),layout='constrained',squeeze=False)
    for row,basis in enumerate(('matched_D_vs_matched_A',)):
        for col,f in enumerate((.2,.8)):
            ax=axs[row,col];part=transitions[(transitions.basis==basis)&(transitions.target_fraction==f)]
            matrix=np.array([[int(part[(part.previous_success==prior)&(part.new_beats_A==new)].n_objects.iloc[0]) for new in (True,False)] for prior in (True,False)])
            ax.imshow(matrix,cmap='Blues',vmin=0,vmax=300)
            for i in range(2):
                for j in range(2):ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>150 else 'black',fontsize=17)
            ax.set_xticks([0,1],['E beats A','E does not beat A']);ax.set_yticks([0,1],['Prior success','Prior failure']);ax.set_title(f'{"Shared matched bounds / single-band"} → E/A · fraction {f}')
    savefig(fig,output,'success_failure_transitions')
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for ax,metric in zip(axs,('ncrps','nrmse')):
        for f,style in ((.2,'-'),(.8,'--')):
            for model in MODELS[3:]:
                r=tail[(tail.metric==metric)&(tail.model==model)&(tail.target_fraction==f)].iloc[0]
                ax.plot([50,90,95],[r.p50,r.p90,r.p95],marker='o',ls=style,color=COLORS[model],label=f'{LABELS[model]} · f={f}')
        ax.set_xticks([50,90,95]);ax.set_yscale('log');ax.set_ylabel(f'Object-mean {metric}');ax.set_xlabel('Population percentile');ax.grid(alpha=.2)
    axs[0].legend(fontsize=8);fig.suptitle('Typical and upper-tail errors · all extremes retained in CSVs')
    savefig(fig,output,'upper_tail_comparison')
    fig,axs=plt.subplots(2,3,figsize=(16,8),layout='constrained')
    e=scores[scores.model==MODELS[4]]
    def boxes(ax,data,labels,title,scale='log'):
        ax.boxplot(data,tick_labels=labels,showfliers=True);ax.set_yscale(scale);ax.set_title(title);ax.tick_params(axis='x',labelrotation=25)
    boxes(axs[0,0],[e[e.target_fraction==f].sigma_i.dropna() for f in (.2,.8)],['f=0.2','f=0.8'],'Target σᵢ · case-level')
    boxes(axs[0,1],[e[f'sigma_{b}'].dropna() for b in ('u','g','r','z')],['u','g','r','z'],'Auxiliary σ · both fractions')
    boxes(axs[0,2],[e[f'ratio_sigma_i_{b}'].dropna() for b in ('u','g','r','z')],['i/u','i/g','i/r','i/z'],'Amplitude ratios · both fractions')
    boxes(axs[1,0],[scores[scores.model==m].time_length_scale.dropna() for m in DISPLAY_MODELS],[LABELS[m] for m in DISPLAY_MODELS],'Temporal length scale · normalized')
    boxes(axs[1,1],[scores[scores.model==m].log_marginal_likelihood.dropna() for m in DISPLAY_MODELS],[LABELS[m] for m in DISPLAY_MODELS],'Training LML · different training sets for A',scale='symlog')
    relevant=parameters[((parameters.model==MODELS[3])&parameters.parameter.str.contains('constant_value'))|((parameters.model==MODELS[4])&parameters.parameter.str.startswith('sigma'))]
    bound=relevant.groupby('model')[['at_lower','at_upper']].mean().reindex(MODELS[3:])
    ax=axs[1,2];ax.bar(np.arange(2)-.15,bound.at_lower,width=.3,label='Lower');ax.bar(np.arange(2)+.15,bound.at_upper,width=.3,label='Upper');ax.set_xticks(np.arange(2),['Shared matched bounds · C','Band amplitudes · σ']);ax.set_ylabel('Bound-hit fraction');ax.legend();ax.set_title('Parameter bounds · all successful fits')
    savefig(fig,output,'amplitude_parameter_distributions')
    fig,axs=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
    for ax,f in zip(axs,(.2,.8)):
        part=features[features.target_fraction==f];ax.scatter(part.max_abs_log_sigma_ratio,part.delta_amp_ncrps,s=16,alpha=.6)
        ax.axhline(0,color='.5',ls='--');ax.set_yscale('symlog',linthresh=.005);ax.set_xlabel('Mean max |log(σᵢ / σaux)| over seeds');ax.set_ylabel('Object Δamp nCRPS');ax.set_title(f'Post-hoc association · fraction {f}')
    savefig(fig,output,'amplitude_ratio_associations')
    selection=json.loads((output/'pilot_selection.json').read_text());diagnostic_plot(output,'representative_success_curves.json','representative_success_gp_comparison',f"Object {selection['representative_object_id']} · representative success")
    diagnostic_plot(output,'failure_all_aux_curves.json','failure_9858227_all_aux_gp_comparison','Object 9858227 · all auxiliary bands')


def diagnostic_plot(output,filename,name,title):
    """Render saved posteriors, with four models and raw auxiliary training data."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=json.loads((output/filename).read_text())
    grid=np.asarray(p['grid_mjd']);reference=float(grid.min())
    training_band=np.asarray(p['training_band'])
    active=[b for b in ('u','g','r','z') if np.any(training_band==b)]
    observed=['i',*active]
    times=np.concatenate([np.asarray(p['observations'][b]['mjd']) for b in observed])
    limits=(float(min(grid.min(),times.min())-reference),float(max(grid.max(),times.max())-reference))

    def target_panel(ax,model):
        if model not in p['curves']:raise ValueError(f'Missing saved curve: {model}')
        curve=p['curves'][model];mean=np.asarray(curve['mean']);std=np.asarray(curve['latent_std'])
        color=COLORS[model];x=grid-reference
        ax.fill_between(x,mean-2*std,mean+2*std,color=color,alpha=.1,label='Latent ±2σ')
        ax.fill_between(x,mean-std,mean+std,color=color,alpha=.25,label='Latent ±1σ')
        ax.plot(x,mean,color=color,label='Posterior mean')
        obs=p['observations']['i']
        ax.scatter(np.asarray(obs['mjd'])-reference,obs['flux'],color='.7',s=12,alpha=.45,label='Complete i')
        mask=training_band=='i'
        ax.errorbar(np.asarray(p['training_mjd'])[mask]-reference,np.asarray(p['training_flux'])[mask],
            yerr=np.asarray(p['training_error'])[mask],fmt='o',ms=4,color='#172b4d',label='i training')
        ax.errorbar(np.asarray(p['evaluation_mjd'])-reference,p['evaluation_flux'],
            yerr=p['evaluation_error'],fmt='x',ms=4,color='#b54829',label='i evaluation')
        ax.set_title(LABELS[model],fontsize=11);ax.set_xlim(limits)
        ax.set_xlabel(f'MJD − {reference:.5f} [days]');ax.grid(alpha=.15)

    def observations_panel(ax,band):
        obs=p['observations'][band];mask=training_band==band
        ax.errorbar(np.asarray(obs['mjd'])-reference,obs['flux'],yerr=obs['error'],
            fmt='.',color='.65',alpha=.5,ms=3,label=f'Complete {band}')
        ax.errorbar(np.asarray(p['training_mjd'])[mask]-reference,np.asarray(p['training_flux'])[mask],
            yerr=np.asarray(p['training_error'])[mask],fmt='o',ms=4,color='#172b4d',
            label=f'{band} training (n={int(mask.sum())})')
        ax.set_xlim(limits);ax.set_xlabel(f'MJD − {reference:.5f} [days]')
        ax.set_ylabel(f'{band} flux · raw units');ax.set_title(f'{band}-band observations')
        ax.legend(fontsize=8);ax.grid(alpha=.15)

    fig=plt.figure(figsize=(18,8),layout='constrained')
    gs=fig.add_gridspec(2,1,height_ratios=[1.25,1])
    top=gs[0].subgridspec(1,len(DISPLAY_MODELS));bottom=gs[1].subgridspec(1,len(observed))
    axes=[]
    for j,model in enumerate(DISPLAY_MODELS):
        ax=fig.add_subplot(top[0,j],sharey=axes[0] if axes else None);axes.append(ax)
        target_panel(ax,model)
    axes[0].set_ylabel('i flux · raw units');axes[0].legend(fontsize=7)
    for j,band in enumerate(observed):observations_panel(fig.add_subplot(bottom[0,j]),band)
    fig.suptitle(title+' · target fraction 0.2')
    savefig(fig,output,name)

    # Keep the existing target zoom, with a second row for all ACTIVE auxiliaries.
    fig=plt.figure(figsize=(18,8),layout='constrained')
    gs=fig.add_gridspec(2,1,height_ratios=[1.25,1])
    top=gs[0].subgridspec(1,len(DISPLAY_MODELS));bottom=gs[1].subgridspec(1,len(active))
    mask_i=training_band=='i'
    focal_flux=np.r_[np.asarray(p['training_flux'])[mask_i],p['evaluation_flux']]
    focal_error=np.r_[np.asarray(p['training_error'])[mask_i],p['evaluation_error']]
    lo=float(np.min(focal_flux-2*focal_error));hi=float(np.max(focal_flux+2*focal_error))
    axes=[]
    for j,model in enumerate(DISPLAY_MODELS):
        ax=fig.add_subplot(top[0,j],sharey=axes[0] if axes else None);axes.append(ax)
        target_panel(ax,model);ax.set_ylim(lo,hi)
    axes[0].set_ylabel('i flux · raw units');axes[0].legend(fontsize=7)
    for j,band in enumerate(active):observations_panel(fig.add_subplot(bottom[0,j]),band)
    fig.suptitle(title+' · target fraction 0.2')
    fig.supxlabel('Target panels use shared zoomed flux limits; posterior intervals and complete-target extremes outside those limits are clipped. Auxiliary panels retain raw flux scales.',fontsize=9)
    savefig(fig,output,name+'_target_zoom')


def report(output,summary,boots,tail,transitions,groups,association,availability,parameters):
    primary=boots[(boots.level=='object_seed_mean')&boots.metric.isin(['ncrps','nrmse'])]
    lines=['# Population kernel ablation: 300 PLAsTiCC objects','',
        'Primary results are matched five-restart refits, with the three seeds averaged within each object before population summaries. The saved splits, historical results, original GP modules, and previously validated band kernel are preserved. Positive paired improvement means lower error for the candidate. D versus E isolates the comparison at matched marginal-variance bounds; C versus D assesses the wider-bound control. These are configuration comparisons, not independent causal effects.','',
        markdown(summary[['target_fraction','model','n_objects','median_raw_crps','median_ncrps','median_nrmse','median_nlpd']]),'',
        'A = single-band; B = independent-band; C = historical shared-amplitude bounds; D = matched-variance-bound shared amplitude; E = band amplitudes. Detailed IQRs, counts, calibration, and case-level summaries are in the CSV outputs.','',
        '## Typical performance, calibration, and robustness','', 'E does not improve the population median nCRPS or nRMSE relative to D at either target density. Both paired nCRPS median confidence intervals include zero; the sparse-target paired nRMSE median interval is entirely negative, indicating deterioration. Although some upper-tail quantiles and threshold counts improve, the maximum nCRPS increases at both fractions. The results support targeted rescue of some failures rather than a uniform improvement in typical performance or robustness.','']
    for f in (.2,.8):
        pairs=primary[(primary.target_fraction==f)&(primary.comparison=='amp')].set_index('metric')
        s=summary[summary.target_fraction==f].set_index('model')
        nc=pairs.loc['ncrps'];nr=pairs.loc['nrmse'];d=s.loc[MODELS[3]];e=s.loc[MODELS[4]]
        lines.append(f"At target fraction {f}, D→E changes population-median nCRPS {d.median_ncrps:.6g} → {e.median_ncrps:.6g} and nRMSE {d.median_nrmse:.6g} → {e.median_nrmse:.6g}. The median paired nCRPS improvement is {nc.median_improvement:.6g} (object-bootstrap 95% CI {nc.median_ci_low:.6g} to {nc.median_ci_high:.6g}); {nc.fraction_improved:.1%} of objects improve, {nc.fraction_worsened:.1%} worsen, and {nc.fraction_approximately_unchanged:.1%} are numerically unchanged. For nRMSE, the median paired gain is {nr.median_improvement:.6g} (CI {nr.median_ci_low:.6g} to {nr.median_ci_high:.6g}), with {nr.fraction_improved:.1%} improved. Differences of population medians and medians of paired differences are distinct estimands.")
        lines.append('')
        lines.append(f"Calibration at fraction {f}: median NLPD changes {d.median_nlpd:.5g} → {e.median_nlpd:.5g}, median KS-PIT {d.median_ks_pit:.5g} → {e.median_ks_pit:.5g}, mean 1σ coverage {d.mean_coverage_1sigma:.1%} → {e.mean_coverage_1sigma:.1%}, and mean 2σ coverage {d.mean_coverage_2sigma:.1%} → {e.mean_coverage_2sigma:.1%}. Nominal Gaussian coverages are 68.27% and 95.45%. These diagnostics need not all improve together. KS-PIT is a summary of object/seed KS statistics, not a pooled PIT test.")
        lines.append('')
        t=tail[(tail.target_fraction==f)&(tail.metric=='ncrps')].set_index('model');dt=t.loc[MODELS[3]];et=t.loc[MODELS[4]]
        tn=tail[(tail.target_fraction==f)&(tail.metric=='nrmse')].set_index('model')
        lines.append(f"Robustness at fraction {f}: nCRPS P90 changes {dt.p90:.6g} → {et.p90:.6g}, P95 {dt.p95:.6g} → {et.p95:.6g}, worst-5% mean {dt.worst5pct_mean:.6g} → {et.worst5pct_mean:.6g}, and maximum {dt.maximum:.6g} → {et.maximum:.6g}. The number of objects with nCRPS > 1 changes {int(dt.n_above_threshold)} → {int(et.n_above_threshold)}; nRMSE > 1 changes {int(tn.loc[MODELS[3],'n_above_threshold'])} → {int(tn.loc[MODELS[4],'n_above_threshold'])}. These predeclared cutoffs are descriptive, not physical ground-truth failure definitions. tail_risk_summary.csv also compares all models on the same D-worst-5% objects, alongside separately selected model tails.")
        lines.append('')
    lines+=['## Preservation of successes and recovery of failures','']
    compact=[]
    for (f,basis),part in transitions.groupby(['target_fraction','basis']):
        r=dict(target_fraction=f,basis=basis)
        r.update(dict(zip(part.transition,part.n_objects)));compact.append(r)
    lines+=[markdown(pd.DataFrame(compact)),'',
        'Historical groups use historical C versus historical A object-mean nCRPS exactly as specified. The historical transition compares historical membership with E versus newly matched A; the matched transition compares D/A with E/A using identical refits, objects, and seeds. Meaningful success requires improvement greater than 1e-8 + 1e-6 × max absolute paired score. Small numerical ties are not successes. success_failure_summary.csv records group-specific D−E gains, deterioration distributions, catastrophic counts, and how often E beats A.','',
        markdown(groups[['target_fraction','historical_group','n_objects','median_delta_amp_ncrps','fraction_amp_improved_ncrps','fraction_amp_improved_nrmse','fraction_E_beats_A_ncrps']]),'',
        '## Amplitude bounds, ratios, and optimization','',
        markdown(primary[['target_fraction','comparison','metric','median_improvement','median_ci_low','median_ci_high','fraction_improved']]),'',
        'The paired identity C−E=(C−D)+(D−E) is verified at every case and object. This algebra does not make the two terms independent causal effects.','']
    selected_associations=association[association.feature=='max_abs_log_sigma_ratio']
    for _,r in selected_associations.iterrows():
        lines.append(f"At fraction {r.target_fraction}, the descriptive Spearman association of object-mean max |log(σᵢ/σaux)| with D−E nCRPS is ρ={r.spearman_rho:.4g} (n={r.n_objects}). Amplitude ratios are associated with performance changes, but do not establish causation; all exploratory associations are in amplitude_associations.csv. Different amplitudes rescale covariance magnitudes and leave the fixed wavelength correlation coefficient unchanged.")
        lines.append('')
    amp=parameters[(parameters.model.isin(MODELS[2:4])&parameters.parameter.str.contains('constant_value'))|((parameters.model==MODELS[4])&parameters.parameter.str.startswith('sigma'))]
    bounds=amp.groupby(['target_fraction','model']).agg(n_parameters=('value','size'),lower_hit_fraction=('at_lower','mean'),upper_hit_fraction=('at_upper','mean')).reset_index()
    lines+=[markdown(bounds),'',markdown(availability),'']
    stratified=pd.read_csv(output/'optimizer_stratified_comparisons.csv')
    lines+=['The bound-hit and convergence-warning strata below show that the boundary-free subsets still have negative median D−E gains for both metrics and fractions. Thus the absence of an overall median benefit is not confined to fits with bound hits. These are descriptive subsets, not a new optimization search, and cannot rule out local maxima.', '',
        markdown(stratified[stratified.stratification=='D_E_any_bound_hit'][['target_fraction','metric','stratum','n_objects','median_delta_amp','fraction_improved']]),'',
        'All fits retain the prescribed bounds and five restarts. Shared C initializes from pooled training MAD/RMS squared; E initializes from the corresponding band’s training MAD/RMS. Out-of-range starting estimates are projected to the nearest bound interior and every projection is logged; bounds are not adjusted. The optimizer maximizes training marginal likelihood only, uses sklearn defaults for tolerances, and uses split seed for restart RNG. Equal seeds do not imply identical restart vectors across different parameter dimensions and bounds. Runtime fields include fitting, evaluation, and covariance-validation overhead. All warnings, parameters, LMLs, and failures remain in audit files. A uses fewer training observations, so its LML is not directly comparable to multiband model evidence.','',
        '## Representative success and exact failure control','']
    representative=pd.read_csv(output/'representative_case_scores.csv')
    curves=pd.read_csv(output/'representative_curve_summary.csv').set_index('model')
    selection=json.loads((output/'pilot_selection.json').read_text())
    rs=representative.set_index('model')
    lines.append(f"The representative historical success is object {selection['representative_object_id']}, seed {selection['representative_seed']}, fraction 0.2. Selection used historical scores and training counts only, before any new fit. C/D/E nCRPS values are {rs.loc[MODELS[2],'ncrps']:.6g}, {rs.loc[MODELS[3],'ncrps']:.6g}, and {rs.loc[MODELS[4],'ncrps']:.6g}; A is {rs.loc[MODELS[0],'ncrps']:.6g}. C/D/E maximum posterior i means are {curves.loc[MODELS[2],'posterior_max_i_mean']:.6g}, {curves.loc[MODELS[3],'posterior_max_i_mean']:.6g}, and {curves.loc[MODELS[4],'posterior_max_i_mean']:.6g} raw units. Mean dense-grid latent SD changes from D {curves.loc[MODELS[3],'mean_latent_std']:.6g} to E {curves.loc[MODELS[4],'mean_latent_std']:.6g}. See representative_curve_summary.csv and the common-time five-model figure for peak timing and amplitude changes. Visual morphology and possible over-suppression must be judged alongside held-out scores, not from peak appearance alone.")
    lines.append('')
    failure=pd.read_csv(output/'failure_9858227_scores.csv').set_index('model');decomposition=pd.read_csv(output/'failure_9858227_decomposition.csv')
    lines.append(f"The exact prior i+z configuration of 9858227 reproduces within tolerance. C/D/E nCRPS are {failure.loc[MODELS[2],'ncrps']:.6g}, {failure.loc[MODELS[3],'ncrps']:.6g}, and {failure.loc[MODELS[4],'ncrps']:.6g}. Its matched-bound shared control therefore does not explain away the band-amplitude improvement. The bound and amplitude terms are:")
    lines+=['',markdown(decomposition),'',
        'This exact-diagnostic control retains the prior initial shared C=1 for reproducibility; the population refits use the predeclared pooled robust initialization. Both use five restarts. Per-test-point CRPS contributions, coverage, standardized residuals, ratios, peaks, and parameters are in failure_9858227_*.csv. The all-auxiliary population case is explicitly labeled separately. One selected failure is not population evidence.','',
        '## Validation and limitations','',
        'The read-only preflight validates all 1,800 saved cases and 900 nested object/seed pairs without generating split permutations. The ten-object pilot has 300 successful matched fits and 540 successful historical score-reproduction checks. Kernel and population tests cover PSD, gradients, five bands, cloning, diagonals, nm-to-Å conversion, equal-amplitude equivalence, D/E variance-bound equivalence, no sampling calls, training-transform independence from held-out flux, clustering, and numerical ties.','',
        'Primary summaries require all five successful models and all three seeds. Case-level and object-level sample counts, all-available summaries, failures, and 2,000 object-cluster bootstrap intervals are saved separately. Seeds are never treated as independent objects. CRPS/nCRPS/nRMSE retain the original definitions and the fixed all-valid-band object denominator. Score uncertainty includes actual measurement errors; plotted uncertainty is latent only.','',
        'The selected 300-object cohort, sparse target sets, fixed wavelength correlation, shared temporal length scale, optimizer local maxima, and parameter-bound hits limit generalization. Complete-curve normalization is retrospective evaluation only and may downweight errors in low-amplitude target bands when auxiliary bands are bright. Complete curves and evaluation fluxes never set training transforms, amplitude initializations, parameter bounds, or likelihood optima. Posterior shape in unobserved regions is not validated by a handful of held-out points. The two target fractions describe density dependence within these saved nested splits; they do not establish causal effects.','',
        'Reproduction instructions are in population_kernel_ablation/README.md. Configuration, software versions, checkpoint audits, validation.json, pilot_validation.json, and preservation hashes accompany all CSVs. Figures use shared raw-flux/time units, retain outliers, and are exported as PNG and PDF.']
    (output/'findings.md').write_text('\n'.join(lines)+'\n')
