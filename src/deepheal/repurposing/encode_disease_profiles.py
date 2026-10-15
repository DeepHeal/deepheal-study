"""Align frozen external signatures and encode all inputs in each fixed VAE bank.

Only projection and its technical QA are performed here. Scores/statistics/plots
are calculated by the standalone R scripts.
"""

from deepheal.paths import ARTICLE_ROOT

import argparse
import pickle
import io
import warnings
from vae_common import *
from cptac_annotations import load_annotation, proteome_dir

DATA = PKG/'data/processed'
DATA.mkdir(parents=True,exist_ok=True)

def prepare():
    features,drugs,meta,mapping=projection.load_source_feature_data(ROOT)
    # Stable common drug order for every downstream analysis.
    drugs=drugs.sort_index()
    meta=meta.set_index('drug_norm').loc[drugs.index].reset_index()
    annotation=load_annotation(proteome_dir(ROOT))
    gene_to_sp=projection.annotation_gene_to_swissprot(annotation)
    vectors=[]; masks=[]; records=[]; maps=[]
    def add(source,context,genes,values):
        genes=[projection.norm(g) for g in genes]
        fi,gi,methods=projection.build_feature_assignment(genes,features,mapping,gene_to_sp)
        raw=np.zeros(len(features),dtype=np.float32)
        mask=np.zeros(len(features),dtype=bool)
        valid=np.isfinite(values[gi]);mask[fi[valid]]=True
        raw[fi[valid]]=values[gi[valid]]
        raw[np.abs(raw)<=.15]=0
        key=f'{source}:{context}'
        vectors.append(raw);masks.append(mask)
        records.append(dict(query_key=key,source=source,context=context,mapped_features=int(mask.sum()),
            mapped_fraction=float(mask.mean()),nonzero_features=int(np.count_nonzero(raw))))
        maps.extend(dict(query_key=key,gene_symbol=genes[g],feature=features[f],mapping_method=m,observed=bool(v))
                    for f,g,m,v in zip(fi,gi,methods,valid))
    for path in sorted((FIG8/'tables').glob('*_gene_signature.csv')):
        frame=pd.read_csv(path)
        add('CPTAC',path.stem.replace('_gene_signature','').upper(),frame.gene_symbol,frame.median_log2fc.to_numpy())
    tpcpa=pd.read_csv(OLD/'tables/tpcpa_gene_signatures_long.csv')
    for context,frame in tpcpa.groupby('tpcpa_cancer',sort=True):
        add('TPCPA',context,frame.gene_symbol,frame.effect_log2fc.to_numpy())
    patient=np.load(FIG8/'results/intermediates/patient_disease_vectors_COAD.npz')
    patient_meta=pd.read_csv(FIG8/'tables/cptac_patient_projection_metadata.csv')
    paired_ids=set(patient_meta.loc[patient_meta.cohort.eq('COAD') & patient_meta.reference_method.eq('paired_normal'),'patient_id'].astype(str))
    assert len(paired_ids)==96
    for i,patient_id in enumerate(patient['patient_ids'].astype(str)):
        if patient_id not in paired_ids: continue
        add('PATIENT',patient_id,patient['genes'].astype(str),patient['disease'][:,i])
    query=pd.DataFrame(records); values=np.vstack(vectors); observed=np.vstack(masks)
    assert len(features)==9960 and len(drugs)==875
    assert (query.source=='TPCPA').sum()==21 and (query.source=='PATIENT').sum()==96
    pd.DataFrame(maps).to_csv(DATA/'feature_mapping.csv.gz',index=False)
    query.to_csv(DATA/'query_metadata.csv',index=False)
    meta.to_csv(DATA/'drug_metadata.csv',index=False)
    pd.DataFrame({'feature':features}).to_csv(DATA/'feature_order.csv',index=False)
    # Transposed raw interfaces are numeric, compressed and read directly by R.
    pd.DataFrame(drugs.to_numpy(),index=drugs.index,columns=features).rename_axis('drug_norm').to_csv(DATA/'drug_protein_matrix.csv.gz')
    pd.DataFrame(values,index=query.query_key,columns=features).rename_axis('query_key').to_csv(DATA/'disease_protein_matrix.csv.gz')
    np.savez_compressed(DATA/'projection_inputs.npz',drug_values=drugs.to_numpy(dtype=np.float32),
        disease_values=values,observed=observed,drug_names=drugs.index.to_numpy(dtype=str),query_keys=query.query_key.to_numpy(dtype=str))
    return features,drugs,query,values,observed

def run(branch):
    inp=np.load(DATA/'projection_inputs.npz')
    query=pd.read_csv(DATA/'query_metadata.csv'); features=pd.read_csv(DATA/'feature_order.csv').feature.tolist()
    drug=inp['drug_values']; disease=inp['disease_values']; masks=inp['observed']
    names=inp['drug_names']; n=len(drug); q=len(query)
    pca=projection.load_pickle(ROOT/f'results/{branch}/{branch}_pca.pkl')
    native=drug
    if branch=='supcon':
        native=projection.load_supcon_drug_matrix(ROOT,features).loc[names].to_numpy(dtype=np.float32)
    raw=np.vstack([native,drug,np.zeros((1,len(features)),np.float32),disease,-disease])
    descriptors=[dict(kind='native_drug',item_id=x,context='') for x in names]
    descriptors += [dict(kind='drug',item_id=x,context='') for x in names]
    descriptors += [dict(kind='zero',item_id='zero',context='')]
    descriptors += [dict(kind='disease',item_id=k,context=k) for k in query.query_key]
    descriptors += [dict(kind='inverse',item_id=k,context=k) for k in query.query_key]
    pca_main=pca.transform(raw).astype(np.float32)
    blocks=[pca_main]
    for i,row in query.loc[query.source.ne('PATIENT')].iterrows():
        masked=drug*masks[i][None,:]
        blocks.append(pca.transform(masked).astype(np.float32))
        descriptors += [dict(kind='masked_drug',item_id=x,context=row.query_key) for x in names]
    values=np.vstack(blocks); desc=pd.DataFrame(descriptors)
    assert len(desc)==len(values)
    node_labels=pd.read_csv(ROOT/f'results/{branch}/{branch}_ensemble_node_labels.csv')
    dim=int(node_labels.ensemble_label.max())+1
    counts=np.zeros(dim,np.int32); consensus=np.zeros((len(values),dim),np.float64)
    ckpt_dir=PKG/'models/average_replayed' if branch=='average' else ROOT/'results/source_model_checkpoints/supcon'
    paths=sorted(ckpt_dir.glob('*.pt'));assert len(paths)==600
    mapping={(int(d),int(r)):g.sort_values('source_node_idx').ensemble_label.to_numpy(dtype=int)
             for (d,r),g in node_labels.groupby(['source_latent_dim','source_run'])}
    qa=[]; provenance=[]
    for j,path in enumerate(paths):
        payload=torch.load(path,map_location='cpu',weights_only=False); s=payload['source_summary']
        model=build_model(branch,payload); z=encode(model,values)
        if not np.isfinite(z).all(): raise RuntimeError(f'Nonfinite encoding: {path}')
        labels=mapping[int(s['source_latent_dim']),int(s['source_run'])]
        # Linear consensus aggregation occurs identically for all sample types.
        for k,l in enumerate(labels):
            consensus[:,l]+=z[:,k];counts[l]+=1
        if j in [0,199,399,599]:
            test=values[np.array([0,n,2*n,2*n+1,len(values)-1])]
            z1=encode(model,test,1); z5=encode(model,test,5)
            buffer=io.BytesIO();torch.save(model.state_dict(),buffer);buffer.seek(0)
            reloaded=build_model(branch,payload);reloaded.load_state_dict(torch.load(buffer,map_location=DEVICE,weights_only=True))
            zr=encode(reloaded,test,5)
            a=float(np.max(np.abs(z1-z5)));b=float(np.max(np.abs(zr-z5)))
            qa.append(dict(branch=branch,checkpoint=path.name,batch_max_abs_error=a,serialization_max_abs_error=b,
                batch_pass=bool(np.allclose(z1,z5,atol=2e-4,rtol=2e-4)),serialization_pass=bool(np.allclose(zr,z5,atol=1e-6,rtol=1e-6))))
            del reloaded
        provenance.append(dict(branch=branch,checkpoint=str(path),sha256=digest(path),source_latent_dim=int(s['source_latent_dim']),source_run=int(s['source_run'])))
        del model,z,payload
        if (j+1)%50==0: print(f'{branch}: encoded checkpoint {j+1}/600, {len(values)} vectors',flush=True)
    assert (counts>0).all()
    consensus=(consensus/counts).astype(np.float32)
    archive=pd.read_csv(ROOT/f'results/{branch}/{branch}_consensus_embedding_including_marginal.csv')
    archive['drug_norm']=archive.Drug_Name.map(projection.norm)
    archive=archive.set_index('drug_norm').loc[names]
    az=archive[[c for c in archive if c.startswith('latent_')]].to_numpy(dtype=np.float32)
    fidelity=rowcos(consensus[:n],az)
    empirical=pd.read_csv(ROOT/'results/hepg2_projection/tables/hepg2_consensus_reconstruction_fidelity.csv')
    reference=float(empirical.loc[empirical.branch==branch,'consensus_cosine_vs_archive'].median())
    passed=float(np.nanmedian(fidelity))>=max(.90,reference-.02)
    summary=dict(branch=branch,model_version='Average original-recipe replay' if branch=='average' else 'SupCon materialized 257D',
        n_checkpoints=600,consensus_dimensions=dim,median_cosine=float(np.nanmedian(fidelity)),
        fidelity_threshold=max(.90,reference-.02),meets_empirical_fidelity=passed,n_drugs=n,n_queries=q,n_encoded=len(values),
        pca_path=str(ROOT/f'results/{branch}/{branch}_pca.pkl'),pca_sha256=digest(ROOT/f'results/{branch}/{branch}_pca.pkl'),
        all_projection_checks_pass=all(r['batch_pass'] and r['serialization_pass'] for r in qa))
    pd.DataFrame({'branch':branch,'drug_norm':names,'consensus_cosine_vs_archive':fidelity,
        'rmse_vs_archive':np.sqrt(np.mean((consensus[:n]-az)**2,axis=1))}).to_csv(PKG/f'tables/{branch}_archive_fidelity.csv',index=False)
    pd.DataFrame(qa).to_csv(PKG/f'tables/{branch}_technical_checks.csv',index=False)
    pd.DataFrame(provenance).to_csv(PKG/f'results/{branch}_checkpoint_manifest.csv',index=False)
    write_json(PKG/f'results/{branch}_projection_summary.json',summary)
    if not passed: raise RuntimeError(f'{branch} failed archive restoration: {summary}; rebuild requires the separate original recipe stage')
    if not summary['all_projection_checks_pass']: raise RuntimeError('Projection technical QA failed')
    columns=[f'latent_{i:03d}' for i in range(dim)]
    pd.concat([desc,pd.DataFrame(consensus,columns=columns)],axis=1).to_csv(DATA/f'{branch}_embeddings.csv.gz',index=False,float_format='%.9g')
    # OOD thresholds are fitted only to the common HCT116 drug references.
    disease_start=2*n+1
    direct=np.vstack([drug,disease]); direct_pca=np.vstack([pca_main[n:2*n],pca_main[disease_start:disease_start+q]])
    ood=projection.ood_diagnostics(direct,direct_pca,pca,n,query.rename(columns={'query_key':'patient_key'}),branch)
    ood=ood.rename(columns={'patient_key':'query_key'})
    zero=consensus[2*n]; centered=consensus[n:2*n]-zero
    norm_ref=np.linalg.norm(centered,axis=1); latent_threshold=float(np.quantile(norm_ref,.99))
    ood['centered_latent_norm']=np.linalg.norm(consensus[disease_start:disease_start+q]-zero,axis=1)
    ood['latent_norm_99pct']=latent_threshold;ood['ood_latent_norm']=ood.centered_latent_norm>latent_threshold
    ood.to_csv(PKG/f'tables/{branch}_projection_ood.csv',index=False)
    print(summary,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--branch',choices=['prepare','average','supcon','all'],default='all');args=parser.parse_args()
    if args.branch in ['prepare','all']:prepare()
    for b in ['average','supcon']:
        if args.branch in [b,'all']:run(b)
