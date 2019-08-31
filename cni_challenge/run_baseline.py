import os
import numpy as np
import pandas as pd
from nilearn.connectome import ConnectivityMeasure
from sklearn.preprocessing import StandardScaler
from TPy.did import DISVM

train_dir = './inputdir'
outputdir = './outputdir'


def _load_fmri(sub_ids, path, atlas='cc200'):
    """Load time-series extracted from the fMRI using a specific atlas."""
    return [pd.read_csv(os.path.join(path,'%s/timeseries_%s.csv'%(sub_id, atlas)),
                        header=None).values.T
                     for sub_id in sub_ids]


def _load_data(datapath, partition='Training', atlas='cc200'):
    pheno_path = os.path.join(datapath, 'SupportingInfo/phenotypic_%s.csv'%partition.lower())

    pheno_df = pd.read_csv(pheno_path)

    data = _load_fmri(pheno_df['Subj'], os.path.join(datapath, partition), atlas=atlas)

    pheno_df_ = pheno_df.copy()

    for i in pheno_df_.index.values:
        if pheno_df_.loc[i, 'DX'] == 'ADHD':
            pheno_df_.loc[i, 'DX'] = 1
        else:
            pheno_df_.loc[i, 'DX'] = -1

    return data, pheno_df_


def get_train_data(atlas='aal'):
    X_train, pheno_train = _load_data(train_dir, partition='Training', atlas=atlas)
    X_valid, pheno_valid = _load_data(train_dir, partition='Validation', atlas=atlas)
    X_ = X_train + X_valid
    pheno_ = pd.concat([pheno_train, pheno_valid])

    return X_, pheno_


def get_test_data(test_dir, atlas='aal'):
    return _load_data(test_dir, partition='Test', atlas=atlas)


def sex2vec(sex_):
    sex = np.ones(sex_.shape)
    for i in range(sex_.shape[0]):
        if sex_[i] != 'M':
            sex[i] = -1
    return sex.reshape(-1, 1)


def run_baseline(test_dir, outdir, atlas='aal'):
    X_train, pheno_train = get_train_data(atlas=atlas)
    X_test, pheno_test = get_test_data(test_dir, atlas=atlas)
    X_all = X_train + X_test

    y_train = pheno_train['DX'].values
    n_train = len(X_train)
    n_sub = len(X_all)

    n_roi = X_all[0].shape[1]

    measure = ConnectivityMeasure(kind='tangent', vectorize=True)

    X_connectome = measure.fit_transform(X_all)
    X_meanstd = np.zeros((n_sub, n_roi*2))
    for i in range(n_sub):
        X_meanstd[i, :n_roi] = np.mean(X_all[i], axis=0)
        X_meanstd[i, n_roi:] = np.std(X_all[i], axis=0)
    X = np.concatenate([X_connectome, X_meanstd], axis=1)

    sex_train = pheno_train['Sex'].values
    age_train = pheno_train['Age'].values
    hand_train = pheno_train['Edinburgh_Handedness'].values

    sex_test = pheno_test['Sex'].values
    age_test = pheno_test['Age'].values
    hand_test = pheno_test['Edinburgh_Handedness'].values

    scaler = StandardScaler()

    sex = np.concatenate([sex2vec(sex_train), sex2vec(sex_test)])
    age = np.concatenate([scaler.fit_transform(age_train.reshape(-1, 1)),
                          scaler.fit_transform(age_test.reshape(-1, 1))])
    hand = np.concatenate([scaler.fit_transform(hand_train.reshape(-1, 1)),
                          scaler.fit_transform(hand_test.reshape(-1, 1))])

    D = np.concatenate([sex, age, hand], axis=1)

    clf = DISVM(kernel='linear', C=0.01, lambda_=1000, solver='osqp')
    clf.fit(X[:n_train, :], X[n_train:, :], y_train, D[:n_train, :], D[n_train:, :])

    pred = clf.predict(X[n_train:, :])
    score = clf.decision_function(X[n_train:, :])

    for i in range(pred.shape[0]):
        if pred[i] == -1:
            pred[i] = 0

    np.savetxt('%s/%s' % (outdir, 'classification.txt'), pred, fmt='%0.7f', delimiter='\t')
    np.savetxt('%s/%s' % (outdir, 'score.txt'), score, fmt='%0.7f', delimiter='\t')


