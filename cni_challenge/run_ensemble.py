import os
import scipy
import numpy as np
import pandas as pd
from nilearn.connectome import ConnectivityMeasure
from sklearn.preprocessing import StandardScaler
from TPy.did import DISVM
from sklearn.metrics import accuracy_score, roc_auc_score

train_dir = './inputdir'

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


def run_model(test_dir, outdir, atlas='aal'):
    X_train, pheno_train = get_train_data(atlas=atlas)
    X_test, pheno_test = get_test_data(test_dir, atlas=atlas)
    X_all = X_train + X_test

    y_train = pheno_train['DX'].values
    n_train = len(X_train)
    n_sub = len(X_all)

    n_roi = X_all[0].shape[1]

    X_ = dict()
    measure = ConnectivityMeasure(kind='correlation')
    X_cor = measure.fit_transform(X_all)

    measure = ConnectivityMeasure(kind='correlation', vectorize=True)
    X_['cor'] = measure.fit_transform(X_all)

    measure = ConnectivityMeasure(kind='tangent', vectorize=True)
    X_['tan'] = measure.fit_transform(X_all)
    X_['tancor'] = measure.fit_transform(X_cor)

    measure = ConnectivityMeasure(kind='covariance', vectorize=True)
    X_['cov'] = measure.fit_transform(X_all)

    X_['meanstd'] = np.zeros((n_sub, n_roi*2))
    for i in range(n_sub):
        X_['meanstd'][i, :n_roi] = np.mean(X_all[i], axis=0)
        X_['meanstd'][i, n_roi:] = np.std(X_all[i], axis=0)

    for key in X_:
        scaler = StandardScaler()
        X_[key] = scaler.fit_transform(X_[key])

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

    clf = dict()
    pred_list = []
    score_list = []

    for key in X_:
        Xdata = X_[key]
        # clf[key] = make_pipeline(StandardScaler(), SVC(kernel='linear', max_iter=10000))
        clf[key] = DISVM(kernel='linear', C=0.01, lambda_=100)
        clf[key].fit(Xdata[:n_train, :], Xdata[n_train:, :], y_train, D[:n_train, :], D[n_train:, :])
        pred_list.append(clf[key].predict(Xdata[n_train:, :]).reshape(-1, 1))
        score_list.append(clf[key].decision_function(Xdata[n_train:, :]).reshape(-1, 1))

    preds = np.concatenate(pred_list, axis=1)
    scores = np.concatenate(score_list, axis=1)
    pred = scipy.stats.mode(preds, axis=1)[0].reshape(n_sub-n_train)
    score = np.sum(scores, axis=1)

    for i in range(pred.shape[0]):
        if pred[i] == -1:
            pred[i] = 0

    np.savetxt('%s/%s' % (outdir, 'classification.txt'), pred, fmt='%0.7f', delimiter='\t')
    np.savetxt('%s/%s' % (outdir, 'score.txt'), score, fmt='%0.7f', delimiter='\t')
