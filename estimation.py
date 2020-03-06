import numpy as np
import numba
import statsmodels.api as sm
import pandas as pd
from Constant import *
import matplotlib.pyplot as plt
import topicmodels
import pickle

topic_num = 40
maxit = 1000
draw_num = 120
eps = 1e-5

def LDA_implementation(text, alpha=1.25, beta=0.025, burning=4000, sample_freq=50, sample_size=80, keep_num=5):
    ldaobj = topicmodels.LDA.LDAGibbs(text, topic_num)
    ldaobj.set_priors(alpha=alpha, beta=beta)
    ldaobj.sample(burning,sample_freq,sample_size)
    ldaobj.samples_keep(5)
    theta = ldaobj.dt_avg()
    B = ldaobj.tt_avg()
    perplexity = ldaobj.perplexity()
    return theta, B, perplexity


@numba.jit()
def NNF(P, k, eps, maxit):
    V, D = P.shape
    W = P

    B = np.random.uniform(1e-5, 1, size=(V, k))
    B = B / B.sum(axis=0)

    Theta = np.random.uniform(1e-5, 1, size=(k, D))
    Theta = Theta / Theta.sum(axis=0)

    KL_div_prev = np.inf
    for i in range(maxit):
        Theta = (Theta / np.dot(B.T, W)) * np.dot(B.T, (W * P) / np.dot(B, Theta))
        B = (B / np.dot(W, Theta.T)) * np.dot((W * P) / np.dot(B, Theta), Theta.T)
        KL_div = KL(W, P, B, Theta)
        if abs(KL_div_prev - KL_div) < eps:
            print("converged after {}".format(str(i)))
            break
        KL_div_prev = KL_div

    B_norm = B / B.sum(axis=0)
    Theta_norm = Theta / Theta.sum(axis=0)
    return B_norm, Theta_norm


@numba.jit()
def KL(W, P, B, Theta):
    P_hat = np.matmul(B, Theta)
    loss = np.sum(W * (P * np.log(P / P_hat) - P + P_hat))

    return loss


def band(P, k, eps, maxit, M, stem_num, cov_type='HAC', maxlags=4):
    record = np.zeros((M, P.shape[1]))
    covariates = pd.read_csv(os.path.join(UTILFILE_PATH,'covariates.csv'))
    covariates['num_stems'] = stem_num
    covariates['Intercept'] = 1
    params = pd.DataFrame(index=range(M),
                          columns=['Transparency', 'Recession', 'EPU', 'Twoday', 'PhDs', 'num_stems', 'Intercept'])
    bse = pd.DataFrame(index=range(M),
                       columns=['Transparency', 'Recession', 'EPU', 'Twoday', 'PhDs', 'num_stems', 'Intercept'])

    for m in range(M):
        print('Trial number {}'.format(str(m)))

        B, Theta = NNF(P, k, eps, maxit)
        HHI = (Theta ** 2).sum(axis=0)
        record[m] = HHI
        model = sm.OLS(HHI, covariates[
            ['Transparency', 'Recession', 'EPU', 'Twoday', 'PhDs', 'num_stems', 'Intercept']].values).fit()

        params.loc[m] = model.get_robustcov_results(cov_type=cov_type, maxlags=maxlags).params
        bse.loc[m] = model.get_robustcov_results(cov_type=cov_type, maxlags=maxlags).bse

    return record, params, bse


def plot_region(Herfindahl, HHI, index, section, dpi=100):

    df = pd.DataFrame(columns=['NMF max', 'NMF min', 'LDA'], index=index)
    df['NMF max'] = Herfindahl.max(axis=0)
    df['NMF min'] = Herfindahl.min(axis=0)
    df['LDA'] = HHI

    plt.plot(df['NMF max'], c='k', ls='--')
    plt.plot(df['NMF min'], c='k', ls='--')
    plt.fill_between(df.index, df['NMF max'], df['NMF min'], color='grey', alpha=0.1)
    LDA, = plt.plot(df['LDA'], c='r', linewidth=2.5)

    plt.title('Herfindahl measure of topic concentration in {}'.format(section))
    # plt.legend(handles = [LDA])
    plt.savefig(os.path.join(PLOT_PATH, 'HHI_{}.eps'.format(section)), format='eps', dpi=dpi)

def regression_single(HHI, stem_num, cov_type='HAC', maxlags=4):

    covariates = pd.read_csv(os.path.join(UTILFILE_PATH,'covariates.csv'))
    covariates['num_stems'] = stem_num
    covariates['Intercept'] = 1
    model = sm.OLS(HHI, covariates[
        ['Transparency', 'Recession', 'EPU', 'Twoday', 'PhDs', 'num_stems', 'Intercept']]).fit()

    return model

def main():

    print('Reading cached data')
    with open(os.path.join(MATRIX_PATH, 'FOMC1_text.pkl'),'rb') as f:
        FOMC1_text = pickle.load(f)
    with open(os.path.join(MATRIX_PATH, 'FOMC2_text.pkl'),'rb') as f:
        FOMC2_text = pickle.load(f)
    td_matrix1_raw = pd.read_excel(os.path.join(MATRIX_PATH,'FOMC1_meeting_matrix.xlsx'), index_col=0)
    td_matrix2_raw = pd.read_excel(os.path.join(MATRIX_PATH,'FOMC2_meeting_matrix.xlsx'), index_col=0)

    print('Running Standard LDA Implementation')
    theta_FOMC1, _, _ = LDA_implementation(FOMC1_text, alpha=1.25, beta=0.025, burning=4000, sample_freq=50, sample_size=80, keep_num=5)
    theta_FOMC2, _, _ = LDA_implementation(FOMC2_text, alpha=1.25, beta=0.025, burning=4000, sample_freq=50, sample_size=80, keep_num=5)
    HHI_FOMC1 = (theta_FOMC1 ** 2).sum(axis=1)
    HHI_FOMC2 = (theta_FOMC2 ** 2).sum(axis=1)
    print('Finished')

    td_matrix1 = td_matrix1_raw / td_matrix1_raw.sum(axis=0)
    td_matrix2 = td_matrix2_raw / td_matrix2_raw.sum(axis=0)

    stem_num1 = td_matrix1_raw.sum(axis=0)
    stem_num2 = td_matrix2_raw.sum(axis=0)

    td_matrix1[td_matrix1 == 0] = 1e-12
    td_matrix2[td_matrix2 == 0] = 1e-12

    # estimate the band of LDA result
    print('Running Robust LDA Algo')
    Herfindahl1, params1, bse1 = band(td_matrix1, topic_num, eps, maxit, draw_num, stem_num1)
    Herfindahl2, params2, bse2 = band(td_matrix2, topic_num, eps, maxit, draw_num, stem_num2)
    print('Finished')

    index = pd.to_datetime(pd.read_excel(os.path.join(UTILFILE_PATH,'separation_rules.xlsx')).iloc[:, 0], format='%Y%m')

    # plot estimation results
    plot_region(Herfindahl1, HHI_FOMC1, index,'FOMC1')
    plot_region(Herfindahl2, HHI_FOMC2, index,'FOMC2')

    # save regression results
    params1.to_excel('FOMC1_coef.xlsx')
    bse1.to_excel('FOMC1_bse.xlsx')
    params2.to_excel('FOMC2_coef.xlsx')
    bse2.to_excel('FOMC2_bse.xlsx')

    model_FOMC1 = regression_single(HHI_FOMC1, stem_num1, cov_type='HAC', maxlags=4)
    model_FOMC2 = regression_single(HHI_FOMC2, stem_num2, cov_type='HAC', maxlags=4)

    summary1 = pd.DataFrame(index = model_FOMC1.params.index)
    summary1['Coef'] = model_FOMC1.params
    summary1['Std Error'] = model_FOMC1.bse
    summary1['Min'] = params1.min(axis = 0)
    summary1['Max'] = params1.max(axis = 0)

    summary2 = pd.DataFrame(index=model_FOMC2.params.index)
    summary2['Coef'] = model_FOMC2.params
    summary2['Std Error'] = model_FOMC2.bse
    summary2['Min'] = params2.min(axis=0)
    summary2['Max'] = params2.max(axis=0)

    print('**************************************************************************************')
    print('Result for FOMC1')
    print(summary1.round(4).T.to_latex())
    print('**************************************************************************************')
    print('Result for FOMC2')
    print(summary1.round(4).T.to_latex())

if __name__ == '__main__':
    main()
