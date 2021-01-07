from estimation import *

topic_num = 40
maxit = 10000
NMF_draw_num = 120
eps = 1e-2
noise_scale = 0.0001

def algo1(alpha, eta, draw_num=1000):

    print('Reading cached data')
    td_matrix1_raw = pd.read_excel(os.path.join(MATRIX_PATH,'FOMC1_meeting_matrix_onlyTF.xlsx'), index_col=0, engine='openpyxl')
    td_matrix2_raw = pd.read_excel(os.path.join(MATRIX_PATH,'FOMC2_meeting_matrix_onlyTF.xlsx'), index_col=0, engine='openpyxl')

    sec1_max_HHI = []
    sec1_max_params = []
    sec1_max_bse = []
    sec1_max_tstat = []
    sec1_min_HHI = []
    sec1_min_params = []
    sec1_min_bse = []
    sec1_min_tstat = []

    sec2_max_HHI = []
    sec2_max_params = []
    sec2_max_bse = []
    sec2_max_tstat = []
    sec2_min_HHI = []
    sec2_min_params = []
    sec2_min_bse = []
    sec2_min_tstat = []

    HHI_FOMC1_all = []
    HHI_FOMC2_all = []

    HHI1_reg_params_all = []
    HHI2_reg_params_all = []
    HHI1_bse_all = []
    HHI2_bse_all = []
    for i in range(draw_num):
        print('Drawing posterior number {}'.format(i+1))
        start = timeit.default_timer()
        sec1_max, sec1_min, sec2_max, sec2_min, HHI_FOMC1, HHI_FOMC2, model_FOMC1, model_FOMC2, \
            = estimate_on_posterior(td_matrix1_raw, td_matrix2_raw, topic_num=topic_num,
                                                                       alpha=alpha, eta=eta, init_random_method='gamma',
                                                                       init_param=(-3,1), verbose=False)
        sec1_max_HHI.append(sec1_max[0])
        sec1_max_params.append(sec1_max[1])
        sec1_max_bse.append(sec1_max[2])
        sec1_max_tstat.append(sec1_max[3])

        sec1_min_HHI.append(sec1_min[0])
        sec1_min_params.append(sec1_min[1])
        sec1_min_bse.append(sec1_min[2])
        sec1_min_tstat.append(sec1_min[3])

        sec2_max_HHI.append(sec2_max[0])
        sec2_max_params.append(sec2_max[1])
        sec2_max_bse.append(sec2_max[2])
        sec2_max_tstat.append(sec2_max[3])

        sec2_min_HHI.append(sec2_min[0])
        sec2_min_params.append(sec2_min[1])
        sec2_min_bse.append(sec2_min[2])
        sec2_min_tstat.append(sec2_min[3])
        end = timeit.default_timer()

        HHI_FOMC1_all.append(HHI_FOMC1)
        HHI_FOMC2_all.append(HHI_FOMC2)

        HHI1_reg_params_all.append(model_FOMC1.get_robustcov_results(cov_type=cov_type, maxlags=maxlags).params)
        HHI1_bse_all.append(model_FOMC1.get_robustcov_results(cov_type=cov_type, maxlags=maxlags).bse)
        HHI2_reg_params_all.append(model_FOMC2.get_robustcov_results(cov_type=cov_type, maxlags=maxlags).params)
        HHI2_bse_all.append(model_FOMC2.get_robustcov_results(cov_type=cov_type, maxlags=maxlags).bse)

        print('Finished posterior draw {}. Time: {}'.format(i+1, end-start))

    index = pd.to_datetime(pd.read_excel(os.path.join(UTILFILE_PATH,'separation_rules.xlsx'), engine='openpyxl').iloc[:, 0], format='%Y%m')

    # plot estimation results
    plot_region(np.array(sec1_max_HHI).mean(axis=0), np.array(sec1_min_HHI).mean(axis=0),
                np.array(HHI_FOMC1_all).mean(axis=0), index,'FOMC1', '_algo1')
    plot_region(np.array(sec2_max_HHI).mean(axis=0), np.array(sec2_min_HHI).mean(axis=0),
                np.array(HHI_FOMC2_all).mean(axis=0), index,'FOMC2','_algo1')

    # save regression results
    # params1.to_excel('FOMC1_coef.xlsx')
    # bse1.to_excel('FOMC1_bse.xlsx')
    # params2.to_excel('FOMC2_coef.xlsx')
    # bse2.to_excel('FOMC2_bse.xlsx')
    # tstat1.to_excel('FOMC1_tstat.xlsx')
    # tstat2.to_excel('FOMC2_tstat.xlsx')

    regressors = pd.DataFrame(sec1_min_params).columns

    summary1 = pd.DataFrame(index=regressors)
    HHI1_params_df = pd.DataFrame(HHI1_reg_params_all, columns=regressors)
    HHI1_bse_df = pd.DataFrame(HHI1_bse_all, columns=regressors)

    summary1['Coef'] = HHI1_params_df.mean()
    summary1['Std Error'] = HHI1_bse_df.mean()
    summary1['Coef Min'] = pd.DataFrame(sec1_min_params).mean()
    summary1['Coef Max'] = pd.DataFrame(sec1_max_params).mean()
    summary1['Tstat Min'] = pd.DataFrame(sec1_min_tstat).mean()
    summary1['Tstat Max'] = pd.DataFrame(sec1_max_tstat).mean()

    summary2 = pd.DataFrame(index=regressors)
    HHI2_params_df = pd.DataFrame(HHI2_reg_params_all, columns=regressors)
    HHI2_bse_df = pd.DataFrame(HHI2_bse_all, columns=regressors)

    summary2['Coef'] = HHI2_params_df.mean()
    summary2['Std Error'] = HHI2_bse_df.mean()
    summary2['Coef Min'] = pd.DataFrame(sec2_min_params).mean()
    summary2['Coef Max'] = pd.DataFrame(sec2_max_params).mean()
    summary2['Tstat Min'] = pd.DataFrame(sec2_min_tstat).mean()
    summary2['Tstat Max'] = pd.DataFrame(sec2_max_tstat).mean()

    print('**************************************************************************************')
    print('Result for FOMC1')
    print(summary1.round(4).T.to_latex())
    print('**************************************************************************************')
    print('Result for FOMC2')
    print(summary2.round(4).T.to_latex())

def algo2(alpha, eta, init_random_method='gamma', init_param=(-3,1), verbose=True):
    print('Reading cached data')
    td_matrix1_raw = pd.read_excel(os.path.join(MATRIX_PATH,'FOMC1_meeting_matrix_onlyTF.xlsx'), index_col=0, engine='openpyxl')
    td_matrix2_raw = pd.read_excel(os.path.join(MATRIX_PATH,'FOMC2_meeting_matrix_onlyTF.xlsx'), index_col=0, engine='openpyxl')

    index = pd.to_datetime(pd.read_excel(os.path.join(UTILFILE_PATH,'separation_rules.xlsx'), engine='openpyxl').iloc[:, 0], format='%Y%m')

    stem_num1 = td_matrix1_raw.sum(axis=0)
    stem_num2 = td_matrix2_raw.sum(axis=0)

    td_matrix1 = td_matrix1_raw / td_matrix1_raw.sum(axis=0)
    td_matrix2 = td_matrix2_raw / td_matrix2_raw.sum(axis=0)

    td_matrix1[td_matrix1 == 0] = 1e-12
    td_matrix2[td_matrix2 == 0] = 1e-12

    # make posterior draws
    matrix_FOMC1_all, matrix_FOMC2_all, HHI_FOMC1_all, HHI_FOMC2_all = posterior_draw(topic_num, alpha, eta, 120)
    HHI_FOMC1 = np.array(HHI_FOMC1_all).mean(axis=0)
    HHI_FOMC2 = np.array(HHI_FOMC2_all).mean(axis=0)

    matrix_FOMC2_onlyone = [(x[0] + noise_scale*np.random.random(size=x[0].shape),
                             x[1] + noise_scale*np.random.random(size=x[1].shape)) for x in matrix_FOMC2_all]
    # prior robust LDA
    Herfindahl1, params1, bse1, tstat1 = band(td_matrix1, topic_num, eps, maxit, NMF_draw_num, stem_num1, init_random_method=init_random_method,
                                              init_random_matrix_list=matrix_FOMC1_all,
                                              gamma_param=init_param, verbose=verbose)
    Herfindahl2, params2, bse2, tstat2 = band(td_matrix2, topic_num, eps, maxit, NMF_draw_num, stem_num2, init_random_method=init_random_method,
                                              init_random_matrix_list=matrix_FOMC2_all,
                                              gamma_param=init_param, verbose=verbose)

    # # LDA using Gibbs sampling
    # with open(os.path.join(MATRIX_PATH, 'FOMC1_text_onlyTF.pkl'),'rb') as f:
    #     FOMC1_text = pickle.load(f)
    # with open(os.path.join(MATRIX_PATH, 'FOMC2_text_onlyTF.pkl'),'rb') as f:
    #     FOMC2_text = pickle.load(f)
    #
    # # print('Running Gibbs Sampling LDA')
    # # theta_FOMC1, _, _ = LDA_implementation(FOMC1_text, alpha=alpha, beta=0.025, burning=4000, sample_freq=50, sample_size=80, keep_num=5)
    # # theta_FOMC2, _, _ = LDA_implementation(FOMC2_text, alpha=alpha, beta=0.025, burning=4000, sample_freq=50, sample_size=80, keep_num=5)
    # # HHI_FOMC1 = (theta_FOMC1 ** 2).sum(axis=1)
    # # HHI_FOMC2 = (theta_FOMC2 ** 2).sum(axis=1)
    # # print('Finished')

    plot_region(Herfindahl1.min(axis=0), Herfindahl1.max(axis=0), HHI_FOMC1, index,'FOMC1', '_algo2')
    plot_region(Herfindahl2.min(axis=0), Herfindahl2.max(axis=0), HHI_FOMC2, index,'FOMC2', '_algo2')

    model_FOMC1 = regression_single(HHI_FOMC1, stem_num1)
    model_FOMC2 = regression_single(HHI_FOMC2, stem_num2)

    summary1 = pd.DataFrame(index = model_FOMC1.params.index)
    summary1['Coef Min'] = params1.min(axis = 0)
    summary1['Coef Max'] = params1.max(axis = 0)
    summary1['Tstat Min'] = tstat1.min(axis = 0)
    summary1['Tstat Max'] = tstat1.max(axis = 0)

    summary2 = pd.DataFrame(index=model_FOMC2.params.index)
    summary2['Coef Min'] = params2.min(axis = 0)
    summary2['Coef Max'] = params2.max(axis = 0)
    summary2['Tstat Min'] = tstat2.min(axis = 0)
    summary2['Tstat Max'] = tstat2.max(axis = 0)

    print('**************************************************************************************')
    print('Result for FOMC1')
    print(summary1.round(3).T.to_latex())
    print('**************************************************************************************')
    print('Result for FOMC2')
    print(summary2.round(3).T.to_latex())

if __name__ == '__main__':
    algo2(alpha=1.25, eta=0.025, init_random_method='init_matrix', init_param=(-3,1), verbose=True)