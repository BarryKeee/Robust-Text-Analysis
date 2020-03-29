from Constant import MATRIX_PATH
import pandas as pd
import os
import pickle
import onlineldavb

def vb_estimate(section, onlyTF=True, K=40, alpha=0.025, eta=0.025, tau=1024, kappa=0.7):
    vocab_1 = pd.read_excel(os.path.join(MATRIX_PATH, section + '_dictionary_meeting{}.xlsx'.format('_onlyTF' if onlyTF else '')), index_col=0).index
    vocab_1 = list(vocab_1)

    with open(os.path.join(MATRIX_PATH, section + '_text{}.pkl'.format('_onlyTF' if onlyTF else '')), 'rb') as f:
        text1 = pickle.load(f)

    text1 = [' '.join(x) for x in text1]

    D = len(text1) # the number of documents

    # initialize online LDA algorithm
    olda = onlineldavb.OnlineLDA(vocab_1, K, D, alpha, eta, tau, kappa)

    # since we have limited amount of text, to ensure convergence we pass in the same
    # text for multiple batches
    (gamma, bound) = olda.update_lambda(text1)

    # posterior mean of theta is gamma_dk / sum_d gamma_dk
    posterior_mean = gamma.T / gamma.sum(axis=1)
    # compute herfindehl of the posterior mean
    return (posterior_mean**2).sum(axis=0),posterior_mean, gamma

if __name__ == '__main__':
    vb_estimate('FOMC1')

