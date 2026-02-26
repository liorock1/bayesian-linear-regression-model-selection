import numpy as np
from matplotlib import pyplot as plt
from ex3_utils import BayesianLinearRegression, polynomial_basis_functions

# random seed for reproducibility
np.random.seed(0)


def log_evidence(model: BayesianLinearRegression, X, y):
    """
    Calculate the log-evidence of some data under a given Bayesian linear regression model
    :param model: the BLR model whose evidence should be calculated
    :param X: the observed x values
    :param y: the observed responses (y values)
    :return: the log-evidence of the model on the observed data
    """
    # extract the variables of the prior distribution
    mu = model.mu
    sig = model.cov

    # extract the variables of the posterior distribution
    model.fit(X, y)
    map = model.fit_mu
    map_cov = model.fit_cov

    # calculate the log-evidence
    H = model.h(X)
    sig2 = model.sig
    N = len(y)

    logdet_post = np.linalg.slogdet(map_cov)[1]
    logdet_prior = np.linalg.slogdet(sig)[1]

    resid = y - H @ map

    term_det = 0.5 * (logdet_post - logdet_prior)
    term_prior_quad = -0.5 * (map - mu).T @ model.prec @ (map - mu)
    term_data = -(0.5 / sig2) * (resid.T @ resid)
    term_norm = -0.5 * N * np.log(2 * np.pi * sig2)

    return float(term_det + term_prior_quad + term_data + term_norm)


def main():
    # ------------------------------------------------------ section 2.1
    # set up the response functions
    f1 = lambda x: x ** 2 - 1
    f2 = lambda x: (-x ** 2 + 10 * x ** 3 + 50 * np.sin(x / 6)) / 100
    f3 = lambda x: (0.5 * x ** 6 - 0.75 * x ** 4 + 2.75 * x ** 2) / 50
    f4 = lambda x: 5 / (1 + np.exp(-4 * x)) - (x - 2 > 0) * x
    f5 = lambda x: np.cos(4 * x) + 4 * np.abs(x - 2)
    functions = [f1, f2, f3, f4, f5]

    noise_var = .25
    X = np.linspace(-3, 3, 500)

    degrees = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    # go over each response function and polynomial basis function
    for i, f in enumerate(functions):
        y = f(X) + np.sqrt(noise_var) * np.random.randn(len(X))

        evidences = np.zeros(len(degrees))
        means = []
        stds = []

        for j, d in enumerate(degrees):
            # set up model parameters
            pbf = polynomial_basis_functions(d)
            mean, cov = np.zeros(d + 1), np.eye(d + 1)

            # calculate evidence
            model = BayesianLinearRegression(mean, cov, noise_var, pbf)
            log_ev = log_evidence(model, X, y)

            evidences[j] = log_ev
            means.append(model.predict(X))
            stds.append(model.predict_std(X))

        best_j = int(np.argmax(evidences))
        worst_j = int(np.argmin(evidences))

        best_d, worst_d = degrees[best_j], degrees[worst_j]

        best_mean, best_std = means[best_j], stds[best_j]
        worst_mean, worst_std = means[worst_j], stds[worst_j]

        # plot evidence versus degree and predicted fit
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        fig.suptitle(f'Function {i + 1}\nbest evidence d={best_d}, worst evidence d={worst_d}')

        axes[0].plot(degrees, evidences, 'o-', color='black')

        axes[0].set_xlabel('degree')
        axes[0].set_ylabel('log-evidence')
        axes[0].grid(True)

        axes[1].plot(X, best_mean, lw=2, label='best evidence')
        axes[1].fill_between(X, best_mean - best_std, best_mean + best_std, alpha=.5)
        axes[1].plot(X, worst_mean, lw=2, label='worst evidence')
        axes[1].fill_between(X, worst_mean - worst_std, worst_mean + worst_std, alpha=.5)
        axes[1].scatter(X, y, alpha=0.5, c='black', s=9, linewidths=0)

        axes[1].set_xlabel('$x$')
        axes[1].set_ylabel(r'$f_{\theta}(x)$')
        axes[1].legend()
        axes[1].grid(True)
        plt.show()

    # ------------------------------------------------------ section 2.2
    # load relevant data
    nov16 = np.load('nov162024.npy')
    hours = np.arange(0, 24, .5)
    train = nov16[:len(nov16) // 2]
    hours_train = hours[:len(nov16) // 2]

    # load prior parameters and set up basis functions
    params = np.load('temp_prior.npy')
    mean, cov = params[:, 0], params[:, 1:]

    pbf = polynomial_basis_functions(7)

    noise_vars = np.linspace(.05, 2, 100)
    evidences = np.zeros(len(noise_vars))

    for i, nv in enumerate(noise_vars):
        # calculate the evidence
        model = BayesianLinearRegression(mean, cov, nv, pbf)
        log_ev = log_evidence(model, hours_train, train)
        evidences[i] = log_ev

    best_nv = noise_vars[int(np.argmax(evidences))]

    # plot log-evidence versus amount of sample noise
    plt.figure(figsize=(6, 4))
    plt.plot(noise_vars, evidences, color='black')
    plt.xlabel('sample noise')
    plt.ylabel('log-evidence')
    plt.title(f'Best noise variance: {best_nv:.3f}')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
