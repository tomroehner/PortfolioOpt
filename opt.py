import numpy as np

from scipy import optimize

import comp
import obj_func


def inverse_volatility(cov_returns: np.ndarray) -> np.ndarray:
    """
    Function for generating the weights for an inverse volatility weighted portfolio

    :param cov_returns:
        Covariance matrix of the stock returns
    :return:
        Array of weights
    """
    var = np.diagonal(cov_returns)
    risk = np.sqrt(var)
    weights = (1/risk)/np.sum(1/risk)
    return weights


def fully_invested_con(portfolio_size: int, relax=(0, 0)):
    A = np.ones((1, portfolio_size))
    lb = 1
    ub = 1
    if relax[0] < 0:
        lb = -np.inf
    elif relax[0] != 0:
        lb = 1 - (10 ** -relax[0])
    if relax[1] < 0:
        ub = np.inf
    elif relax[1] != 0:
        ub = 1 + (10 ** -relax[1])
    return optimize.LinearConstraint(A, lb=lb, ub=ub)


def min_vol(cov_returns, portfolio_size, xinit=None, add_constraints=None, tol=10 ** -10, method='SLSQP', relax=(0, 0),
            maxiter=1000, polish=False, disp=False, workers=1):
    """
    Function for determining the minimum risk portfolio

    :param cov_returns:
        Covariance matrix of the stock returns
    :param portfolio_size:
        Size of the portfolio
    :param xinit:
        Optional initial Value for the optimizer
    :param add_constraints:
        Additional constraints. method 'diff-ev' requires constraints to be formulated as
        scipy.optimize.LinearConstraint or scipy.optimize.NonLinearConstraint objects
    :param tol:
        Tolerance for termination
    :param method:
        Specifies what method should be used to solve the optimization problem. Can be either 'SLSQP' (default) or
        'diff-ev'. The respective scipy implementation is used for both methods.
    :param relax:
        Specifies by what degree the bounds of the fully invested constraint should be relaxed. First value of the Tuple
        affects the lower bound, the second value affects the upper bound. Negative values fully remove the respective
        bound. Relaxing the bounds can help with the performance of the 'diff-ev' method.
    :param maxiter:
        maximum number of iterations that will be performed
    :param polish:
        For method 'diff-ev' only. If True the result is additionally polished via the 'trust-constr' method.
    :param disp:
        For method 'diff-ev' only. If True the evaluated objective function is printed for every iteration.
    :param workers:
        For method 'diff-ev' only. enables parallelization of the algorithm. Supply -1 to use all available cpu cores.
    :return:
        Result of the optimization
    """
    args = (cov_returns, )
    opt = optimizer(
        objective=obj_func.min_variance,
        args=args,
        add_constraints=add_constraints,
        method=method,
        portfolio_size=portfolio_size,
        tol=tol,
        relax=relax,
        maxiter=maxiter,
        polish=polish,
        disp=disp,
        workers=workers,
        xinit=xinit
    )
    return opt


def max_return(mean_returns, portfolio_size, add_constraints=None, tol=10 ** -10, method='SLSQP', relax=(0, 0),
               maxiter=1000, polish=False, disp=False, workers=1):
    """
    Function for determining the portfolio that maximizes the expected return

    :param mean_returns:
        Mean returns for the individual stocks
    :param portfolio_size:
        Size of the portfolio
    :param add_constraints:
        Additional constraints. method 'diff-ev' requires constraints to be formulated as
        scipy.optimize.LinearConstraint or scipy.optimize.NonLinearConstraint objects
    :param tol:
        Tolerance for termination
    :param method:
        Specifies what method should be used to solve the optimization problem. Can be either 'SLSQP' (default) or
        'diff-ev'. The respective scipy implementation is used for both methods.
    :param relax:
        Specifies by what degree the bounds of the fully invested constraint should be relaxed. First value of the Tuple
        affects the lower bound, the second value affects the upper bound. Negative values fully remove the respective
        bound. Relaxing the bounds can help with the performance of the 'diff-ev' method.
    :param maxiter:
        maximum number of iterations that will be performed
    :param polish:
        For method 'diff-ev' only. If True the result is additionally polished via the 'trust-constr' method.
    :param disp:
        For method 'diff-ev' only. If True the evaluated objective function is printed for every iteration.
    :param workers:
        For method 'diff-ev' only. enables parallelization of the algorithm. Supply -1 to use all available cpu cores.
    :return:
        Result of the optimization
    """
    args = (mean_returns, )
    opt = optimizer(
        objective=obj_func.max_expected_return,
        args=args,
        add_constraints=add_constraints,
        method=method,
        portfolio_size=portfolio_size,
        tol=tol,
        relax=relax,
        maxiter=maxiter,
        polish=polish,
        disp=disp,
        workers=workers,
        xinit=None
    )
    return opt


def efficient_return(mean_returns, cov_returns, portfolio_size, target_return, xinit=None, add_constraints=None,
                     tol=10 ** -10, method='SLSQP', relax=(0, 0), maxiter=3000, polish=False, disp=False, workers=1):
    """
    Function for determining the portfolio that achieves a certain target return with minimal risk

    :param mean_returns:
        Mean returns for the individual stocks
    :param cov_returns:
        Covariance matrix of the stock returns
    :param portfolio_size:
        Size of the portfolio
    :param target_return:
        Target daily return the portfolio has to achieve
    :param xinit:
        Optional initial Value for the optimizer
    :param add_constraints:
        Additional constraints. method 'diff-ev' requires constraints to be formulated as
        scipy.optimize.LinearConstraint or scipy.optimize.NonLinearConstraint objects
    :param tol:
        Tolerance for termination
    :param method:
        Specifies what method should be used to solve the optimization problem. Can be either 'SLSQP' (default) or
        'diff-ev'. The respective scipy implementation is used for both methods.
    :param relax:
        Specifies by what degree the bounds of the fully invested constraint should be relaxed. First value of the Tuple
        affects the lower bound, the second value affects the upper bound. Negative values fully remove the respective
        bound. Relaxing the bounds can help with the performance of the 'diff-ev' method.
    :param maxiter:
        maximum number of iterations that will be performed
    :param polish:
        For method 'diff-ev' only. If True the result is additionally polished via the 'trust-constr' method.
    :param disp:
        For method 'diff-ev' only. If True the evaluated objective function is printed for every iteration.
    :param workers:
        For method 'diff-ev' only. enables parallelization of the algorithm. Supply -1 to use all available cpu cores.
    :return:
        Result of the optimization
    """
    if add_constraints is None:
        add_constraints = []

    lc = optimize.LinearConstraint(A=mean_returns, lb=target_return, ub=np.inf)
    add_constraints.append(lc)

    opt = min_vol(
        cov_returns=cov_returns,
        portfolio_size=portfolio_size,
        xinit=xinit,
        add_constraints=add_constraints,
        tol=tol,
        method=method,
        relax=relax,
        maxiter=maxiter,
        polish=polish,
        disp=disp,
        workers=workers
    )

    return opt


def efficient_frontier(mean_returns, cov_returns, portfolio_size, add_constraints=None, tol=10 ** -10, method='SLSQP',
                       relax=(0, 0), maxiter=1000, polish=False, disp=False, workers=1, step_size=0.00001):
    """
    Function for determining the set of efficient portfolios.

    :param mean_returns:
        Mean returns for the individual stocks
    :param cov_returns:
        Covariance matrix of the stock returns
    :param portfolio_size:
        Size of the portfolio
    :param add_constraints:
        Additional constraints. method 'diff-ev' requires constraints to be formulated as
        scipy.optimize.LinearConstraint or scipy.optimize.NonLinearConstraint objects
    :param tol:
        Tolerance for termination
    :param method:
        Specifies what method should be used to solve the optimization problem. Can be either 'SLSQP' (default) or
        'diff-ev'. The respective scipy implementation is used for both methods.
    :param relax:
        Specifies by what degree the bounds of the fully invested constraint should be relaxed. First value of the Tuple
        affects the lower bound, the second value affects the upper bound. Negative values fully remove the respective
        bound. Relaxing the bounds can help with the performance of the 'diff-ev' method.
    :param maxiter:
        maximum number of iterations that will be performed
    :param polish:
        For method 'diff-ev' only. If True the result is additionally polished via the 'trust-constr' method.
    :param disp:
        For method 'diff-ev' only. If True the evaluated objective function is printed for every iteration.
    :param workers:
        For method 'diff-ev' only. enables parallelization of the algorithm. Supply -1 to use all available cpu cores.
    :param step_size:
        Controls the resolution of the Efficient Frontier. Higher values result in fewer iterations.
    :return:
        List containing all efficient portfolio weights
    """
    if add_constraints is None:
        add_constraints = []

    min_return_weights = min_vol(
        cov_returns=cov_returns,
        portfolio_size=portfolio_size,
        add_constraints=add_constraints,
        tol=tol,
        method=method,
        relax=relax,
        maxiter=maxiter,
        polish=polish,
        disp=disp,
        workers=workers
    ).x
    min_exp_return = comp.exp_pf_return(mean_returns, min_return_weights)

    max_return_weights = max_return(
        mean_returns=mean_returns,
        portfolio_size=portfolio_size,
        add_constraints=add_constraints,
        tol=tol,
        method=method,
        relax=relax,
        maxiter=maxiter,
        polish=polish,
        disp=disp,
        workers=workers
    ).x
    max_exp_return = comp.exp_pf_return(mean_returns, max_return_weights)

    target_return = min_exp_return
    it_res = np.repeat(1 / portfolio_size, portfolio_size)
    c = 1
    res = []
    while target_return < max_exp_return:
        print('iteration ', c)
        c += 1
        it_res = efficient_return(
            mean_returns=mean_returns,
            cov_returns=cov_returns,
            portfolio_size=portfolio_size,
            target_return=target_return,
            xinit=it_res,
            add_constraints=add_constraints,
            tol=tol,
            method=method,
            relax=relax,
            maxiter=maxiter,
            polish=polish,
            disp=disp,
            workers=workers,
        ).x
        res.append(it_res)
        target_return += step_size
    return res


def max_sharpe(mean_returns, cov_returns, risk_free_rate, portfolio_size, add_constraints=None, tol=10 ** -4,
               method='SLSQP', relax=(0, 0), maxiter=1000, polish=False, disp=False, workers=1):
    """
    Function for maximizing the sharpe ratio for a given portfolio.

    :param mean_returns:
        Mean returns for the individual stocks
    :param cov_returns:
        Covariance matrix of the stock returns
    :param risk_free_rate:
        Risk free rate
    :param portfolio_size:
        Size of the portfolio
    :param add_constraints:
        Additional constraints. method 'diff-ev' requires constraints to be formulated as
        scipy.optimize.LinearConstraint or scipy.optimize.NonLinearConstraint objects
    :param tol:
        Tolerance for termination
    :param method:
        Specifies what method should be used to solve the optimization problem. Can be either 'SLSQP' (default) or
        'diff-ev'. The respective scipy implementation is used for both methods.
    :param relax:
        Specifies by what degree the bounds of the fully invested constraint should be relaxed. First value of the Tuple
        affects the lower bound, the second value affects the upper bound. Negative values fully remove the respective
        bound. Relaxing the bounds can help with the performance of the 'diff-ev' method.
    :param maxiter:
        maximum number of iterations that will be performed
    :param polish:
        For method 'diff-ev' only. If True the result is additionally polished via the 'trust-constr' method
    :param disp:
        For method 'diff-ev' only. If True the evaluated objective function is printed for every iteration
    :param workers:
        For method 'diff-ev' only. enables parallelization of the algorithm. Supply -1 to use all available cpu cores.
    :return:
        Result of the optimization
    """
    args = (mean_returns, cov_returns, risk_free_rate)
    opt = optimizer(
        objective=obj_func.max_sharpe_ratio,
        args=args,
        add_constraints=add_constraints,
        method=method,
        portfolio_size=portfolio_size,
        tol=tol,
        relax=relax,
        maxiter=maxiter,
        polish=polish,
        disp=disp,
        workers=workers,
        xinit=None
    )
    return opt


def optimizer(objective, args, add_constraints, method, portfolio_size, tol, relax, maxiter, polish, disp, workers,
              xinit):
    if add_constraints is None:
        add_constraints = []

    # fully invested constraint
    lc = fully_invested_con(portfolio_size, relax)
    cons = [lc]
    # add additionally defined constraints
    for c in add_constraints:
        cons.append(c)

    if xinit is None:
        xinit = np.repeat(1 / portfolio_size, portfolio_size)

    bnds = tuple([0, 1] for x in xinit)

    if method.upper() == 'SLSQP':
        opt = optimize.minimize(
            fun=objective,
            args=args,
            bounds=bnds,
            constraints=cons,
            x0=xinit,
            tol=tol,
            method='SLSQP',
            options={'maxiter': maxiter}
        )
    elif method.upper() == 'DIFF-EV':
        opt = optimize.differential_evolution(
            func=objective,
            args=args,
            bounds=bnds,
            constraints=cons,
            x0=xinit,
            tol=tol,
            disp=disp,
            polish=polish,
            maxiter=maxiter,
            workers=workers
        )
    else:
        raise ValueError(f'Selected method {method} is not recognised')

    return opt
