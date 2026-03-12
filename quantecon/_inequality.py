"""
Implements inequality and segregation measures such as the Gini coefficient,
Lorenz Curve, Shorrocks Mobility Index, and rank-size distributions.
"""

import numpy as np
from numba import njit, prange


@njit
def lorenz_curve(y):
        """
            Calculates the Lorenz Curve, a graphical representation of the
                distribution of income or wealth.

                    It returns the cumulative share of people (x-axis) and the
                        cumulative share of income earned.

                            Parameters
                                ----------
                                    y : array_like(float or int, ndim=1)
                                            Array of income/wealth for each individual. Unordered or
                                                    ordered is fine.

                                                        Returns
                                                            -------
                                                                cum_people : array_like(float, ndim=1)
                                                                        Cumulative share of people for each person index (i/n)
                                                                            cum_income : array_like(float, ndim=1)
                                                                                    Cumulative share of income for each person index

                                                                                        References
                                                                                            ----------
                                                                                                .. [1] https://en.wikipedia.org/wiki/Lorenz_curve

                                                                                                    Examples
                                                                                                        --------
                                                                                                            >>> a_val, n = 3, 10_000
                                                                                                                >>> y = np.random.pareto(a_val, size=n)
                                                                                                                    >>> f_vals, l_vals = lorenz_curve(y)
                                                                                                                    
                                                                                                                        """
        n = len(y)
        y = np.sort(y)
        s = np.zeros(n + 1)
        s[1:] = np.cumsum(y)
        cum_people = np.zeros(n + 1)
        cum_income = np.zeros(n + 1)
        for i in range(1, n + 1):
                    cum_people[i] = i / n
                    cum_income[i] = s[i] / s[n]
                return cum_people, cum_income


@njit(parallel=True)
def gini_coefficient(y):
        r"""
            Implements the Gini inequality index.

                Parameters
                    ----------
                        y : array_like(float)
                                Array of income/wealth for each individual. Ordered or
                                        unordered is fine.

                                            Returns
                                                -------
                                                    Gini index : float
                                                            The Gini index describing the inequality of the array of
                                                                    income/wealth. Ranges from 0 (perfect equality) to 1
                                                                            (maximum inequality).

                                                                                References
                                                                                    ----------
                                                                                        .. [1] https://en.wikipedia.org/wiki/Gini_coefficient

                                                                                            Examples
                                                                                                --------
                                                                                                    >>> a_val, n = 3, 10_000
                                                                                                        >>> y = np.random.pareto(a_val, size=n)
                                                                                                            >>> gini = gini_coefficient(y)
                                                                                                            
                                                                                                                """
    n = len(y)
    i_sum = np.zeros(n)
    for i in prange(n):
                for j in range(n):
                                i_sum[i] += abs(y[i] - y[j])
                        return np.sum(i_sum) / (2 * n * np.sum(y))


def shorrocks_index(A):
        r"""
            Implements the Shorrocks mobility index.

                Parameters
                    ----------
                        A : array_like(float)
                                Square matrix with transition probabilities (mobility matrix)
                                        of dimension m x m, where entry A[i, j] represents the
                                                probability of transitioning from state i to state j.

                                                    Returns
                                                        -------
                                                            Shorrocks index : float
                                                                    The Shorrocks mobility index calculated as

                                                                            .. math::

                                                                                        s(A) = \frac{m - \sum_j a_{jj} }{m - 1} \in (0, 1)

                                                                                                An index equal to 0 indicates complete immobility (identity
                                                                                                        matrix). An index equal to 1 indicates maximum mobility.
                                                                                                        
                                                                                                            Raises
                                                                                                                ------
                                                                                                                    ValueError
                                                                                                                            If A is not a square matrix.
                                                                                                                            
                                                                                                                                References
                                                                                                                                    ----------
                                                                                                                                        .. [1] Benhabib, Bisin, Luo (2017). Wealth distribution and social
                                                                                                                                               mobility in the US: A quantitative approach. American Economic
                                                                                                                                                      Review, 107(5), 1623-1658.
                                                                                                                                                             https://www.aeaweb.org/articles?id=10.1257/aer.20151684
                                                                                                                                                             
                                                                                                                                                                 Examples
                                                                                                                                                                     --------
                                                                                                                                                                         >>> A = np.array([[0.9, 0.1], [0.4, 0.6]])
                                                                                                                                                                             >>> shorrocks_index(A)
                                                                                                                                                                                 0.5
                                                                                                                                                                                 
                                                                                                                                                                                     """
    A = np.asarray(A)  # Convert to array if not already
    m, n = A.shape
    if m != n:
                raise ValueError('A must be a square matrix')
    diag_sum = np.trace(A)
    return (m - diag_sum) / (m - 1)


def rank_size(data, c=1.0):
        """
            Generate rank-size data corresponding to a distribution.

                This function is useful for visualizing power-law or Pareto-type
                    distributions common in economics (e.g., firm size, city population,
                        income distribution).

                            Parameters
                                ----------
                                    data : array_like
                                            The set of observations (e.g., firm sizes, incomes, city
                                                    populations).
                                                        c : float, optional(default=1.0)
                                                                Restrict the plot to the top (c x 100)% of the distribution.
                                                                        Must be in (0, 1].

                                                                            Returns
                                                                                -------
                                                                                    rank_data : array_like(float, ndim=1)
                                                                                            Rank of each observation (from largest to smallest), starting
                                                                                                    at 1.
                                                                                                        size_data : array_like(float, ndim=1)
                                                                                                                Size data for the top (c x 100)% of the observations, sorted
                                                                                                                        in descending order.
                                                                                                                        
                                                                                                                            Examples
                                                                                                                                --------
                                                                                                                                    >>> y = np.exp(np.random.randn(1000))  # simulate lognormal data
                                                                                                                                        >>> rank_data, size_data = rank_size(y, c=0.85)
                                                                                                                                        
                                                                                                                                            """
    w = - np.sort(- data)        # Reverse sort
    w = w[:int(len(w) * c)]      # extract top (c * 100)%
    rank_data = np.arange(len(w)) + 1
    size_data = w
    return rank_data, size_data
