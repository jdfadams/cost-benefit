import numpy as np
import pandas as pd


def compute_cost(df, C_first, C):
    first = df.iloc[1]
    cost_first = first.dot(C_first)
    remaining = df.iloc[1:]
    cost_remaining = remaining.dot(C)
    return pd.concat([pd.DataFrame([cost_first]), cost_remaining])


def compute_states(M, n_initial):

    def all_dead():
        n_dead = V[2]
        return round(n_dead) == n_initial

    V = np.array([n_initial, 0, 0])
    states = [V, ]
    while not all_dead():
        V = np.matmul(M, V)
        states += [V, ]
    df = pd.DataFrame(states, columns=['stable', 'progressed', 'died'])
    return df


def format_usd(s):
    return f'${s:,.2f}'


def run_simulation(M, n, C_first, C):
    df = compute_states(M, n)
    df['cost'] = compute_cost(df, C_first, C)
    return df


def generate_parameters():
    M = np.vstack([
            np.random.dirichlet((10, 1, 1)),
            np.append([0], np.random.dirichlet((15, 1))),
            (0, 0, 1),
    ]).transpose()
    n = 1000
    C_first = np.array([2924.2] * 3)
    C = np.array([1464.26, 4888.24, 0])
    return M, n, C_first, C


def main():
    '''
    P = np.array([[0.8834, 0.1145, 0.0021],
                  [0, 0.9803, 0.0197],
                  [0, 0, 1]]).transpose()

    Q = np.array([[0.9377, 0.061, 0.0013],
                  [0, 0.9803, 0.0197],
                  [0, 0, 1]]).transpose()
    '''

    M, n, C_first, C = generate_parameters()
    df = run_simulation(M, n, C_first, C)

    # Compute costs:
    total = df['cost'].sum()

    # FORMATTING:

    # Round the first three columns:
    df = df.round({'stable': 0, 'progressed': 0, 'died': 0})

    # Format the costs as dollar amounts:
    df['cost'] = df['cost'].map(format_usd)

    print(df)
    print('M:')
    print(M.transpose())
    print('C_first:')
    print(C_first)
    print('C:')
    print(C)
    print('Total:', format_usd(total))


if __name__ == '__main__':
    main()
