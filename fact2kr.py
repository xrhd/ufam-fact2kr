import numpy as np
import pandas as pd
import json
from itertools import combinations
from functools import reduce

def get_data_from_path(path):
    with open(path, 'r') as f:
        return json.load(f)


def fact2kr(data: dict) -> dict:
    """ factorial2kr
    Perform factorial 2^k r analysis of the effects.

    Args:
        data (dict): eg. ./data/test.json

    Returns:
        dict: [description]
    """
    res = {
        i: None for i in [
            'sse',
            'sst',
            'variable_effects'
        ]
    }

    def data2df(data):
        df = pd.DataFrame([
            map(int, value.split(' ')) for value in data['values']
        ])
        df.columns = data['vars']

        var_variation = [
            ''.join(variation)
            for i in range(2, len(data['vars'])+1)
            for variation in combinations(data['vars'], i)
        ]

        for variation in var_variation:
            aux = [
                df[var] for var in variation
            ]
            df[variation] = list(reduce(lambda a, b: a*b, aux))

        df['I'] = [1 for _ in range(len(data['values']))]

        # order of cols
        cols = df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        df = df[cols]

        # y col
        df['y_mean'] = [np.mean(ys) for ys in data['values'].values()]

        return df


    # frame format

    df = data2df(data)

    total = df.y_mean.dot(df).to_frame().T
    total.pop('y_mean')
    total.index = ['total']

    factor = 2**len(data['vars'])
    q = total / factor
    q.index = [f'total/{factor}']

    print(pd.concat([
        df,
        total,
        q
    ]))

    # res
    error_table = np.array([
        ys
        for ys in data['values'].values()
    ])
    error_table = (error_table.T - np.array(df.y_mean)).T
    res['sse'] = (error_table ** 2).sum().round(4)

    q.pop('I')
    partial_effects = factor * q**2

    res['sst'] = float(partial_effects.sum(1).round(4))
    res['variable_effects'] = ((partial_effects.iloc[0] / res['sst']).round(4)).to_dict()


    return res

if __name__ == "__main__":
    path = './data/data.json'
    data = get_data_from_path(path)
    res = fact2kr(data)
    print(json.dumps(res, indent=4))
