import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


def linear_interpolation(input_, interval):
    input_ = input_[np.lexsort([input_[:, 0], input_[:, 1]])]
    output_ = input_.copy()

    id_pre, f_pre, row_pre = -1, -1, np.zeros((10,))

    for row in input_:
        f_curr, id_curr = row[:2].astype(int)

        if id_curr == id_pre:
            if f_pre + 1 < f_curr < f_pre + interval:
                for i, f in enumerate(range(f_pre + 1, f_curr), start=1):
                    step = (row - row_pre) / (f_curr - f_pre) * i
                    row_new = row_pre + step
                    output_ = np.append(output_, row_new[np.newaxis, :], axis=0)
        else:
            id_pre = id_curr

        row_pre = row
        f_pre = f_curr

    output_ = output_[np.lexsort([output_[:, 0], output_[:, 1]])]

    return output_


def gradient_boosting_smooth(input_, tau):
    output_ = list()
    ids = set(input_[:, 1])

    for id_ in ids:
        tracks = input_[input_[:, 1] == id_]
        t = tracks[:, 0].reshape(-1, 1)
        x = tracks[:, 2].reshape(-1, 1)
        y = tracks[:, 3].reshape(-1, 1)
        w = tracks[:, 4].reshape(-1, 1)
        h = tracks[:, 5].reshape(-1, 1)

        regr = GradientBoostingRegressor(n_estimators=115, learning_rate=0.065, min_samples_split=6)

        regr.fit(t, x[:, 0])
        xx = regr.predict(t)
        regr.fit(t, y[:, 0])
        yy = regr.predict(t)
        regr.fit(t, w[:, 0])
        ww = regr.predict(t)
        regr.fit(t, h[:, 0])
        hh = regr.predict(t)

        output_.extend([[t[i, 0], id_, xx[i], yy[i], ww[i], hh[i], 1, -1, -1, -1] for i in range(len(t))])

    return output_


def gb_interpolation(path_in, path_out, interval, tau):
    input_ = np.loadtxt(path_in, delimiter=',')
    li_result = linear_interpolation(input_, interval)
    gbi_result = gradient_boosting_smooth(li_result, tau)
    np.savetxt(path_out, gbi_result, fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d')
