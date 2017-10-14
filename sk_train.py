from PIL import Image

import sys
import os
import re
import math
import numpy
import pickle


def polynomial_kernel(x, y):
    return math.pow(numpy.dot(x, y) + 1, 4)


def sk_algorithm(X, I_positive, I_negative, epsilon, max_updates, model_file_name):
    # initialization
    lam = determine_lambda(X, I_positive, I_negative)
    X_dash = get_X_dash(X, lam, I_positive, I_negative)

    # alpha initialization as per slides
    alpha = [0 for i in xrange(len(X))]
    alpha[I_positive[0]] = 1
    alpha[I_negative[0]] = 1

    x_dash_i1 = X_dash[I_positive[0]]
    x_dash_j1 = X_dash[I_negative[0]]

    A = polynomial_kernel(x_dash_i1, x_dash_i1)
    B = polynomial_kernel(x_dash_j1, x_dash_j1)
    C = polynomial_kernel(x_dash_i1, x_dash_j1)
    D = [polynomial_kernel(X_dash[i], x_dash_i1) for i in xrange(len(X_dash))]
    E = [polynomial_kernel(X_dash[i], x_dash_j1) for i in xrange(len(X_dash))]

    current_updates = 0

    while current_updates < max_updates:
        # calculate m_t
        const = math.sqrt(A+B-2*C)
        m_t = sys.maxint
        t = 0
        for i in xrange(len(X_dash)):
            m_temp = 0.0
            if i in I_positive:
                m_temp = (D[i]-E[i]+B-C)/const
            else:
                m_temp = (E[i]-D[i]+A-C)/const

            if m_temp < m_t:
                m_t = m_temp
                t = i

        # check for stop condition
        if (const - m_t) < epsilon:
            break

        # adaptation step
        if t in I_positive:
            q = min(1.0, (A - D[t] + E[t] - C) / (A + polynomial_kernel(X_dash[t], X_dash[t]) - 2 * (D[t] - E[t])))
            for i in I_positive:
                d_t = 1 if i == t else 0
                alpha[i] = (1-q) * alpha[i] + q * d_t
            A = A * math.pow((1 - q), 2) + 2 * (1 - q) * q * D[t] + (q * q) * polynomial_kernel(X_dash[t], X_dash[t])
            C = (1 - q) * C + q * E[t]
            D = [(1 - q) * D[i] + q * polynomial_kernel(X_dash[i], X_dash[t]) for i in xrange(len(X_dash))]
        else:
            q = min(1.0, (B - E[t] + D[t] - C) / (B + polynomial_kernel(X_dash[t], X_dash[t]) - 2 * (E[t] - D[t])))
            for i in I_negative:
                d_t = 1 if i == t else 0
                alpha[i] = (1 - q) * alpha[i] + q * d_t
            B = B * math.pow((1 - q), 2) + 2 * (1 - q) * q * E[t] + (q * q) * polynomial_kernel(X_dash[t], X_dash[t])
            C = (1 - q) * C + q * D[t]
            E = [(1 - q) * E[i] + q * polynomial_kernel(X_dash[i], X_dash[t]) for i in xrange(len(X_dash))]

        current_updates += 1
        print current_updates

    alpha_dict = {}
    for i in xrange(len(alpha)):
        if alpha[i] != 0:
            alpha_dict.update({i : alpha[i]})
    svm_model = {'m_positive' : get_centroid(X, I_positive), 'm_negative' : get_centroid(X, I_negative), 'lamda' : lam, 'A' : A, 'B': B, 'alphas' : alpha_dict}
    with open(model_file_name, "wb") as f:
        pickle.dump(svm_model, f)
    f.close()
    return


def get_centroid(X, indices):
    if indices is not None:
        Z = []
        for index in indices:
            Z.append(X[index])
        X = numpy.array(Z)
    centroid = []
    length = X.shape[0]
    for i in xrange(X[0].size):
        centroid.append(numpy.sum(X[:, i])/length)
    return numpy.array(centroid)


def get_X_dash(X, lam, I_positive, I_negative):
    m = get_centroid(X, None)
    m_positive = get_centroid(X, I_positive)
    m_negative = get_centroid(X, I_negative)
    X_dash = []
    for x in xrange(len(X)):
        if x in I_positive:
            X_dash.append(numpy.array((lam * X[x]) + ((1-lam) * m_positive)))
        else:
            X_dash.append(numpy.array((lam * X[x]) + ((1 - lam) * m_negative)))
    return numpy.array(X_dash)


def convert_images_to_numpy_arrays(X, flodername):
    X_2D_vector = numpy.array([numpy.array(Image.open(flodername + "/" + filename).convert('L'), 'f') for filename in X])
    X_flatten = []
    for x in X_2D_vector:
        b = x.flatten()
        X_flatten.append(b)
    return numpy.array(X_flatten)


def determine_lambda(X, I_positive, I_negative):
    m_positive = get_centroid(X, I_positive)
    m_negative = get_centroid(X, I_negative)
    r = numpy.linalg.norm(m_positive - m_negative)

    r_positive = 0
    for i in I_positive:
        r_temp = numpy.linalg.norm(X[i] - m_positive)
        if r_temp >= r_positive:
            r_positive = r_temp

    r_negative = 0
    for i in I_positive:
        r_temp = numpy.linalg.norm(X[i] - m_negative)
        if r_temp >= r_negative:
            r_negative = r_temp

    return r / (r_positive + r_negative)


def get_class_label(filename):
    words = filename.split("_")
    return words[1].split(".")[0]


class NoDataException(Exception):
    pass

''' python sk_train.py epsilon max_updates class_letter model_file_name train_folder_name '''
if __name__ == "__main__":
    epsilon = float(sys.argv[1])
    max_updates = int(sys.argv[2])
    class_letter = sys.argv[3]
    model_file_name = sys.argv[4]
    train_folder_name = sys.argv[5]

    X = []
    I_positive = []
    I_negative = []

    try:
        index = 0
        for filename in os.listdir(train_folder_name):
            if not filename.endswith(".png"):
                raise NoDataException("Invalid file format")
            else:
                regex = r"([0-9]+)_[O,P,W,Q,S].png$"
                if re.match(regex, filename) is None:
                    raise NoDataException("Invalid file format")

                X.append(filename)
                if get_class_label(filename) == class_letter:
                    I_positive.append(index)
                else:
                    I_negative.append(index)
                index += 1

    except NoDataException:
        print "NO DATA"

    X_vector = convert_images_to_numpy_arrays(X, train_folder_name)
    sk_algorithm(X_vector, I_positive, I_negative, epsilon, max_updates, model_file_name)
