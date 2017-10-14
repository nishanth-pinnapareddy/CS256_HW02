import sys
import pickle
import os
import re
import numpy

from sk_train import NoDataException, get_class_label, convert_images_to_numpy_arrays, polynomial_kernel


def label_from_model_file_name(model_file_name):
    words = model_file_name.split("_")
    if words[0] == 'plus':
        return 'P'
    elif words[0] == 'circle':
        return 'O'
    elif words[0] == 'star':
        return 'S'
    elif words[0] == 'waves':
        return 'W'
    elif words[0] == 'square':
        return 'Q'
    else:
        return None


def label_from_test(foldername):
    test_labels = []
    for filename in os.listdir(foldername):
        if('P' in filename):
            test_labels.append('P')
        elif('W' in filename):
            test_labels.append('W')
        elif('Q' in filename):
            test_labels.append('Q')
        elif('S' in filename):
            test_labels.append('S')
        elif('O' in filename):
            test_labels.append('O')
    return test_labels

def get_X_dash(X_train, svm_model, y):
    X_dash_train = []
    lam = svm_model.get('lamda')
    for x in xrange(len(X_train)):
        if y[x] == 1:
            X_dash_train.append(numpy.array((lam * X_train[x]) + ((1 - lam) * svm_model.get('m_positive'))))
        else:
            X_dash_train.append(numpy.array((lam * X_train[x]) + ((1 - lam) * svm_model.get('m_negative'))))
    return numpy.array(X_dash_train)

# python svm_model_tester.py model_file_name train_folder_name test_folder_name
if __name__ == "__main__":
    model_file_name = sys.argv[1]
    train_folder_name = sys.argv[2]
    test_folder_name = sys.argv[3]

    svm_model = None
    with open(model_file_name, "rb") as f:
        svm_model = pickle.load(f)
    f.close()

    y = []
    X_train = []
    X_test = []
    try:
        for filename in os.listdir(train_folder_name):
            if not filename.endswith(".png"):
                raise NoDataException("Invalid file format")
            else:
                regex = r"([0-9]+)_[O,P,W,Q,S].png$"
                if re.match(regex, filename) is None:
                    raise NoDataException("Invalid file format")

                X_train.append(filename)
                if get_class_label(filename) == label_from_model_file_name(model_file_name):
                    y.append(1)
                else:
                    y.append(-1)

    except NoDataException:
        print "NO TRAINING DATA"

    try:
        for filename in os.listdir(test_folder_name):
            if not filename.endswith(".png"):
                raise NoDataException("Invalid file format")
            else:
                regex = r"([0-9]+)_[O,P,W,Q,S].png$"
                if re.match(regex, filename) is None:
                    raise NoDataException("Invalid file format")

                X_test.append(filename)
    except NoDataException:
        print "NO TEST DATA"

    X_train_vector = convert_images_to_numpy_arrays(X_train, train_folder_name)
    X_test_vector = convert_images_to_numpy_arrays(X_test, test_folder_name)

    X_dash_train_vector = get_X_dash(X_train_vector, svm_model, y)

    alpha_dict = svm_model.get('alphas')
    A = svm_model.get('A')
    B = svm_model.get('B')

    test_labels = label_from_test(test_folder_name)
    label = label_from_model_file_name(model_file_name)

    total_test_data = len(X_test_vector)
    fn = 0.0
    tp = 0.0
    fp = 0.0
    for j in xrange(len(X_test_vector)):
        x = X_test_vector[j]
        sum = 0.0
        for i in xrange(len(X_dash_train_vector)):
            alpha_i = 0.0
            if alpha_dict.get(i) is not None:
                alpha_i = alpha_dict.get(i)
            y_i = y[i]
            sum += (alpha_i * y_i * polynomial_kernel(x, X_dash_train_vector[i]))

        g_of_x = sum + ((B-A)/2)
        if g_of_x < 0 and test_labels[j] == label:
            print str(j+1) + " " + "False Negative"
            fn += 1.0
        elif g_of_x > 0 and test_labels[j] == label or g_of_x < 0 and test_labels[j] != label:
            print str(j+1) + " " + "Correct"
            tp += 1.0
        elif g_of_x  > 0 and test_labels[j] != label:
            print str(j+1) + " " + "False Positive"
            fp += 1.0

    print "Fraction Correct: " + str(tp/total_test_data)
    print "Fraction False Positive: " + str(fp / total_test_data)
    print "Fraction False Negative: " + str(fn / total_test_data)
