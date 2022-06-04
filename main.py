import random, arff, sys, pandas as pd

from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupShuffleSplit
import tensorflow as tf
from functools import reduce

from config import get_extractor, get_classifier, get_sampler
from pipeline import Pipeline
from pipeline.extractor.xbi_extractor import XBIExtractor
from pipeline.loader.arff_loader import ArffLoader

random.seed(42)

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6

gpus = tf.config.experimental.list_physical_devices('GPU')

config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)


if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=(1024 * 4))])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

    except RuntimeError as e:
        print(e)

device_name = tf.test.gpu_device_name()

if device_name != '/device:GPU:0':
    print('ATENTION: GPU device not found. CPU will be used!')
else:
    print('Found GPU at: {}'.format(device_name))

def main(extractor_name, class_attr, sampler_name, n_splits, path='.'):
    #print("[INFO] main...")
    (extractor, features, nfeatures, max_features) = get_extractor(extractor_name, class_attr)
    sampler = get_sampler(sampler_name)

    print('running --- %s-%s...' % (extractor_name, class_attr))

    extractor_pipeline = Pipeline([ArffLoader(), XBIExtractor(features, class_attr), extractor])

    #file_name = '%s/data/danilo/dataset.32x32.%s.arff' % (path, class_attr)

    file_name = '/home/danilo/Mestrado/JANEIRO_2022/xbi-detection-V2/xbi-detection/data/danilo/dataset.32x32.internal.arff'

    print('[INFO] dataset: %s ' % file_name)

    data = extractor_pipeline.execute(open(file_name).read())

    X, y, attributes, features = data['X'], data['y'], [ attr[0] for attr in data['attributes'] ], data['features']
    groups = list(data['data'][:, attributes.index('URL')])

    cv = GroupShuffleSplit(n_splits=n_splits, random_state=42)

    cache = {}

    for classifier_name in ['svm', 'nn', 'dt', 'randomforest', 'cnn']:  # in ['cnn']:

        if classifier_name == 'cnn':
            if extractor_name != 'image_diff_extractor':
                continue

            X = X.reshape((X.shape[0], 32, 32, 1))
            print('CNN reshaped data to 32x32...')

        rankings, fscore, precision, recall, roc, train_fscore = [], [], [], [], [], []
        approach = '%s-%s-%s' % (extractor_name, classifier_name, class_attr)
        print('running --- %s...' % (approach))
        #gridsearch = get_classifier(classifier_name, nfeatures, max_features)

        for ind, (train_index, test_index) in enumerate(cv.split(X, y, groups)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            groups_train = [i for i in groups if groups.index(i) in train_index]

            gridsearch = get_classifier(classifier_name, nfeatures, max_features)

            '''if ind in cache:
                print('Sample recovered from cache...')
                (X_samp, y_samp, groups_samp) = cache[ind]
            else:
                print('Running sampling strategy: fit_resample')
                X_samp, y_samp = sampler.fit_resample(X_train, y_train)
                print('Running sampling strategy: groups_samp')
                groups_samp = [groups_train[X_train.tolist().index(row)] for row in X_samp.tolist() ]
                cache[ind] = (X_samp, y_samp, groups_samp)'''

            X_samp, y_samp, groups_samp = X_train, y_train, groups_train

            print('Model trainning with: X (%s)' % (str(X_samp.shape)))

            gridsearch.fit(X_samp, y_samp, groups=groups_samp)
            print('Model trained with fscore %s, and params %s ' % (str(gridsearch.best_score_), str(gridsearch.best_params_)))

            #selector = gridsearch.best_estimator_.named_steps['selector']
            #rankings.append(selector.get_support(indices=False))

            if (classifier_name == 'svm'):
                model = CalibratedClassifierCV(gridsearch.best_estimator_)
                model.fit(X_samp, y_samp)
            else:
                model = gridsearch

            y_pred = model.predict_proba(X_train)
            probability = y_pred[:, list(
                gridsearch.best_estimator_.classes_).index(1)]

            precision2, recall2, threasholds = metrics.precision_recall_curve(
                    y_train, probability)
            best_f = 0
            threashold = 0
            for i in range(len(precision2)):
                new_fscore = (2 * precision2[i] * recall2[i]) / (precision2[i] + recall2[i])
                if new_fscore > best_f:
                    best_f = new_fscore
                    threshold = threasholds[i]

            print('Model training F-Score with selected threshold: %f' % (metrics.f1_score(y_train, [ 0 if prob < threshold else 1 for prob in probability])))

            train_fscore.append(metrics.f1_score(y_train, [ 0 if prob < threshold else 1 for prob in probability]))
            y_pred = model.predict_proba(X_test)
            probability = y_pred[:, list(
                gridsearch.best_estimator_.classes_).index(1)]

            y_threashold = [ 0 if y < threshold else 1 for y in probability]
            print('Model tested with F-Score: %f' % (metrics.f1_score(y_test, y_threashold)))
            fscore.append(metrics.f1_score(y_test, y_threashold))
            precision.append(metrics.precision_score(y_test, y_threashold))
            recall.append(metrics.recall_score(y_test, y_threashold))
            roc.append(metrics.roc_auc_score(y_test, y_threashold))


        print('Features: ' + str(features))
        print('X dimensions:' + str(X.shape))
        print('Test     ROC: %f' % (reduce(lambda x,y: x+y, roc) / n_splits))
        print('Test      F1: %f' % (reduce(lambda x,y: x+y, fscore) / n_splits))
        print('Test      F1: ' + str(fscore))
        print('Train     F1: ' + str(train_fscore))
        print('Test      Precision: ' + str(precision))
        print('Test      Recall: ' + str(recall))

        try:
            fscore_csv = pd.read_csv('results/fscore-%s.csv' % (class_attr), index_col=0)
            precision_csv = pd.read_csv(
                    'results/precision-%s.csv' % (class_attr), index_col=0)
            recall_csv = pd.read_csv('results/recall-%s.csv' % (class_attr), index_col=0)
            roc_csv = pd.read_csv('results/roc-%s.csv' % (class_attr), index_col=0)
        except:
            fscore_csv = pd.DataFrame()
            precision_csv = pd.DataFrame()
            recall_csv = pd.DataFrame()
            roc_csv = pd.DataFrame()

        fscore_csv.loc[:, approach] = fscore
        precision_csv.loc[:, approach] = precision
        recall_csv.loc[:, approach] = recall
        roc_csv.loc[:, approach] = roc

        fscore_csv.to_csv('results/fscore-%s.csv' % (class_attr))
        precision_csv.to_csv('results/precision-%s.csv' % (class_attr))
        recall_csv.to_csv('results/recall-%s.csv' % (class_attr))
        roc_csv.to_csv('results/roc-%s.csv' % (class_attr))

        try:
            features_csv = pd.read_csv('results/features-%s.csv' % (class_attr), index_col=0)
        except:
            features_csv = pd.DataFrame(columns=features)

        features_len = features_csv.shape[1]
        print(features)
        print(features_len)
        #print(rankings)

        '''if extractor_name == 'browserninja2':
            for i in range(len(rankings)):
                features_csv.loc[
                        '%s-%d' % (classifier_name, (i + features_len)), :] = rankings[i]
            features_csv.to_csv('results/features-%s.csv' % (class_attr))'''


if __name__ == '__main__':
    #assert len(sys.argv) == 4, 'The script requires 3 parameters: feature extractor (browserbite|crosscheck|browserninja1|browserninja2|cnn), type of xbi (internal|external) and sampler strategy (none|tomek|near|repeated|rule|random)'
    #extractor_name = sys.argv[1]
    #class_attr = sys.argv[2]
    #sampler_name = sys.argv[3]
    #n_splits = 24

    #  ===== Browserbite  =====
    #f = open('/home/danilo/Mestrado/JANEIRO_2022/xbi-detection-V2/xbi-detection/results/browserbite-internal.results.txt', 'w')
    #f = open('/home/danilo/Mestrado/JANEIRO_2022/xbi-detection-V2/xbi-detection/results/browserbite-external.results.txt', 'w')
    #extractor_name = 'browserbite'

    #  ===== CrossCheck  =====
    #f = open('/home/danilo/Mestrado/JANEIRO_2022/xbi-detection-V2/xbi-detection/results/crosscheck-internal.results.txt', 'w')
    #f = open('/home/danilo/Mestrado/JANEIRO_2022/xbi-detection-V2/xbi-detection/results/crosscheck-external.results.txt', 'w')
    #extractor_name = 'crosscheck'

    #  ===== BrowserNinja 1  =====
    #f = open('/home/danilo/Mestrado/JANEIRO_2022/xbi-detection-V2/xbi-detection/results/browserninja1-internal.results.txt', 'w')
    f = open('/home/danilo/Mestrado/JANEIRO_2022/xbi-detection-V2/xbi-detection/results/browserninja1-external.results.txt', 'w')
    extractor_name = 'browserninja1'

    #  ===== CNN  =====
    #f = open('/home/danilo/Mestrado/JANEIRO_2022/xbi-detection-V2/xbi-detection/results/image_diff_extractor-internal.results.txt', 'w')
    # f = open('/home/danilo/Mestrado/JANEIRO_2022/xbi-detection-V2/xbi-detection/results/image_diff_extractor-external.results.txt', 'w')
    #extractor_name = 'image_diff_extractor'

    class_attr = 'external'
    sampler_name = 'none'
    n_splits = 10
    sys.stdout = f
    main(extractor_name, class_attr, sampler_name, n_splits)
    sys.out = sys.stdout

