from imblearn.under_sampling import TomekLinks, ClusterCentroids, NearMiss, RandomUnderSampler, RepeatedEditedNearestNeighbours, NeighbourhoodCleaningRule, OneSidedSelection
from imblearn.over_sampling import SMOTE
from sklearn import tree, svm, ensemble
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV,GroupShuffleSplit
from sklearn.pipeline import Pipeline as Pipe
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier

from classifier.classifier_cnn import get_params_grid_cnn, model_function_cnn
from pipeline.extractor.crosscheck_extractor import CrossCheckExtractor
from pipeline.extractor.browserbite_extractor import BrowserbiteExtractor
from pipeline.extractor.browserninja import *
from pipeline.extractor.browserninja.font_family_extractor import FontFamilyExtractor
from pipeline.extractor.browserninja.image_moments_extractor import ImageMomentsExtractor
from pipeline.extractor.browserninja.relative_position_extractor import RelativePositionExtractor
from pipeline.extractor.image_diff_extractor import ImageDiffExtractor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, AvgPool2D

input_shape = None

def get_extractor(name, class_attr):
    #print("[INFO] extractor...")
    features = []
    max_features = []
    nfeatures=[]
    extractor = None
    extractors = []

    if name == 'browserbite':
        #print("[INFO] extractor browserbite...")
        extractor = BrowserbiteExtractor(class_attr)
    elif name == 'crosscheck':
        #print("[INFO] extractor crosscheck...")
        extractor = CrossCheckExtractor(class_attr)
    elif name == 'browserninja1':
        #print("[INFO] extractor browserninja1...")
        if (class_attr == 'internal'):
            extractors = [
                    ComplexityExtractor(),
                    ImageComparisonExtractor(),
                    SizeViewportExtractor(),
                    VisibilityExtractor(),
                    PositionViewportExtractor(),
                ]
        else:
            extractors = [
                    ComplexityExtractor(),
                    SizeViewportExtractor(),
                    VisibilityExtractor(),
                    PositionViewportExtractor(),
                ]
        extractor = BrowserNinjaCompositeExtractor(class_attr,
            extractors=extractors)
    elif name == 'browserninja2':
        #print("[INFO] extractor browserninja2...")
        max_features = [5, 10, 15]
        nfeatures = [5, 10, 15]
        extractors = []
        if (class_attr == 'internal'):
            features = [ 'emd', 'ssim', 'mse', 'ncc', 'sdd', 'missmatch', 'psnr',
                         'base_centroid_x', 'base_centroid_y', 'base_orientation',
                         'target_centroid_x', 'target_centroid_y',
                         'target_orientation',
                         'base_bin1', 'base_bin2', 'base_bin3',
                         'base_bin4', 'base_bin5', 'base_bin6',
                         'base_bin7', 'base_bin8', 'base_bin9', 'base_bin10',
                         'target_bin1', 'target_bin2', 'target_bin3',
                         'target_bin4', 'target_bin5', 'target_bin6',
                         'target_bin7', 'target_bin8', 'target_bin9', 'target_bin10' ]
            extractors = [
                ComplexityExtractor(),
                ImageComparisonExtractor(),
                SizeViewportExtractor(),
                VisibilityExtractor(),
                PositionViewportExtractor(),
                RelativePositionExtractor(),
                PlatformExtractor(),
                ImageMomentsExtractor()
            ]
        else:
            extractors = [
                ComplexityExtractor(),
                SizeViewportExtractor(),
                VisibilityExtractor(),
                PositionViewportExtractor(),
                RelativePositionExtractor(),
                PlatformExtractor(),
            ]
        extractor = BrowserNinjaCompositeExtractor(class_attr,
            extractors=extractors)
    elif name == 'image_diff_extractor':
        #print("[INFO] extractor image_diff...")
        extractor = ImageDiffExtractor(class_attr)

    return (extractor, features, nfeatures, max_features)


def get_classifier(classifier_name, nfeatures, max_features):
    score_funcs = [f_classif]
    if len(nfeatures) > 0:
        score_funcs = [f_classif, mutual_info_classif]
    if classifier_name == 'randomforest':
        model = Pipe([
            ('preprocessor', StandardScaler()),
            ('selector', SelectKBest(f_classif)),
            ('classifier', ensemble.RandomForestClassifier())])
        classifier = GridSearchCV(model, {
                'selector__k': nfeatures + ['all'],
                'selector__score_func': score_funcs,
                'classifier__n_estimators': [10, 20],
                'classifier__criterion': ["gini", "entropy"],
                'classifier__max_depth': [10, 20],
                'classifier__min_samples_split': [6, 10],
                'classifier__min_samples_leaf': [3, 5],
                'classifier__max_features': ['auto'], #max_features + ['auto'],
                'classifier__class_weight': ['balanced'], #[None, 'balanced']
            }, cv=GroupShuffleSplit(n_splits=2, random_state=42),
            scoring='f1', error_score=0, verbose=0)

    elif classifier_name == 'dt':
        model = Pipe([
            ('preprocessor', StandardScaler()),
            ('selector', SelectKBest(f_classif)),
            ('classifier', tree.DecisionTreeClassifier())])
        classifier = GridSearchCV(model, {
                'selector__k': nfeatures + ['all'],
                'selector__score_func': score_funcs,
                'classifier__criterion': ["gini", "entropy"],
                'classifier__max_depth': [10, 20],
                'classifier__min_samples_split': [6, 10],
                'classifier__min_samples_leaf': [3, 5],
                'classifier__class_weight': ['balanced'], #[None, 'balanced']
                'classifier__max_features': ['auto'], # max_features + ['auto'],
            }, cv=GroupShuffleSplit(n_splits=2, random_state=42),
            scoring='f1', error_score=0, verbose=0)

    elif classifier_name == 'cnn':
        print('[INFO] CNN classifier...')
        #cnn = CnnClassifier(32)

        '''model = Pipe([
            #('preprocessor', StandardScaler()),
            #('selector', SelectKBest(f_classif)),
            ('classifier', KerasClassifier(build_fn=model_function_cnn, verbose=1))])'''

        #model = KerasClassifier(build_fn=model_function_cnn, verbose=1)
        model = KerasClassifier(build_fn=model_function_cnn, verbose=0)

        classifier = GridSearchCV(estimator=model,
                                  param_grid=get_params_grid_cnn(),
                                  cv=GroupShuffleSplit(n_splits=2, random_state=42),
                                  scoring='f1', error_score=0, verbose=0)

    elif classifier_name == 'svm':
        #model = Pipe([
        #    ('selector', SelectKBest(f_classif)),
        #    ('classifier', svm.SVC(probability=True))])
        model = Pipe([
            ('preprocessor', StandardScaler()),
            ('selector', SelectKBest(f_classif)),
            ('classifier', svm.LinearSVC(random_state=42))])
        classifier = GridSearchCV(model, {
                'selector__k': nfeatures + ['all'],
                'selector__score_func': score_funcs,
                #'classifier__kernel': ['linear', 'rbf'], #'poly', 'sigmoid'],
                #'classifier__degree': [2, 3],
                #'classifier__coef0': [0, 10, 100],
                'classifier__C': [1, 10, 100],
                'classifier__tol': [0.001, 0.1, 1],
                'classifier__dual': [False],
                'classifier__class_weight': ['balanced'], #[None, 'balanced']
                'classifier__max_iter': [10000]
            }, cv=GroupShuffleSplit(n_splits=2, random_state=42),
            scoring='f1', error_score=0, verbose=0)
    else:
        model = Pipe([
            ('preprocessor', StandardScaler()),
            ('selector', SelectKBest(f_classif)),
            ('classifier', MLPClassifier())])
        classifier = GridSearchCV(model, {
                'selector__k': nfeatures + ['all'],
                'selector__score_func': score_funcs,
                'classifier__hidden_layer_sizes': [10, 20, 30],
                #'classifier__activation': ['identity', 'logistic', 'tanh', 'relu'],
                'classifier__activation': ['tanh', 'relu'],
                'classifier__solver': ['adam'], #'lbfgs', 'sgd', 'adam'],
                'classifier__alpha': [0.0001, 0.01, 0.1],
                'classifier__max_iter': [10000],
                #'classifier__learning_rate': ['constant', 'invscaling', 'adaptive'],
                'classifier__random_state': [42]
            }, cv=GroupShuffleSplit(n_splits=2, random_state=42),
            scoring='f1', error_score=0, verbose=0)

    return classifier


class NoneSampler:
    def fit_resample(self, X, y):
        return X, y

samplers = {
    'none': NoneSampler(),
    'tomek': TomekLinks(),
    'near': NearMiss(sampling_strategy=0.1, version=2),
    'repeated': RepeatedEditedNearestNeighbours(),
    'rule': NeighbourhoodCleaningRule(threshold_cleaning=0.1),
    'random': RandomUnderSampler(sampling_strategy=0.1, random_state=42),
    'one_sided': OneSidedSelection()
}

def get_sampler(sampler_name):
    return samplers[sampler_name]


'''def model_function_cnn(neurons = 1):
    model = Sequential()

    image_width = 32
    image_height = 32

    model.add(Dense(neurons, activation='relu'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(image_width, image_height, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Número de saídas do classificador

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # print(model.summary())

    return model'''

