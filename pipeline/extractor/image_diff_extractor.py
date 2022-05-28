import np

from pipeline.extractor.browserninja import PlatformExtractor

class ImageDiffExtractor ():
    def __init__ (self, class_attr):
        self._class_attr = class_attr

    def execute (self, arff_dataset):
        prev_features = (arff_dataset['features'] if 'features' in arff_dataset else [])

        features_diff = []

        for x in range(1, 10, 1):
            features_diff.append('diff_bin{}{}'.format(0, x))
        for x in range(10, 1025, 1):
            features_diff.append('diff_bin{}'.format(x))

        arff_dataset['features'] = prev_features + features_diff

        X_t = (arff_dataset['X'].T.tolist() if 'X' in arff_dataset else [])

        attributes = [attr[0] for attr in arff_dataset['attributes']]
        data = arff_dataset['data']

        for x in range(1, 10, 1):
            X_t.append(np.array(data[:, attributes.index('diff_bin{}{}'.format(0, x))]))
        for x in range(10, 1025, 1):
            X_t.append(np.array(data[:, attributes.index('diff_bin{}'.format(x))]))

        #(PlatformExtractor()).execute(arff_dataset, attributes, X_t) # TODO: quando passa aqui está com 1 coluna a mais da plataforma

        ##

        #X_aray = np.array(X_t, dtype='float64').T # Não da pra fazer reshape aqui, pois está comdimensão de (1025,8388) e teria que estar ao contrário
        #X_reshape = X_aray.reshape((X_aray.shape[0], 32, 32, 1))
        #arff_dataset['X'] = X_reshape

        #X_t = X_t.reshape((X_t.shape[0], 32, 32, 1))
        #X_t_np = np.array(X_t, dtype='float64').T
        #X_t_np = X_t_np.reshape(list(X_t_np.shape) + [1])
        #arff_dataset['X'] = (np.array(X_t, dtype='float64').T).reshape(8388, 32, 32, 1)


        ##

        arff_dataset['X'] = np.array(X_t, dtype='float64').T
        arff_dataset['y'] = np.array(data[:, attributes.index(self._class_attr)], dtype='float64')

        return arff_dataset
