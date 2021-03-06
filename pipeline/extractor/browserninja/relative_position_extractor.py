import np

class RelativePositionExtractor(object):

    def execute(self, arff_data, attributes, X):
        nrows = len(X) if len(X) > 0 else 1

        base_x = np.array(arff_data['data'][:, attributes.index('baseX')], dtype='float64')
        base_y = np.array(arff_data['data'][:, attributes.index('baseY')], dtype='float64')
        base_height = np.array(arff_data['data'][:, attributes.index('baseHeight')], dtype='float64')
        base_width = np.array(arff_data['data'][:, attributes.index('baseWidth')], dtype='float64')
        target_x = np.array(arff_data['data'][:, attributes.index('targetX')], dtype='float64')
        target_y = np.array(arff_data['data'][:, attributes.index('targetY')], dtype='float64')
        target_height = np.array(arff_data['data'][:, attributes.index('targetHeight')], dtype='float64')
        target_width = np.array(arff_data['data'][:, attributes.index('targetWidth')], dtype='float64')

        base_parent_x = np.array(arff_data['data'][:, attributes.index('baseParentX')], dtype='float64')
        base_parent_y = np.array(arff_data['data'][:, attributes.index('baseParentY')], dtype='float64')
        target_parent_x = np.array(arff_data['data'][:, attributes.index('targetParentX')], dtype='float64')
        target_parent_y = np.array(arff_data['data'][:, attributes.index('targetParentY')], dtype='float64')

        base_p_sibling_x = np.array(arff_data['data'][:, attributes.index('basePreviousSiblingLeft')], dtype='float64')
        base_p_sibling_y = np.array(arff_data['data'][:, attributes.index('basePreviousSiblingTop')], dtype='float64')
        target_p_sibling_x = np.array(arff_data['data'][:, attributes.index('targetPreviousSiblingLeft')], dtype='float64')
        target_p_sibling_y = np.array(arff_data['data'][:, attributes.index('targetPreviousSiblingTop')], dtype='float64')

        base_n_sibling_x = np.array(arff_data['data'][:, attributes.index('baseNextSiblingLeft')], dtype='float64')
        base_n_sibling_y = np.array(arff_data['data'][:, attributes.index('baseNextSiblingTop')], dtype='float64')
        target_n_sibling_x = np.array(arff_data['data'][:, attributes.index('targetNextSiblingLeft')], dtype='float64')
        target_n_sibling_y = np.array(arff_data['data'][:, attributes.index('targetNextSiblingTop')], dtype='float64')

        X.append(np.abs((base_x - base_parent_x) - (target_x - target_parent_x)) / np.minimum(target_width, base_width))
        X.append(np.abs((base_y - base_parent_y) - (target_y - target_parent_y)) / np.minimum(target_height, base_height))

        X.append(np.abs((base_x - base_p_sibling_x) - (target_x - target_p_sibling_x)) /
                np.minimum(target_width, base_width))
        X.append(np.abs((base_y - base_p_sibling_y) - (target_y - target_p_sibling_y)) /
                np.minimum(target_height, base_height))

        X.append(np.abs((base_x - base_n_sibling_x) - (target_x - target_n_sibling_x)) /
                np.minimum(target_width, base_width))
        X.append(np.abs((base_y - base_n_sibling_y) - (target_y - target_n_sibling_y)) /
                np.minimum(target_height, base_height))

        arff_data['features'] = arff_data['features'] + ['parent_x', 'parent_y',
                                 'previous_sibling_x', 'previous_sibling_y',
                                 'next_sibling_x', 'next_sibling_y']

        return X
