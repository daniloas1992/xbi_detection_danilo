import np

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

class FontFamilyExtractor():

    def execute(self, arff_data, attributes, X):

        base_fonts = arff_data['data'][:, attributes.index('baseFontFamily')]
        one_hot = OneHotEncoder(handle_unknown='ignore', sparse=False)
        label = LabelEncoder()
        label_encoded = label.fit_transform(base_fonts)
        encoded_base_fonts = one_hot.fit_transform(label_encoded.reshape(-1,1))

        target_fonts = arff_data['data'][:, attributes.index('targetFontFamily')]
        enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
        label = LabelEncoder()
        label_encoded = label.fit_transform(target_fonts)
        encoded_target_fonts = enc.fit_transform(label_encoded.reshape(-1,1))

        X_list = X
        if (encoded_base_fonts.shape[1] == 1):
            encoded_base_fonts = [encoded_base_fonts.reshape(-1).tolist()]
        else:
            encoded_base_fonts = np.array(encoded_base_fonts).T.tolist()
        for font_column in encoded_base_fonts:
            X_list.append(font_column)
        if (encoded_target_fonts.shape[1] == 1):
            encoded_target_fonts = [encoded_target_fonts.reshape(-1).tolist()]
        else:
            encoded_target_fonts = np.array(encoded_target_fonts).T.tolist()
        for font_column in encoded_target_fonts:
            X_list.append(font_column)

        return X_list
