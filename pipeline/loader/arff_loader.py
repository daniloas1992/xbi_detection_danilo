import arff, np

class ArffLoader():

    def __init__(self):
        pass

    def execute(self, data):
        print('[INFO] ArffLoader...')
        result = arff.load(data)
        result['data'] = np.array(result['data'])
        return result
