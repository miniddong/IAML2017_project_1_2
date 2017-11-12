JSON_SEPARATORS = (',', ': ')

class Constant:
    class Project2:
        METADATA_PATH = '../dataset/track_metadata.csv'
        LABEL_COLUMN_NAME = 'listens'
        SAVE_CHECKPOINT_PATH = '../checkpoint'

        def __init__(self):
            pass

    class Data:
        class Mfcc:
            def __init__(self):
                pass

        class ChromaStftHop512:
            FEATURE_NAME = 'chroma_stft'
            IMAGE_HEIGHT = 12
            IMAGE_WIDTH = 2498

            def __init__(self):
                pass

        def __init__(self):
            pass

