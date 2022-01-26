import numpy as np

FEATURE_PATH = r'D:\Dataset\Sign Language\Phoenix\vac\baseline-18\features'


def check_features():
    # features = np.load(r'./features/01April_2010_Thursday_heute_default-0_features.npy', allow_pickle=True).item()
    # features = np.load(r'./features/01April_2010_Thursday_tagesschau_default-0_features.npy', allow_pickle=True).item()
    # features = np.load(r'./features/01April_2011_Friday_tagesschau_default-2_features.npy', allow_pickle=True).item()
    features = np.load(f'{FEATURE_PATH}/train/01August_2011_Monday_heute_default-0_features.npy',
                       allow_pickle=True).item()

    print(features.get('label').numpy())
    print(features.get('features').numpy())
    print(features.get('label').numpy().shape)
    print(features.get('features').numpy().shape[0])

    # __ON__ LIEB ZUSCHAUER ABEND WINTER GESTERN loc-NORD SCHOTTLAND loc-REGION UEBERSCHWEMMUNG AMERIKA IX


PREPROCESS_PATH = r'D:\Dataset\Sign Language\Phoenix\vac\phoenix2014'


def check_preprocess():
    # gloss_dict = np.load(f'{PREPROCESS_PATH}/gloss_dict.npy', allow_pickle=True).item()
    # print(gloss_dict)
    #
    # # for key, value in gloss_dict.items():
    # #     print(value)
    # #     print(key)
    #
    # i2g_dict = dict((v[0], k) for k, v in gloss_dict.items())
    #
    # g2i_dict = {v: k for k, v in i2g_dict.items()}
    #
    # print(i2g_dict)
    # print(g2i_dict)

    dev_info = np.load(f'{PREPROCESS_PATH}/train_info.npy', allow_pickle=True).item()
    print(dev_info)
    print(len(dev_info))

    # for key, value in dev_info.items():
    #     print(value)


if __name__ == '__main__':
    # check_features()
    check_preprocess()
