import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
import functools

CSV_PATH = r"F:\Dataset\Sign Language\GSL\GSL_continuous\GSL_continuous\merged_continuous.csv"

PREVIEW = False
DEBUG = True
TESTING = False

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    gsl_all = pd.read_csv(CSV_PATH, sep='|', header=None)

    # Creating a Dataframe
    df = pd.DataFrame(gsl_all)

    # show the dataframe
    if DEBUG:
        print(df.head())
        print(df[2].head())

    all_sentences = df[2].unique().tolist()

    sentences = []
    count = 0
    for col in all_sentences:
        words = str(col).split()
        sentences.append(words)
        if DEBUG:
            print(words)
            print(count)

        count += 1

    # print(list(sentences).item)

    # exit()

    # mapped = map(lambda x: set(x), sentences)
    # print(list(mapped))
    # exit()

    target_tokens = list(
        functools.reduce(lambda a, b: a.union(b), list(map(lambda x: set(x), sentences))))
    target_tokens += ['#START', '#END']

    sentences_array = list(map(lambda x: set(x), sentences))
    max_sentence = len(max(sentences_array, key=lambda x: len(x))) + 2
    num_classes = len(target_tokens)

    le = LabelEncoder()
    le.fit(target_tokens)

    len(target_tokens)
    print(num_classes)
    print(target_tokens)

    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(le_name_mapping)
    print(le.transform(sentences[0]))
    print(le.transform(sentences[1]))

    # # iterating the columns
    # for col in gsl_all.row:
    #     print(col)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
