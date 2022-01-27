import sys
import math
import pandas as pd


def convert_label():
    with open(
            r'D:\Dataset\Sign Language\Phoenix\phoenix-2014.v3.tar\phoenix2014-release\phoenix-2014-multisigner\annotations\automatic\train.alignment') as f:
        lines = f.readlines()

    class_list = []

    for line in lines:
        split = str.split(line)
        frame_num = split[1]
        # print(frame_num)
        class_list.append(frame_num)

    print(class_list)

    # list(map(lambda x: x, class_list))

    new_class_list = []

    for idx, val in enumerate(class_list):
        if idx < len(class_list) - 1:
            if abs(int(class_list[idx + 1]) - int(val)) == 1:
                class_list[idx + 1] = val
            # print(val)

    for idx, val in enumerate(class_list):
        if idx < len(class_list) - 1:
            if abs(int(class_list[idx + 1]) - int(val)) == 2:
                class_list[idx + 1] = val
        # print(val)

    with open(
            r'D:\Dataset\Sign Language\Phoenix\phoenix-2014.v3.tar\phoenix2014-release\phoenix-2014-multisigner\annotations\automatic\newClass.txt',
            'w') as f:
        for item in class_list:
            f.write("%s\n" % item)


def count_class_count():
    with open(
            r'D:\Dataset\Sign Language\Phoenix\phoenix-2014.v3.tar\phoenix2014-release\phoenix-2014-multisigner\annotations\automatic\newClass.txt') as f:
        lines = f.readlines()

    class_no = -1
    count = 1

    newest_class = []
    newest_class_count = []

    for idx, val in enumerate(lines):
        split = str.split(val)
        class_num = split[0]
        # print(class_num)

        if idx < len(lines) - 1:
            if abs(int(lines[idx + 1]) - int(class_num)) == 0:
                class_no = class_num
                count += 1
            else:
                # print(class_no)
                # print(count)
                newest_class.append(class_no)
                newest_class_count.append(count)
                class_no = -1
                count = 0

    with open(
            r'D:\Dataset\Sign Language\Phoenix\phoenix-2014.v3.tar\phoenix2014-release\phoenix-2014-multisigner\annotations\automatic\classCount.txt',
            'w') as f:
        for idx, val in enumerate(newest_class):
            f.write(f'{val} {newest_class_count[idx]}\n')


def determine_max_length():
    df = pd.read_csv(
        r'D:\Dataset\Sign Language\Phoenix\phoenix-2014.v3.tar\phoenix2014-release\phoenix-2014-multisigner\annotations\automatic\classCount.txt',
        sep='\s+', header=None)

    print(df.head())

    grouped_df = df.groupby(0)

    maximums = grouped_df.max()

    maximums = maximums.reset_index()

    print(maximums.head())

    maximums.to_csv(
        r'D:\Dataset\Sign Language\Phoenix\phoenix-2014.v3.tar\phoenix2014-release\phoenix-2014-multisigner\annotations\automatic\maxLength.txt',
        header=None, index=None, sep=' ', mode='a')


if __name__ == '__main__':
    with open(
            r'D:\Dataset\Sign Language\Phoenix\phoenix-2014.v3.tar\phoenix2014-release\phoenix-2014-multisigner\annotations\automatic\trainingClasses.txt') as f:
        lines = f.readlines()

    signs = []
    class_nums = []

    for idx, val in enumerate(lines):
        if idx > 0:
            split = str.split(val)
            sign_state = split[0]
            class_num = split[1]
            # print(sign_state)

            signs.append(sign_state)
            class_nums.append(class_num)

    print(signs)
    print(class_nums)

    with open(
            r'D:\Dataset\Sign Language\Phoenix\phoenix-2014.v3.tar\phoenix2014-release\phoenix-2014-multisigner\annotations\automatic\maxLength.txt') as f:
        max_lengths = f.readlines()

    class_words = []
    all_lengths = []

    for idx, val in enumerate(max_lengths):
        split = str.split(val)
        class_num = split[0]
        length = split[1]

        for class_no_idx, class_no_val in enumerate(class_nums):
            if int(class_num) == int(class_no_val):
                class_words.append(signs[class_no_idx])
                all_lengths.append(length)

    # new_class = list(map(lambda x: set(x[:-1]), class_words))
    new_class = [a[:-1] for a in class_words]

    print(len(new_class))
    print(len(all_lengths))

    with open(
            r'D:\Dataset\Sign Language\Phoenix\phoenix-2014.v3.tar\phoenix2014-release\phoenix-2014-multisigner\annotations\automatic\glossMaxCount.txt',
            'w') as f:
        for idx, val in enumerate(new_class):
            f.write(f'{val} {all_lengths[idx]}\n')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
