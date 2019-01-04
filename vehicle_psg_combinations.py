# -*- coding: utf-8 -*-
# vehicle_psg_combinations.py

# This file is used to generate all possibilities of vehicle-passenger combinations
# with a given number of passengers and vehicles in a custom format.
# The result is then saved to a json file under /benchmark/json_customize directory.

import json
import os
import re
from _ctypes import PyObj_FromPtr
from src import BASE_DIR, utils


def generate_combinations(psg_list, vehicle_number):
    """Returns a list of all unique vehicle_number-partitions of `psg_list`.

    Each partition is a list of parts, and each part is a tuple.

    The parts in each individual partition will be sorted in shortlex
    order (i.e., by length first, then lexicographically).

    The overall list of partitions will then be sorted by the length
    of their first part, the length of their second part, ...,
    the length of their last part, and then lexicographically.
    """

    n = len(psg_list)
    groups = []  # a list of lists, currently empty

    def generate_partitions(i):
        if i >= n:
            yield list(map(tuple, groups))
        else:
            if n - i > vehicle_number - len(groups):
                for group in groups:
                    group.append(psg_list[i])
                    yield from generate_partitions(i + 1)
                    group.pop()

            if len(groups) < vehicle_number:
                groups.append([psg_list[i]])
                yield from generate_partitions(i + 1)
                groups.pop()

    result = generate_partitions(0)

    # Sort the parts in each partition in shortlex order
    result = [sorted(ps, key = lambda p: (len(p), p)) for ps in result]
    # Sort partitions by the length of each part, then lexicographically.
    result = sorted(result, key = lambda ps: (*map(len, ps), ps))
    # print(result)
    return result


class NoIndent(object):
    """ Wrapper class to override default JSONEncoder formatting of Python lists and tuples
        which are converted into JSON arrays."""
    def __init__(self, obj):
        self.obj = obj

    def __repr__(self):
        return repr(self.obj)


class MyEncoder(json.JSONEncoder):
    FORMAT_SPEC = '@@{}@@'  # to convert NoIndent object id's in a unique string pattern
    obj_id_pattern = re.compile(FORMAT_SPEC.format(r'(\d+)'))  # regex: r'@@(\d+)@@'

    @staticmethod
    def di(obj_id):
        """ Inverse of built-in id() function.
        """
        return PyObj_FromPtr(obj_id)

    def default(self, obj):
        if isinstance(obj, NoIndent):
            return self.FORMAT_SPEC.format(id(obj))
        else:
            return super(MyEncoder, self).default(obj)

    def iterencode(self, obj, **kwargs):
        for encoded in super(MyEncoder, self).iterencode(obj, **kwargs):
            # check for list and tuple value that was turned into NoIndent instance
            match = self.obj_id_pattern.search(encoded)
            if match:
                id = int(match.group(1))  # turn it into list object for formatting
                list_obj = list(self.di(int(id)).obj)
                encoded = encoded.replace(
                            '"{}"'.format(self.FORMAT_SPEC.format(id)), repr(list_obj))
            yield encoded


def write_to_json(psg_list, vehicle_number):
    result = generate_combinations(psg_list, vehicle_number)
    data_struct = {
        '[(vehicle1),(vehicle2),(vehicle3),(vehicle4)]':[NoIndent(elem) for elem in result]
        # '[(vehicle)]':[NoIndent(elem) for elem in result]
    }
    jsonDataDir = os.path.join(BASE_DIR, 'benchmark', 'json_customize')
    jsonFilename = 'vehicle_psg_combinations_psg16.json'
    jsonPathname = os.path.join(jsonDataDir, jsonFilename)
    print('Write to file: %s' % jsonPathname)
    utils.makeDirsForFile(pathname=jsonPathname)
    with open(jsonPathname, 'w') as f:
        # for group in generate_combinations(psg_list, vehicle_number):
            # print(group)
        json.dump(data_struct, f, cls=MyEncoder, indent=4)


psg_list = range(16)
# psg_list = [0, 1, 2]
# psg_list = [0, 1, 2, 3]
# psg_list = [0, 1, 2, 3, 4]
vehicle_number = 4
# vehicle_number = 1
write_to_json(psg_list, vehicle_number)
