# -*- coding: utf-8 -*-
""" 
@Time    : 2022/9/19 14:42
@Author  : HCF
@FileName: utils.py
@SoftWare: PyCharm
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import os
import struct

def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()

def read_consistency_graph(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.int32)
    return array

def make_counter_map(consistency_graph_array, height, width):
    counter = np.zeros((height, width))
    i=0
    while i < consistency_graph_array.shape[0]:
        x = consistency_graph_array[i]
        y = consistency_graph_array[i+1]
        count = consistency_graph_array[i+2]
        i = i + 3 + count
        counter[int(y),int(x)] = count
    return counter

# plt.figure()
# plt.subplot(1,2,1)
# z=read_consistency_graph('/home/yswang/Downloads/colmapsh_test/stereo/consistency_graphs/rect_012_4_r5000.png.geometric.bin')
# counter = make_counter_map(z,512,640)
# plt.imshow(counter)
# plt.subplot(1,2,2)
# depth = read_array('/home/yswang/Downloads/colmapsh_test/stereo/depth_maps/rect_012_4_r5000.png.geometric.bin')
# plt.imshow(depth)


def write_array(array, path):
    """
    see: src/mvs/mat.h
        void Mat<T>::Write(const std::string& path)
    """
    assert array.dtype == np.float32
    if len(array.shape) == 2:
        height, width = array.shape
        channels = 1
    elif len(array.shape) == 3:
        height, width, channels = array.shape
    else:
        assert False

    with open(path, "w") as fid:
        fid.write(str(width) + "&" + str(height) + "&" + str(channels) + "&")

    with open(path, "ab") as fid:
        if len(array.shape) == 2:
            array_trans = np.transpose(array, (1, 0))
        elif len(array.shape) == 3:
            array_trans = np.transpose(array, (1, 0, 2))
        else:
            assert False
        data_1d = array_trans.reshape(-1, order="F")
        data_list = data_1d.tolist()
        endian_character = "<"
        format_char_sequence = "".join(["f"] * len(data_list))
        byte_data = struct.pack(endian_character + format_char_sequence, *data_list)
        fid.write(byte_data)

