#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 为了保证全局图优化里的 profiling 结果不受到 ci 环境的影响，所以把写死的 profiling 数据存到了 cache 里去，
# 每次跑测试会从内存 cache 里读取 profiling 结果，然后根据 profiling 结果去做全局图优化，这样确保每次运行
# 结果都是一致的。
# ProfilerCache 可以支持把内存中 cache 下来的 profiling 数据 dump 成文件。
# 这个脚本就是用于把 dump 出去的 cache 文件打包成 cache 的头文件，用于测试时读取数据，构建 InMemory 的 ProfilerCache 。
# 如果在 src/gopt/test/layout_transform_pass.cpp 里新添加了全局图优化相关的测试，则需要考虑用这个脚本来
# 更新 cache 头文件中的 profiling 数据。
# 1. 首先将 src/gopt/test/layout_transform_pass.cpp 中的 `#define MGB_WITH_CACHED_TEST 1` 修改为
# `#define MGB_WITH_CACHED_TEST 0`
# 2. 编译megbrain_test，并运行所有全局图优化相关测试：
#    ./megbrain_test --gtest_filter="*LayoutTransform*"
# 3. 用这个脚本把所有的cache文件打包在一起
#    python3 embed_cache.py -o cache_data.h -r -a $(ls /path/to/cache/*.cache)
# 4. 将步骤1中的 define 语句改回原样，这样 profile 过程就会使用 cache 下来的数据。
# 5. 最后可以重新构建一下 megbrain_test ，确保测试结果正确。
import os.path
import logging
import hashlib
import argparse
import struct
import itertools
import sys
import subprocess
import re

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format='%(asctime)-15s %(message)s')

CHAR_MAP = {i: r'{}'.format(i) for i in range(256)}

def _u32(data):
    return struct.unpack('<I', data)[0]

class CacheDataGenerator:
    _cache_files = None

    def __init__(self, cache_files, remove_plat_info=True, append_cache=True):
        self._cache_files = cache_files
        self._remove_plat_info = remove_plat_info
        self._append_cache = append_cache

    def _get_hash(self):
        return _u32(self._hash.digest()[:4])

    def gen_cache_data(self, fpath):
        fname = os.path.basename(fpath)
        with open(fpath, 'rb') as fcache:
            cache_data = fcache.read()
        if self._remove_plat_info:
            for matched in re.finditer(
                rb"(layout_transform_profile:plat=.*);dev=.*;cap=\d.\d",
                cache_data
            ):
                plat_info = matched.group(1)
                cat_info = cache_data[matched.span()[0] - 4: matched.span()[1]]
                cache_data = re.sub(cat_info, struct.pack('I', len(plat_info)) + plat_info, cache_data)
        cache_data = struct.unpack(
            "<{}B".format(len(cache_data)), cache_data)
        ret = list(map(CHAR_MAP.__getitem__, cache_data))
        for i in range(50, len(ret), 50):
            ret[i] = '\n' + ret[i]
        return ','.join(ret)

    def gen_cache_data_header(self, fout, src_map):
        if not self._append_cache:
            fout.write('// generated embed_cache.py\n')
            fout.write('#include <vector>\n')
            fout.write('#include <stdint.h>\n')
        for k, v in sorted(src_map.items()):
            fout.write("""
static const std::vector<uint8_t> {} = {{
""".format(k.replace('.', '_')))
            fout.write('{}'.format(v))
            fout.write('};\n')

    def invoke(self, output):
        logger.info('generate cache_data.h ...')
        fname2cache_data = {}
        for fname in self._cache_files:
            base, ext = os.path.splitext(os.path.basename(fname))
            assert ext == ".cache", "ext: {}, fname {}".format(ext, fname)
            assert base not in fname2cache_data, "duplicated kernel: " + base
            fname2cache_data[base] = self.gen_cache_data(fname)
        if self._append_cache:
            mode = 'a'
        else:
            mode = 'w'
        with open(output, mode) as fout:
            self.gen_cache_data_header(fout, fname2cache_data)
        logger.info('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='embed cubin into cpp source file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output', help='output source file',
                        required=True)
    parser.add_argument(
        "-r",
        "--remove-plat-info",
        action='store_true',
        default=True,
        help="whether remove platform infomation in the cache (default: True)"
    )
    parser.add_argument(
        "-a",
        "--append-cache",
        action='store_true',
        default=True,
        help="whether append the cache (default: True)"
    )
    parser.add_argument('cache', help='cache files to be embedded', nargs='+')
    args = parser.parse_args()
    cache_generator = CacheDataGenerator(args.cache, args.remove_plat_info, 
                                         args.append_cache)
    cache_generator.invoke(args.output)
