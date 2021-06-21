#
# \file lazy_file.py
#
# \brief LazyFile updates the target file only when the content is changed
#        in order to avoid generating new cutlass kimpls each time cmake is called
#

import io
import os

class LazyFile:
    def __init__(self, filename):
        self.filename = filename
        self.buffer = io.StringIO()

    def write(self, data):
        self.buffer.write(str(data))

    def close(self):
        if os.path.isfile(self.filename):
            old_data = open(self.filename).read()
        else:
            old_data = ""
        new_data = self.buffer.getvalue()
        if old_data != new_data:
            with open(self.filename, "w") as f:
                f.write(new_data)
