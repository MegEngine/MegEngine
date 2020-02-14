%pythoncode {
    @property
    def shape(self):
        return self._get_shape()

    @property
    def dtype(self):
        return self._get_dtype()

    def read(self, size):
        """read raw data from the input file

        :param size: number of bytes to be read
        :type size: :class:`int`
        :return: bytes
        """
        return self._read(size)

}
