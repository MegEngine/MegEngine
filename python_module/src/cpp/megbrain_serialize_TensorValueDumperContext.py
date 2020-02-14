%pythoncode {
    @property
    def name(self):
        """name of the param

        :type: str
        """
        return str(self._name())

    @property
    def type(self):
        """operator type

        :type: str
        """
        return str(self._type())

    @property
    def value(self):
        """numerical value of the param

        :type: :class:`numpy.ndarray`
        """
        return self._value()

    def write(self, buf):
        """write raw data to the output file

        :param buf: value to be written
        :type buf: :class:`bytes`
        :return: self
        """
        assert type(buf) is bytes, 'bad value: {!r}'.format(type(buf))
        self._write(buf)

    def write_default(self):
        """dump the numerical value in default format

        :return: self
        """
        self._write_default()
        return self

}
