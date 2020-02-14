%pythoncode {

__owner_graph = None

@property
def owner_graph(self):
    """get the owner graph; note that a reference would be kept in this var"""
    if self.__owner_graph is None:
        self.__owner_graph = self._get_owner_graph()
    return self.__owner_graph

@property
def id(self):
    """an integer identifier for this opr that is unique in the computing
    graph"""
    return int(self._get_id())

@property
def name(self):
    return self._get_name()

@property
def inputs(self):
    return tuple(self._get_inputs())

@property
def outputs(self):
    return tuple(self._get_outputs())

def __repr__(self):
    return 'Operator(id={},name={})'.format(self.id, self.name)
}
