import weakref


class HookHandler:
    hook_num = 0

    def __init__(self, source_dict, hook):
        self.id = HookHandler.hook_num
        HookHandler.hook_num += 1
        source_dict[self.id] = hook
        self.source_ref = weakref.ref(source_dict)

    def remove(self):
        source_dict = self.source_ref()
        if source_dict is not None and self.id in source_dict:
            del source_dict[self.id]
