class PiiManagerException(Exception):
    def __init__(self, msg, *args):
        super().__init__(msg.format(*args))


class InvArgException(PiiManagerException):
    pass


class PiiUnimplemented(PiiManagerException):
    pass
