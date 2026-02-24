class NextStatNlpError(RuntimeError):
    pass


class MissingBackendDependency(NextStatNlpError):
    pass


class ExtractionFailed(NextStatNlpError):
    pass

