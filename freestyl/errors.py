
class BadParameter(ValueError):
    """ Error thrown when a parameter is bad """


class PipelineNotBuilt(Exception):
    """ Error raised when a pipeline has not been built """


class NoTargetsError(ValueError):
    """ Error raised when a df does not include the target property """
