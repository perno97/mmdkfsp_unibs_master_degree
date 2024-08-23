import logging
from logging import LogRecord


class CustomFormatter(logging.Formatter):
    """
    Custom formatter for logging, overriding the default one.
    Credit: https://alexandra-zaharia.github.io/posts/make-your-own-custom-color-formatter-with-python-logging/
    :ivar fmt: The format string to use for the message
    :ivar FORMATS: Dictionary that maps the log level to the color
    """

    grey = '\x1b[38;21m'
    blue = '\x1b[38;5;39m'
    yellow = '\x1b[38;5;226m'
    red = '\x1b[38;5;196m'
    bold_red = '\x1b[31;1m'
    reset = '\x1b[0m'

    def __init__(self, fmt):
        """
        Constructor of the class.
        """
        super().__init__()
        self.fmt = fmt
        self.FORMATS = {
            logging.DEBUG: self.grey + self.fmt + self.reset,
            logging.INFO: self.blue + self.fmt + self.reset,
            logging.WARNING: self.yellow + self.fmt + self.reset,
            logging.ERROR: self.red + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset
        }

    def format(self, record: LogRecord) -> str:
        """
        The record’s attribute dictionary is used as the operand to a string formatting operation.
        Returns the resulting string.
        Before formatting the dictionary, a couple of preparatory steps are carried out.
        :param record: The LogRecord to be formatted.
        :type record: LogRecord
        :return: The formatted message
        :rtype: str
        """
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
