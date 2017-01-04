import logging
import os


LOG_FOLDER = "Logs"


class Logger():

    def __init__(self,logger_name=__file__,logger_type='console',logger_level=30, logger_format='[%(asctime)s] %(message)s'):
        if logger_type == 'file':
            self.logger = self._get_file_log(logger_name, logger_level,logger_format)
        elif logger_type == 'console':
            self.logger = self._get_console_log(logger_name,logger_level,logger_format)
        else:
            self.logger = self._get_default_log(logger_name,logger_level,logger_format)

    def message(self, msg, level=30):
        self.logger.log(level,msg)

    def _get_default_log(self,logger_name,logger_level,logger_format):
        logger, formatter = self._init_log(logger_name, logger_level, logger_format)
        if not self._has_file_handler(logger):
            file_handler = logging.FileHandler(os.path.join('Logs', logger_name))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        if not self._has_stream_handler(logger):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        return logger

    def _get_file_log(self,logger_name,logger_level,logger_format):
        logger, formatter = self._init_log(logger_name, logger_level, logger_format)
        if not os.path.isdir(LOG_FOLDER): os.makedirs(LOG_FOLDER)
        if not self._has_file_handler(logger):
            file_handler = logging.FileHandler(os.path.join('Logs',logger_name))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        return logger

    def _get_console_log(self,logger_name, logger_level,logger_format):
        logger, formatter = self._init_log(logger_name, logger_level, logger_format)
        if not self._has_stream_handler(logger):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        return logger

    def _init_log(self,logger_name, logger_level,logger_format):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logger_level)
        formatter = logging.Formatter(logger_format)
        return logger, formatter

    def _has_file_handler(self,logger):
        hasHandler = False
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                hasHandler = True
                break
        return hasHandler

    def _has_stream_handler(self,logger):
        hasHandler = False
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                hasHandler = True
                break
        return hasHandler
