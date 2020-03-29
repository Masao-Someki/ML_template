# custom log utils class

class CustomLogClass(object):
    def __init__(self, name, logger, writer):
        self.name = name
        self.logger = logger
        self.writer = writer

    def info(self, text):
        self.logger.info(text)

    def wav(self):
        # function to save wav
        pass
    
    def figure(self):
        # function to log figures like loss.
        pass
