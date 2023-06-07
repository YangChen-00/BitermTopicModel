import logging

def get_logger(log_path):
    logger = logging.getLogger('test')
    # 给logger设置是告诉它要记录哪些级别的日志
    # logger 的级别是先过滤的，所以被 logger 过滤的日志 handler 也是无法记录的，这样就可以只改 logger 的级别而影响所有输出
    logger.setLevel(level=logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    stream_handler = logging.StreamHandler()
    # 给handler设是告诉它要输出哪些级别的日志，相当于进行了两次过滤。
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger