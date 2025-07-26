import traceback
import sys
from logger.custom_logger import CustomLogger
logger=CustomLogger().get_logger(__file__)
class DocumentPortalException(Exception):
    def __init__(self,error_msg:str,error_details:sys):
        _,_,exec_tb=error_details.exc_info()# we can capure all system info
        # exec_tb.tb_frame#line no it will give
        self.file_name=exec_tb.tb_frame.f_code.co_filename
        self.line_no=exec_tb.tb_lineno
        self.error_msg=str(error_msg)
        self.traceback_str="".join(traceback.format_exception(*error_details.exc_info()))

    def __str__(self):
        return f"""Error in [{self.file_name}] at line [{self.line_no}]
        Message: {self.error_msg}
        Traceback:{self.traceback_str}"""

if __name__=="__main__":
    try:
        a=1/0
        print(a)
    except Exception as e:
        app_exe=DocumentPortalException(e,sys)
        logger.error(app_exe)
        raise app_exe