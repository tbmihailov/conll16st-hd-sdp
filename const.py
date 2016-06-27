"""
    Consts implementation as seen shown in http://code.activestate.com/recipes/65207-constants-in-python/?in=user-97991
"""
class _const:
    class ConstError(TypeError): pass
    def __setattr__(self,name,value):
        #if self.__dict__.has_key(name):
            # we do not to raise exception as this constant can be binded in a module that can be loaded from different modules!
            # raise self.ConstError, "Can't rebind const(%s)"%name
        if not self.__dict__.has_key(name):
            self.__dict__[name]=value
import sys
sys.modules[__name__]=_const()

# # that's all -- now any client-code can
# import const
# # and bind an attribute ONCE:
# const.magic = 23
# # but NOT re-bind it:
# const.magic = 88      # raises const.ConstError
# # you may also want to add the obvious __delattr__
