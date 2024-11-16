############

# utililities for collecting data.
# including:
# - get classes and class inheritance info in a given library;
# - get method info in the given class.
# 

import inspect, pkgutil

def get_inheritance(module, package_name):
    """
    Collect all the classes and their superclasses 
    (collect only the classes and super classes with the specified prefix)
    """
    inheritance_info = []

    # Traverse all the members in the module
    for name, obj in inspect.getmembers(module):
        # Keep the classes (whose name starts with the given prefix)
        if inspect.isclass(obj) and obj.__module__.startswith(package_name):
            # Get super (base) classes (whose name starts with the given prefix)
            base_classes = [
                base for base in obj.__bases__ 
                if base.__module__.startswith(package_name)
            ]
            inheritance_info.append((obj, base_classes))

    return inheritance_info


def get_all_classes_in_package(package, package_name = None, verbose=False):
    """
    Get inheritance info for all the classes in the given package.
    (collect only the classes and super classes with the specified prefix)
    """
    if package_name == None:
        package_name = package.__name__
    inheritance_info = []

    # Traverse all the sub-modules/packages in the given package
    for _, module_name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__+"."): #, package_name + "."):
        try:
            # Import the found classes
            module = __import__(module_name, fromlist="dummy")
            module_inheritance = get_inheritance(module, package_name)
            if module_inheritance:
                inheritance_info.extend(module_inheritance)
            if verbose:
                print(f"VV Successfully process module {module_name}")
        except (Exception, SystemExit) as e:
            # Failed to import classes => error message (verbose) + continue.
            if verbose:
                print(f"   Failed to process module {module_name}: {e}")

    return set([(x, tuple(y)) for x, y in inheritance_info])


def get_class_source_info(cls, verbose=False):
    """
    Retrieve the source file path and line number range of the class
    """
    try:
        source_lines, start_line = inspect.getsourcelines(cls)
        file_path = inspect.getfile(cls)
        end_line = start_line + len(source_lines) - 1
        return file_path, start_line, end_line
    except (TypeError, OSError):
        # may not obtain source code information of some dynamically generated classes
        if verbose:
            print(f"Could not find src of cls {cls.__qualname__} ({cls.__module__}).")
        return None, None, None

def get_method_source_info(method, verbose=False):
    """
    Retrieve the source file path and line number range of the method
    """
    try:
        source_lines, start_line = inspect.getsourcelines(method)
        file_path = inspect.getfile(method)
        end_line = start_line + len(source_lines) - 1
        return file_path, start_line, end_line
    except (TypeError, OSError):
        # may not obtain source code information of some dynamically generated methods
        if verbose:
            print(f"Could not find src of func {method.__qualname__} ({method.__module__}).")
        return None, None, None


def get_inside_and_outside_methods_via_source(cls, verbose=False):
    """
    Retrieve methods newly defined in the class and the methods inherited from the parent class,
    (Determine whether it is defined in the class by comparing the method's src file and line number range.)
    """
    
    class_file, class_start_line, class_end_line = get_class_source_info(cls, verbose)

    inside_methods = set()
    outside_methods = set()

    for name, member in inspect.getmembers(cls, predicate=funcpredicate):
        method_file, method_start_line, method_end_line = get_method_source_info(member, verbose)
        if method_start_line == None:
            continue

        #Determine whether the method is defined in the class (with the same file and line number range within the class definition line number range)
        if (method_file == class_file and 
            class_start_line <= method_start_line <= method_end_line <= class_end_line):
            inside_methods.add(member)
        else:
            outside_methods.add(member)

    return frozenset(inside_methods), frozenset(outside_methods)


def funcpredicate(member):
    return inspect.isfunction(member) or inspect.ismethod(member)


def get_classes_defmethod_mappings(inheritance_info, verbose=False):
    cls2defmethodsAoutmethods = {cls: get_inside_and_outside_methods_via_source(cls, verbose=verbose) for cls, _ in inheritance_info}
    cls2defmethods = {cls: cls2defmethodsAoutmethods[cls][0] for cls, _ in inheritance_info}
    cls2outmethods = {cls: cls2defmethodsAoutmethods[cls][1] for cls, _ in inheritance_info}
    defmethods2cls = {}
    for cls, defmethods in cls2defmethods.items():
        for defmethod in defmethods:
            assert defmethod not in defmethods2cls
            defmethods2cls[defmethod] = cls
    return cls2defmethods, cls2outmethods, defmethods2cls


############

# class content (all members [methods/funcs, other strs, etc.] in a specified class's src code)

from enum import Enum
class ClsContent:
    
    class CntType(Enum):
        METHOD_ORI = 1      # methods with original src code => getsourcecode is usable.
        METHOD_CHANGED = 2  # methods with changed src code  => getsourcecode is NOT usable while other unchanged method features can still be retrieved.
        CHUNK = 3           # other types of content NOT under manipulation, expressed in code strings currently.
    
    _cntType = None
    _cntInst = None
    _cntText  = None
    
    def __init__(self, cntInst, cntType = None, cntText = None):
        if cntType is None and cntText is None:
            if type(cntInst) == str:
                self._cntInst = None
                self._cntType = self.CntType.CHUNK
                self._cntText = cntInst
            else:
                self._cntInst = cntInst
                self._cntType = self.CntType.METHOD_ORI
                self._cntText = inspect.getsource(cntInst)
        elif cntText is None:
            self._cntInst = cntInst
            self._cntType = cntType
            self._cntText = inspect.getsource(cntInst)
        else:
            assert cntInst is not None and cntType is not None and cntText is not None
            self._cntInst = cntInst
            self._cntType = cntType
            self._cntText = cntText
    
    def set_text(self, new_cntText):
        self._cntText = new_cntText
        if self._cntType == self.CntType.METHOD_ORI:
            self._cntType = self.CntType.METHOD_CHANGED
    
    def get_text(self):
        return self._cntText
    
    def get_type(self):
        return self._cntType
    
    def get_inst(self):
        return self._cntInst


# collect content list (a list of (contentval, start loc, end loc)) for a given class
def get_cls_content_list(cls, cls2defmethods, skipwhitespace=False):
    class_file, class_start_line, class_end_line = get_class_source_info(cls)
    
    if class_start_line == None:
        return None
    
    linescontent = {loc: None for loc in range(class_start_line, class_end_line + 1)}
    
    assert cls in cls2defmethods
    
    cls_source_lines, __ = inspect.getsourcelines(cls)
    
    # mark the code lines corresponding to the methods/funcs newly defined in the given class
    for defmethod in cls2defmethods[cls]:
        method_file, method_start_line, method_end_line = get_method_source_info(defmethod)
        for loc in range(method_start_line, method_end_line + 1):
            assert loc in linescontent
            assert linescontent[loc] == None
            linescontent[loc] = defmethod
    
    # mark the other code lines (keep them as string objects)
    chunkcnt, chunkstartflg = -1, False
    for loc in range(class_start_line, class_end_line + 1):
        if linescontent[loc] != None:
            chunkstartflg = False
            continue
        if not chunkstartflg:
            chunkstartflg = True
            chunkcnt += 1
        linescontent[loc] = chunkcnt
        
    assert class_start_line <= class_end_line
    
    contents = []
    lastcontent, last_start_loc, chunkbuff = linescontent[class_start_line], class_start_line, cls_source_lines[0]
    for loc in range(class_start_line + 1, class_end_line + 1):
        if linescontent[loc] == lastcontent:
            if type(linescontent[loc]) == int:
                if (not skipwhitespace) or (len(cls_source_lines[loc - class_start_line].strip()) > 0):
                    chunkbuff += cls_source_lines[loc - class_start_line]
            continue
        # dump a new content.
        if type(lastcontent) == int:
            lastcontent = chunkbuff
        contents.append( (ClsContent(lastcontent), last_start_loc, loc - 1) )
        lastcontent, last_start_loc = linescontent[loc], loc
        chunkbuff = ""
        if (not skipwhitespace) or (len(cls_source_lines[loc - class_start_line].strip()) > 0):
            chunkbuff += cls_source_lines[loc - class_start_line]
        
    if type(lastcontent) == int:
        lastcontent = chunkbuff
    contents.append( (ClsContent(lastcontent), last_start_loc, class_end_line) )
    
    return [ c for c in contents if skipwhitespace == False or c[0].get_text() ]


# dump a content val list into a string
def contentvals2str(contentvals: list, do_norm=False):
    if contentvals is None:
        return None
    if do_norm:
        return norm_code("".join([ contentval.get_text() for contentval in contentvals ]))
    return "".join([ contentval.get_text() for contentval in contentvals ])


############

# Mutation Utilities and Operators

from abc import ABC, abstractmethod

class ClassRefactorOperator(ABC):
    @abstractmethod
    def refactor(
        self, clscontentvals: list, oricls,
    ) -> list:
        """
        Refactor the given class content list
        @param clscontentvals: the original content value list of the given class
        @param oricls: the original class (for tracing necessary materials)
        @return: the refactored class content list
        """
        pass


class ClassNoRefactor(ClassRefactorOperator):
    def refactor(
        self, clscontentvals: list, oricls,
    ) -> list:
        if clscontentvals is None:
            return clscontentvals, False
        clscontentvalsnew = [_ for _ in clscontentvals]
        return clscontentvalsnew, True


import random

# class ClassAllContentShuffle(ClassRefactorOperator):
#     def refactor(
#         self, clscontentvals: list, oricls,
#     ) -> list:
#         if len(get_cls_content_list(cls, cls2defmethods, False)) < 3:
#             return None
        
#         clscontentvalsnew = [_ for _ in clscontentvals]
#         while contentvals2str(clscontentvalsnew) == contentvals2str(clscontentvals):
#             clscontentvals_to_shuffle = clscontentvals[1:]
#             assert len(clscontentvals_to_shuffle) >= 2
#             random.shuffle(clscontentvals_to_shuffle)
#             clscontentvalsnew = [clscontentvalsnew[0]] + clscontentvals_to_shuffle
#         return clscontentvalsnew

class ClassMethodShuffle(ClassRefactorOperator):
    def refactor(
        self, clscontentvals: list, oricls,
    ) -> list:
        if clscontentvals is None:
            return clscontentvals, False
        
        clscontentvalsnew = [_ for _ in clscontentvals]
        midxmethodlist = [(idx, fv) for idx, fv in enumerate(clscontentvals) if fv.get_type() in [ClsContent.CntType.METHOD_CHANGED, ClsContent.CntType.METHOD_ORI]]
        
        if len(midxmethodlist) < 2:
            return clscontentvals, False
        
        midxlist, methodlist = list(zip(*midxmethodlist))
        
        while contentvals2str(clscontentvalsnew).strip() == contentvals2str(clscontentvals).strip():
            methodlist_to_shuffle = [x for x in methodlist]
            assert len(methodlist_to_shuffle) >= 2
            random.shuffle(methodlist_to_shuffle)
            
            for idx, fv in zip(midxlist, methodlist_to_shuffle):
                clscontentvalsnew[idx] = fv
            
        try:
            norm_code(contentvals2str(clscontentvalsnew))
        except Exception as e:
            return clscontentvals, False
    
        return clscontentvalsnew, True


class ClassAppendInheritedMethods(ClassRefactorOperator):
    def refactor(
        self, clscontentvals: list, oricls,
        inheritance_info_eligible_classes_dict, cls2outmethods, defmethods2cls
    ) -> list:
        if clscontentvals is None:
            return clscontentvals, False
        
        eligible_base_classes_set = frozenset( inheritance_info_eligible_classes_dict[oricls] )
        inheritated_methods = [f for f in cls2outmethods[oricls] if f in defmethods2cls and defmethods2cls[f] in eligible_base_classes_set]
        
        if len(inheritated_methods) <= 0:
            return clscontentvals, False
        
        clscontentvalsnew = [_ for _ in clscontentvals]
#         clscontentvalsnew += inheritated_methods[-3:] # add no more than 3 inherited methods
        for f in inheritated_methods[-3:]:
            clscontentvalsnew.append(ClsContent(f))
        assert contentvals2str(clscontentvalsnew) != contentvals2str(clscontentvals)

        try:
            norm_code(contentvals2str(clscontentvalsnew))
        except Exception as e:
#             print(e)
            return clscontentvals, False

        return clscontentvalsnew, True


import ast, textwrap

def get_ast_and_oriindent(source_code):
    
    def _get_original_indent(source_code):
        # Find the indentation of the first non-empty line
        for line in source_code.splitlines():
            if line.strip():
                return line[:len(line) - len(line.lstrip())]
        return ''
    
    # Dedent the source code and calculate the original indent
    dedented_code = textwrap.dedent(source_code)
    
    # Find the original indentation
    original_indent = _get_original_indent(source_code)
    
    try:
        # Parse the source code into an AST
        tree = ast.parse(dedented_code)
    except Exception as e:
#         print("="*20, "Cannot parse:", "="*20)
#         print(sourcecode)
#         print("="*50)
        raise e

    return tree, original_indent

def get_src_from_ast(tree, original_indent):
    cleaned_code = ast.unparse(tree)
    return textwrap.indent(cleaned_code, original_indent)

def norm_code(source_code):
    tree, original_indent = get_ast_and_oriindent(source_code)
    return get_src_from_ast(tree, original_indent)

class ClassClearAllComments(ClassRefactorOperator): 
    # This operator can only be applied at last, since it needs to change code content 
    # and thus it will make all contents into a string instance instead of a series of
    # func/method/strchunks instances.
    
    def refactor(
        self, clscontentvals: list, oricls,
    ) -> list:
        if clscontentvals is None:
            return clscontentvals, False
        
        contentori_str = contentvals2str(clscontentvals)
        contentnew_str = norm_code(contentori_str)
        
        if contentori_str.strip() == contentnew_str.strip():
            return clscontentvals, False
        return [ClsContent(contentnew_str)], True


import utils.refactor as func_refacotr

class ClassApplyMethodOprWholeClass(ClassRefactorOperator):
    # This operator can only be applied at last, since it needs to change code content 
    # and thus it will make all contents into a string instance instead of a series of
    # func/method/strchunks instances.
    
    def refactor(
        self, clscontentvals: list, oricls,
        refactorer,
        rand=False, max_count=1e10,
    ) -> list:
        
        if clscontentvals is None:
            return clscontentvals, False
        
        contentori_str = contentvals2str(clscontentvals, do_norm=True)
        
        tree, original_indent = get_ast_and_oriindent(contentori_str)
        
        count = -1
        try:
            tree, count = refactorer.refactor(tree, rand=rand, max_count=max_count)
        except Exception as e:
            return clscontentvals, False
        
        contentnew_str = get_src_from_ast(tree, original_indent)
        
        if count <= 0:
            return clscontentvals, False
        
        try:
            norm_code(contentnew_str)
        except Exception as e:
            # print("="*20, "Cannot parse new code:", "="*20)
            # print(contentnew_str)
            # print("-"*50)
            # print("old:")
            # print(contentori_str)
            # print("-"*50)
            # print(e)
            # print("="*50)
            # raise e
            return clscontentvals, False
        
        return [ClsContent(contentnew_str)], True
