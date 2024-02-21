import os
import time
import re
import numpy as np

#para correr remotamente
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy import stats
# from matplotlib.colors import ListedColormap, BoundaryNorm

from PIL import Image as pil_image
from numbers import Number
from pathlib import Path

def bitificar8(im,desv=1):
    '''Devuelve la imagen pasada a 8 bits. Puede reescalar la desviación.'''
    #normalizo la imagen            
    im -= im.mean()
    im /= (im.std() + 1e-5) #por si tengo cero
    im *= desv
    #la llevo a 8 bits
    im *= 64
    im += 128
    im = np.clip(im, 0, 255).astype('uint8') #corta todo por fuera de la escala
    return im

def normalize(im):
    '''Devuelve la imagen reescalada en el intervalo 0-1'''
    im -= im.min()
    im /= np.ptp(im)
    
    return im

def enzip(*iterables):
    """ Shortcut for enumerating a zipped set of iterables -because I seem to
    use that a lot"""
    return enumerate(zip(*iterables))

def sort_by(what, by):
    """ Sort an iterable by the order corresponding to another iterable."""
    return [x for x, _ in sorted(zip(what, by), key = lambda pair: pair[1])]

def sort_index(iterable):
    """ Get the index array needed to get from the input iterable to a sorted 
    version of the iterable."""
    return sort_by(range(len(iterable)), iterable)

def new_name(name, newformater='_%d'):
    '''
    Returns a name of a unique file or directory so as to not overwrite.
    
    If proposed name existed, will return name + newformater%number.
     
    Parameters:
    -----------
        name : str (path)
            proposed file or directory name influding file extension
        newformater : str
            format to give the index to the new name, esuring a unique name
            
    Returns:
    --------
        name : str
            unique namefile using input 'name' as template
    '''
    
    #if file is a directory, extension will be empty
    base, extension = os.path.splitext(name)
    i = 2
    while os.path.exists(name):
        name = base + newformater%i + extension
        i += 1
        
    return name

def make_dirs_noreplace(dirs_paths):
    """
    Creates a new directory only if that directory doesn't exist already'

    Parameters
    ----------
    dirs_paths : str, path-like object
        The candidate directory to create

    Returns
    -------
    dirs_paths : str, path-like object
        same string or path object

    """
    
    try:
        os.makedirs(dirs_paths)
    except FileExistsError:
        print('While creating ', dirs_paths, 'found it already exists.')

    return dirs_paths

def natural_sort(l): 
    """
    Applies natural sort to a list, returns new list. Natural sort treats 
    strings of numbers as one unit, in contrast to alphanumerical (or 
    lexicographic) ordering. This means that 11 comes before 2 in alphanimeric
    sorting, but 2 before 11 in natural sort.

    Parameters
    ----------
    l : iterable
        iterable to sort

    Returns
    -------
    list
        sorted list from the items in l, naturally sorted
        
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


def find_numbers(string):
    """
    Returns a list of numbers found on a given string
    
    Parameters
    ----------
    string: str
        The string where you search.
    
    Returns
    -------
    list
        A list of numbers (each an int or float).
    
    Raises
    ------
    "There's no number in this string" : TypeError
        If no number is found.
    """
    
    numbers = re.findall(r"[-+]?\d+(?:.\d+)?(?:e[-+]?\d+)?", string)
    
    if not numbers:
        raise ValueError("There's no number in this string")
    
    for i, n in enumerate(numbers):
        if '.' in n:
            numbers[i] = float(n)
        else:
            numbers[i] = int(n) 
    
    return numbers

def first(iterable, condition=bool, default=None):
    """
    Fids the first occurance in iterable that evaluates condition to be True.
    By default, condition is the truth value of the elements of the iterable.
    If no matching value was found, returns the given default.

    Parameters
    ----------
    iterable : iterable
        Iterable to scan for a match.
    condition : callable, optional
        The condition that the first element has to match (evaluate to true). The
        return value of this callable should have a significant truth value. The
        default is bool, which represents the truth value of the elements.
    default : optional
        Return value in case no match was found. The default is None.

    Returns
    -------
        First element in the iterable that evaluated condition to True.

    """
    if not hasattr(condition, '__call__'):
        # this won't catch if the signature is wrong, but whatever
        raise TypeError('Condition has to be callable and take in a single argument')
    
    first_one = next((x for x in iterable if condition(x)), default)
    
    return first_one


def first_index(iterable, condition=bool, default=None):
    """
    Fids the first occurance in iterable that evaluates condition to be True 
    and returns its index. By default, condition is the truth value of the 
    elements of the iterable. If no matching value was found, returns the given 
    default.
    
    NOTE: this will consume the iterator, so if it was a generator, you can't
    use the index with the same object afterwards.

    Parameters
    ----------
    iterable : iterable
        Iterable to scan for a match.
    condition : callable, optional
        The condition that the first element has to match (evaluate to true). The
        return value of this callable should have a significant truth value. The
        default is bool, which represents the truth value of the elements.
    default : optional
        Return value in case no match was found. The default is None.

    Returns
    -------
    int
        Index of the first element evaluating condition to True.

    """
    if not hasattr(condition, '__call__'):
        # this won't catch if the signature is wrong, but whatever
        raise TypeError('Condition has to be callable and take in a single argument')
    
    first_one = next((i for i, x in enumerate(iterable) if condition(x)), default)
    
    return first_one

def find_closest_value(array, value):
    """
    Finds and returns the element of array that's closes (has the least 
    absolute difference) to  value.
    
    NOTE: This makes use of find_points_by_value.

    Parameters
    ----------
    array : array-like
        Array-like that will be used to search for value.
    value : type compatible with array's
        Value to search for.

    Returns
    -------
    same type as array elements
        The element in array that's closest to value.

    """
    return array[find_point_by_value(array, value)]

def find_point_by_value(array, value):
    """
    Fids the closest value in array to value and returns its index. 
    If array is not a numpy array, it converts it to one. If there are multiple
    entries in array that are the same distance from value, it returns the 
    first one. If the array is not 1D, the returned index is converted to N-D 
    using np.unravel_index.

    Parameters
    ----------
    array : array-like
        array-like that will be used to search for value.
    value : type should be compatible with array's
        value to search for.

    Returns
    -------
    inx : int or int array
        The index that realizes the closest value in array. If the array is not
        1D, it returns an index array through np.unravel_index.

    """
    try: 
        inx = np.abs(array-value).argmin()
    except TypeError:
        array = np.array(array)
        inx = np.abs(array-value).argmin()
        
    if array.ndim > 1:
        inx = np.unravel_index(inx, array.shape)
    return inx

def smart_number_format(number, length, strip_trailing_zeros=True, end_int_in_zero=False, force_exp=False, force_float=False):
    """
    Formats a number such that the string version is no longer than length 
    characters, optionally strip trailing zeros if the number had an exact 
    representation shorter than length. in some cases (when adding decimal 
    points or using exponential notation), the length of the string 
    representation maybe very slightly longer than requested.

    Parameters
    ----------
    number : Number
        Number to format.
    length : int
        Length in charaters the resulting string should have.
    strip_trailing_zeros : bool, optional
        Whether to keep or remove trailing zeroes in cases where the original 
        number already had a short representation, i.e. 42.0 becomes '42' if 
        True, but stays '42.0' as long as it fits in the desired length. 
        The default is True.
    end_int_in_zero : bool, optional
        Whether to add a '.0' to the end of an integer, if within length 
        bounds, i.e. '45' becomes '45.0'. The default is False.
    force_exp : bool, optional
        Whether to force the number to be in exponential notation. Only one of 
        force_exp and force_float can be True. Default: False
    force_float : bool, optional
        Whether to force the number to use decimal point notation. Only one of 
        force_exp and force_float can be True. Default: False

    Returns
    -------
    str
        Formated number
    """
    
    if not isinstance(number, Number):
        raise TypeError(f'number {number} should be a number, not a {type(number)}')
    if not isinstance(length, int) or length < 0:
        raise ValueError('length has to ba a postivive integer')
    if force_exp and force_float:
        raise ValueError('only one of force_exp and force_float can be True')
    
    proto_number = str(number)
    # exit now if the length is correct
    if len(proto_number) == length:
        return proto_number
    
    # handle integers ending in .0, if requested
    if end_int_in_zero and isinstance(number, int) and len(proto_number) + 2 <= length:
        fmt = '.1f' if strip_trailing_zeros else f'.{length - len(proto_number)}f'
        return format(number, fmt)
    
    # if number is too short, handle cases
    if len(proto_number) < length:
        #return if it doesn't need to be elongated (don't keep trailing zeros)
        if strip_trailing_zeros:
            return proto_number
        #if if it does, add zeros to short floats and exponential numbers
        elif 'e' in proto_number:
            decimals = length - len(proto_number)
            fmt = f'.{decimals}e'
            return format(number, fmt)
        else:
            decimals = length - len(proto_number)
            fmt = f'.{decimals}f'
            return format(number, fmt)
        
    # from now on, all cases refer to if the number is too long
    
    # use exponential notation if
    if  (number>10**length or        # number is too big
        'e' in proto_number or       # number is already in exponential notation, but too long
        number < 10**(-length+2) ):  # number has too many leading zeros
        decimals = max(0, length-4)
        fmt = f'.{decimals}e'
        return format(number, fmt)
    
    # if all fails and it's too long, truncate it
    if proto_number[length-1]=='.':
        return proto_number[:length+1]
    else:
        return proto_number[:length]

def comma_separated_and(*a, skip_nones=False):
    """
    Parameters
    ----------
    *a : 
        Variables to be put into a string separated by commas and, the last one,
        preceded by an and (no oxford comma!). They don't need to be strings, 
        the function will call the str method to convert them.
    skip_nones : bool
        If true, the concatenated string will ignore all inputs that are None

    Returns
    -------
    str

    """
    if skip_nones:
        a = [e for e in a if e is not None]
    
    if len(a) == 0:
        return ''
    elif len(a) == 1:
        return str(a[0])
    else:
        # we only land here if theres at least two values
        last = str(a[-1])
        start = ', '.join(str(e) for e in a[:-1])
        return start + ' and ' + last

def iter_to_csv(iterable, sep=',', fmt='.2f'):
    """
    Joins an iterable into a string of comma separated values.

    Parameters
    ----------
    iterable : iterable
        An iterable to join into comma separated values.
    sep : str, optional
        Separator, defaults to ','
    fmt : str, optional
        A format string to be used when formating the values of the iterable. 
        It can be any valid format string or the keyword 'smart_n', in which 
        case it will use smart_number_format with length n. The default is 
        '.2f'.

    Returns
    -------
    str
        A string of the values in the iterable, separated by commas.

    """
    if re.fullmatch(r'smart_\d+', fmt):
        n = int(re.findall('\d+', fmt)[0])
        return sep.join(map(lambda s: smart_number_format(s, n) if isinstance(s, Number) else str(s), iterable))
    else:    
        return sep.join(map(lambda s: format(s, fmt) if isinstance(s, Number) else str(s), iterable))

class contenidos(list):
    '''Subclase de lista que toma un directorio y crea una lista de los contenidos.
    
    -----------
    Parámetros:
        carpeta : str
            Directorio cuyo contenido se quiere recolectar. Default: el directorio de trabajo actual,
            dado por os.getcwd()
        full_path : bool
            Decide si los elementos de la lista son el contenido del directorio en cuestión,
            si se le agrega el nombre del directorio: ['capeta/subcarpeta1', 'capeta/subcarpeta2',
            'capeta/archivo.py']. Default: True
        sort : str or None
            Decide la regla que utiliza para ordenar los contenidos de la carpeta. Modos permitidos 
            son "natural", "numeric", "lexicographic", "age", "modified" y None. Default: "natural"
        filter_ext : None, str, iterable
            Filtra los contenidos de la lista por extensión. Si es None, no filta. Para quedarse
            sólo con los directorios, utilizar un string vacío ''. Para quedarse con varios tipos
            de archivos, utilizar una tupla de valores. Las extensiones deben ir con punto: '.jpg', 
            '.py', '.txt', etc.
        filter_re : None, str, re.Pattern
            Filtra los contenidos con regex. Si es None, no filtra. Sólo corrobora que los nombres de
            los archivos matcheen el pattern dado, no la ruta completa. El filtrado por regx sucede
            luego del filtrado por extensiones.

    --------
    Métodos:
        update():
            Actualiza la lista si los contenidos de la carpeta cambiaron
        natural_sort():
            Ordena la lista según orden natural
        age_sort():
            Ordena la lista según última fecha de modificación
        print_orden(only_basename=True):
            Imprime los contenidos de la lista con los índices
        filtered_ext(ext):
            devuelve una NUEVA lista sólo con los elementos cuya extensión es ext
        files():
            devuelve una NUEVA lista sólo con los elementos que son archivos
        directories():
            devuelve una NUEVA lista sólo con los elementos que son directorios
        keep_files():
            modifica esta lista para quedarse sólo con los archivos
        keep_dirs():
            modifica esta lista para quedarse sólo con los directorios

    '''
    attr_list = '_filter_ext', '_filter_re', '_sort_type', '_full_path'
    
    def __init__(self, carpeta=None, full_path=True, sort='natural', filter_ext=None, filter_re=None):
        '''Si full_path=True, los elementos de la lista serán los contenidos de la carpeta
        apendeados con el nombre de la carpeta en sí. Si no, serán sólo los contenidos, en
        cuyo caso habrá algunas funconalidads no disponibles.'''

        self.carpeta = Path.cwd() if carpeta is None else Path(carpeta)
        if not self.carpeta.is_dir():
            raise NotADirectoryError(
                    'El nombre del directorio no es válido: '+ 
                    str(self.carpeta))

        self.sort_type = sort
        
        self._full_path = full_path 
        self.filter_ext = filter_ext #also performs the sorting
        self.filter_re = filter_re

    @property
    def filter_ext(self):
        return self._filter_ext
    @filter_ext.setter
    def filter_ext(self, value):
        if isinstance(value, str):
            self._filter_ext_tup = (value,)
        elif hasattr(value, '__iter__'):
            self._filter_ext_tup = value
        elif value is None:
            pass 
        else:
            raise TypeError('filter_ext must be a string representing an extension, or an iterable with with many extensions')
        self._filter_ext = value
        self.update()
        
    @property
    def filter_re(self):
        return self._filter_re
    @filter_re.setter
    def filter_re(self, pattern):
        if pattern is not None:
            try:
                re.compile(pattern)
            except TypeError:
                raise TypeError('filter_re is not a valid pattern')
        self._filter_re = pattern
        self.update()        

    @property
    def full_path(self):
        return self._full_path
    @full_path.setter
    def full_path(self, value):
        if not isinstance(value, bool):
            raise TypeError('full_path must be bool type')
        self._full_path = value
        self.update()
        
    @property
    def sort_type(self):
        return self._sort_type
    @sort_type.setter
    def sort_type(self, value):
        valid = ['natural', 'numeric', 'lexicographic', 'age', 'modified', 'size', None]
        if value not in valid:
            raise ValueError(f"Sort must be one of {valid[:-1]} or None.")
        self._sort_type = value
        self.__sorter()

    def __sorter(self):
        sort = self._sort_type
        if sort=='natural':
            self.natural_sort()
        elif sort=='numeric':
            self.numeric_sort()
        elif sort=='lexicographic':
            self.sort()
        elif sort=='age':
            self.age_sort()
        elif sort=='modified':
            self.last_modified_sort()
        elif sort=='size':
            self.size_sort()
        else:
            pass

    def update(self):
        # check if we are ready to initialize
        if not all( hasattr(self, attr_name) for attr_name in self.attr_list):
            return
        
        #filter extensions        
        if self.filter_ext is None:
            archivos = self.carpeta.iterdir()
        else: #filter extension, can't handle double extensions like .tar.gz
            archivos = (f for f in self.carpeta.iterdir() if f.suffix in self._filter_ext_tup)
        
        #filter with regex pattern
        if self.filter_re is not None:
            archivos = (f for f in archivos if re.match(self.filter_re, f.name))

        #make file
        if self.full_path:
            super().__init__(archivos)
        else:
            super().__init__(Path(f.name) for f in archivos)
        
        self.__sorter()

    def natural_sort(self):
        convert = lambda text: int(text) if text.isdigit() else text.lower() 
        self.sort(key=lambda key: [(convert(c), len(c)) for c in re.split('([0-9]+)', str(key))])
        
    def numeric_sort(self):
        """Extracts numbers from the filename (not the whole path) and uses those
        to sort. It's especially useful when the filenames of all the files follow
        the same parttern, except for a number. Natsort will give the same result,
        unless the numbers include floats."""
        self.sort(key=lambda file: find_numbers(str(file.name)))
    
    def age_sort(self):
        self.sort(key=os.path.getmtime)
        
    def last_modified_sort(self):
        self.sort(key=os.path.getctime)
        
    def size_sort(self):
        self.sort(key=self.get_size_of , reverse=True)
    
    def print_orden(self, filter_pattern=None, only_basename=True):

        # how many digits does the length of the lsit have?
        pad = len(str(len(self)-1))
        
        if only_basename:
            for i, f in enumerate(self):
                if filter_pattern is None or re.search(filter_pattern, f.name):
                    cprint(f'&ly {i:<{pad}}&s : {f.name}')
        else:
            for i, f in enumerate(self):
                if filter_pattern is None or re.search(filter_pattern, f):
                    cprint(f'&ly {i:<{pad}}&s : {f}')
                    
    def print_sizes(self, maxdepth=1):
        """ Print the contents along with the sizes"""
        
        pad = len(str(len(self)-1))
        
        filesizes = self.get_sizes() # in kB
        units = 'kB', 'MB', 'GB', 'TB'
        
        for i, (f, size) in enzip(self, filesizes):
            unit_inx = 0
            while size > 1000:
                size /= 1000
                unit_inx +=1
            
            size = smart_number_format(size, 3)
            cprint(f'&ly {i:<{pad}}&s : &lc {size:>4} {units[unit_inx]}&s  {f.name}')
    
    def get_sizes(self, maxdepth=1):
        """ Return a list containing the sizes of the elements in kB. For 
        directories, recursively compute the size of its contents up to
        maxdepth."""
        
        sizes = [self.get_size_of(f, maxdepth) / 1000 for f in self]
        return sizes
    
    @staticmethod
    def get_size_of(elem, maxdepth=3):
        """ Recursively get the size of an element up to maxdepth """
        
        # spetial case
        if maxdepth == 0 or not elem.is_dir():
            return elem.stat().st_size
        
        else:
            return sum(contenidos.get_size_of(sub_elem, maxdepth-1) for sub_elem in elem.iterdir())

    def filtered_ext(self, extension):
        '''Crea una nueva lista de las cosas con la extensión correspondiente.'''
        # return [elem for elem in self if elem.suffix == extension]
        return self.__class__(self.carpeta, self.full_path, self.sort_type, extension, self.filter_re)

    def filtered_re(self, pattern):
        '''Crea una nueva lista de las cosas que matcheen el patrón.'''
        # return [elem for elem in self if re.match(pattern, elem.name)]
        return self.__class__(self.carpeta, self.full_path, self.sort_type, self.filter_ext, pattern)

    # def filter_ext(self, extension):
    #     '''Elimina de la lista todo lo que no tenga la extensión correspondiente'''
    #     super().__init__(self.filtered_ext(extension))

    def files(self):
        '''Devuelve nueva lista de sólo los elementos que son archivos.'''
        return [elem for elem in self if elem.is_file()]
        
    def keep_fies(self):
        '''Elimina de la lista todo lo que no sean archivos.'''
        super().__init__(self.files())

    def directories(self):
        '''Devuelve nueva lista de sólo los elementos que son carpetas.'''
        return [elem for elem in self if elem.is_dir()]
    
    def keep_dirs(self):
        '''Elimina de la lista todo lo que no sean carpetas.'''
        super().__init__(self.directories())
        
    def basenames(self):
        '''Devuelve un iterador de los nombres de los elementos'''
        return (elem.name for elem in self)
    
    def stems(self):
        '''Devuelve un iterador de los stems de los elementos (sin extensión)'''
        return (elem.stem for elem in self)

class Testimado:
    """Una clase para estimar el tiempo restante de una operación que consiste
    de muchas operaciones similares repetidas. Se inicializa con la cantidad 
    total de operaciones. Cuando se llaman los métodos time_str, horas_minutos,
    horas_minutos_segundos o print_remainig con el íncice de la operación actual,
    estima el tiempo restante a partir del tiempo transcurrido hasta el momento
    y la proporción de tareas completadas.
    
    Ejemplo:
        
        def do_something():
            '''This should take a somewhat long time'''
            ...
        
        N = 50000
        TE = Testimado(N)
        for n in range(N):
            do_something()
            if n%100 == 0:
                TE.print_reamining(n)
    """

    def __init__(self, cant_total):
        self.cant_total = cant_total
        self.inicio = time.time()

    def restante(self, indice):
        return round((self.cant_total / (indice+1) - 1) * self.transcurrido())

    def transcurrido(self):
        return time.time() - self.inicio
    
    def horas_minutos(self, i):
         horas, rem = divmod(self.restante(i), 3600)
         minutos = rem//60
         return horas, minutos
         
    def horas_minutos_segundos(self, i):
         horas, rem = divmod(self.restante(i), 3600)
         minutos, segundos= divmod(rem, 60)
         return (horas, minutos, segundos)
     
    def time_str(self, i, include_times = 'HM'):
        '''Devuelve un string con el tiempo restante formateado según se indica
        en include_times.
        j: días
        H: horas
        M: minutos
        S: segundos'''
        format_string = ':'.join('%{}'.format(s) for s in include_times)
        return time.strftime(format_string, time.gmtime(self.restante(i)))
    
    def print_remaining(self, i, *a, **kw):
        print('ETA: {}'.format(self.time_str(i, *a, **kw)))



def not_instantaible(the_class):
    """ A class decorator to make a class not instantiable"""    
    def __new__(cls, *a, **kw):
        raise TypeError(cls.__name__ + " can't be directly instantiated")
    
    the_class.__new__ = __new__
    
    return the_class

@not_instantaible
class COLORS:
        
    STOP = '\33[00m'
    
    @not_instantaible    
    class FG:
        BLACK =     '\033[30m'
        RED =       '\033[31m'
        GREEN =     '\033[32m'
        YELLOW =    '\033[33m'
        BLUE =      '\033[34m'
        PURPLE =    '\033[35m'
        CYAN =      '\033[36m'
        GRAY =      '\033[37m'
        
        DARKGRAY =   '\033[90m'
        LIGHTRED =   '\033[91m'
        LIGHTGREEN = '\033[92m'
        LIGHTYELLOW ='\033[93m'
        LIGHTBLUE =  '\033[94m'
        LIGHTPURPLE ='\033[95m'
        LIGHTCYAN =  '\033[96m'
        WHITE =      '\033[97m'
        
    @not_instantaible        
    class BG:
        BLACK =     '\033[40m'
        RED =       '\033[41m'
        GREEN =     '\033[42m'
        ORANGE =    '\033[43m'
        BLUE =      '\033[44m'
        PURPLE =    '\033[45m'
        CYAN =      '\033[46m'
        GRAY =      '\033[47m'
        
        DARKGRAY =   '\033[100m'
        LIGHTRED =   '\033[101m'
        LIGHTGREEN = '\033[102m'
        LIGHTYELLOW ='\033[103m'
        LIGHTBLUE =  '\033[104m'
        PINK =       '\033[105m'
        LIGHTCYAN =  '\033[106m'
        WHITE =      '\033[107m'
        
    @not_instantaible
    class STYLE:
        BOLD =      '\033[01m'
        FAINT =     '\033[02m'
        ITALIC =    '\033[03m'
        UNDERLINE = '\033[04m'
        BLINK1 =    '\033[05m'
        BLINK2 =    '\033[06m'
        INVERT =    '\033[07m'
        STRIKET =   '\033[09m'
        
    @staticmethod
    def fg(code):
        if code not in range(16, 256):
            raise ValueError('Code is out of bounds [16, 255]')
        return f'\033[38;5;{code}m'
        
    @staticmethod
    def bg(code):
        if code not in range(16, 256):
            raise ValueError('Code is out of bounds [16, 255]')
        return f'\033[48;5;{code}m'
    
    @classmethod
    def test(cls, what=None):
        """Test the available formats in the current environment. What cna be 
        one of BG, FG or STYLE, or None."""
        
        # if what is None, print all codes
        if what is None:
            # print codes
            for i in range(120):
                end = ' ' if (i+1)%6 else '\n'
                print(f'\33[{i}m' + '{0:>10}'.format(f'\\33[{i}m') + '\33[0m', end=end)
            
            # print indexed colors
            for m in range(0, 3):
                print('\33[37m') if m==0 else print('\33[30m')
                for i in range(16, 232, 36):
                    for j in range(12):
                        if j==6:
                            print('\t', end='')
                        n = i+j+m*12
                        print(f'\33[48;5;{n}m{n:^5}', end='')
                    print(COLORS.STOP)
            
            # print grayscale indexed colors
            print('\33[37m') # white
            for i in range(232, 256):
                if i==244:
                    print('\33[30m')
                print(f'\33[48;5;{i}m{i:^5}', end='\t' if i in (237, 249) else '')
            print(COLORS.STOP)
            
            return
        
        # if it's the string 'LIGHT', print foreground and background colros preceded by bold
        if what.upper() == 'LIGHT':
            for what in (cls.FG, cls.BG):
                for k, v in what.__dict__.items():
                    if '__' not in k and 'LIGHT' not in k:
                        print(v, f'{k:>9}', '-->' , cls.STYLE.BOLD, 'LIGHT', k, cls.STOP)
            return
        
        # else, if it's a string, convert it to the class
        if what in ('BG', 'FG', 'STYLE'):
            what = getattr(cls, what)
        
        # if what is a valid class, print its contents, else, raise
        if any(what==c for c in (cls.FG, cls.BG, cls.STYLE)):
            for k, v in what.__dict__.items():
                if '__' not in k:
                    print(v, k,  cls.STOP)
        else:
            raise ValueError('what has to be one of BG, FG, STYLE or None')
        
# Create the aliases inside colors
for k, v in COLORS.FG.__dict__.items():
    if '__' not in k:
        setattr(COLORS, k, v)

def cprint(*msg, clearformat=True, **kwargs):
    """ Print with colors encoded into the print strings. if clearformat=True,
    then the format will be cleared at the end of the printed string.
    
    The elements of msg will be scanned for a pattern matching '&. ' where '&'
    indicates the start of the pattern, a whitespace indicates the end of the 
    pattern and the dot in the middle represents the code.
    The code has to be either one of the letters representing colors set in the
    variable dictionary below, or a color preceded by the letter b, to apply the
    color as a background.
    For example: 'The next word is in &c cyan, now with &gb green background'.
    Use '&& ' to print a literal '&'.
    
    """
    
    dictionary = {
        'k':'BLACK',
        'r':'RED',
        'g':'GREEN',
        'y':'YELLOW',
        'b':'BLUE',
        'p':'PURPLE',
        'c':'CYAN',
        'a':'GRAY',
        
        's':'STOP',
        
        'i':'ITALIC',
        'l':'BOLD', # use this to get lighter colors in some terminals
        'u':'UNDERLINE',
                  }
    pattern = '&.+? '

    def sub(match):
        styles= 'ilu'
        
        og_str = match.group(0)
        code = og_str[1:-1]
        
        if code == '&':
            return '&'
        
        if len(code)==1:
            if code in dictionary:
                if code in styles:
                    return getattr(COLORS.STYLE, dictionary[code])
                else:
                    return getattr(COLORS, dictionary[code])
        elif len(code)==2:
            if code[0] == 'b' and code[1] in dictionary:
                return getattr(COLORS.BG, dictionary[code[1]])
            elif all(c in dictionary for c in code):
                return ''.join(
                    getattr(COLORS.STYLE if c in styles else COLORS , dictionary[c]) for c in code
                    )
        
        return og_str

    msg = [re.sub(pattern, sub, str(s)) for s in msg]
    print(*msg, COLORS.STOP, **kwargs)


class Grid:
    '''Una clase para crear y llenar una grilla con imagenes. A menos que 
    especifique una cantidad de columnas o de filas, la grilla creada será 
    tan cuadrada como sea posible. Funciona con imágenes de un solo canal 
    (escala de grises) y de tres canales (RGB).
    
    -----------
    Parámetros:
        cant : int
            cantidad de imagenes que se introducirán en la grilla. Es necesario
            pasar este parámetro en la construcción de la clase para poder 
            planear el tamaño de la grilla.
        fill_with : int, float
            elemento con el que rellenar los lugares vacíos de la grilla. Afecta 
            cómo se verá la grilla al graficar. Default: np.nan
        trasponer : bool
            decide si trasponer la grilla calculada o no. Por defecto, si la 
            grilla no es cuadrada, tendrá más columnas que filas. Trasponer 
            invierte esto. Afecta también a grillas con cantidades personalizadas
            de filas y columnas. Default: False
        bordes : bool
            decide si poner bordes entre las imagenes y márgenes en torno a la 
            imagen. Default: True
        cant_col : int
            fija cantidad de columnas a utilizar, sobreescribiendo la cantidad
            calculada para optimizar cuadradez de la grilla. Si cant_col=None, 
            utilizará la calulada. Default: None
        cant_row : int
            fija cantidad de filas a utilizar, sobreescribiendo la cantidad
            calculada para optimizar cuadradez de la grilla. Si cant_row=None, 
            utilizará la calulada. Default: None
            
    --------
    Métodos:
        insert_image(im):
            Inserta una imagen a la grilla
        show(**kwargs):
            Muestra la grilla en su estado actual utilizando plt.imshow(). 
            Pasa los argumentos extra a imshow.
    '''
       
    def __init__(self, cant, fill_with=np.nan, trasponer=False, 
                 bordes=True, cant_col=None, cant_row=None ):

        self.cant = cant #cantidad de imagenes
        self.trasponer = trasponer #por default, la grilla es más ancha que alta
        self.bordes = bordes #si debe o no haber un margen entre figus.
        
        if cant_col is not None and cant_row is not None:
            raise ValueError('Sólo se puede especificar cantidad de filas o cantidad de columnas. La otra será deducida de la cantidad de elementos')
            
        self.shape = self._cant_to_mat(cant_col, cant_row) #tamaño de la matriz de imagenes

        self.grid = None #la grilla a llenar con imagenes
        #self.im_shape = None #tamaño de la imagen
        self.ind = 0 #por qué imagen voy?
        self.fill_with = fill_with #con qué lleno la grilla vacía

    @property
    def im_shape(self):
        return self._im_shape_real
    @im_shape.setter
    def im_shape(self, value):
        self._im_shape_real = value
        self._imRGB = len(value)==3 #if image is RGB
        
        if self.bordes:
            self._im_shape_bordes = (value[0] + 1, value[1] + 1)
        else:
            self._im_shape_bordes = self.im_shape
    
    @property
    def fill_with(self):
        return self._fill_with
    @fill_with.setter
    def fill_with(self, value):
        if self.grid is None:
            self._fill_with = value
        else:
            raise ValueError("Can't set fill_with after grid has been initialized.")

    def _cant_to_mat(self, cant_col, cant_row):
        '''Dimensiones de la cuadrícula más pequeña y más cuadrada
        posible que puede albergar [self.cant] cosas.'''
        
        if cant_col is not None:
            col = int(cant_col)
            row = int(np.ceil(self.cant/col))
        elif cant_row is not None:
            row = int(cant_row)
            col = int(np.ceil(self.cant/row))
        else:
            col = int(np.ceil(np.sqrt(self.cant)))
            row = int(round(np.sqrt(self.cant)))
            
        if self.trasponer:
            return col, row
        else:
            return row, col
        
    def _filcol(self):
        '''Pasa de índice lineal a matricial.'''
        fil = self.ind // self.shape[1]
        col = self.ind % self.shape[1]
        return int(fil), int(col)

    def _create_grid(self):
        shape = (self._im_shape_bordes[0] * self.shape[0], 
                 self._im_shape_bordes[1] * self.shape[1])
        if self.bordes:
            shape = shape[0] + 1, shape[1] + 1
        if self._imRGB:
            shape = *shape, 3
            
        self.grid = np.full(shape, self.fill_with)

    def insert_image(self, im):
        '''Agrego una imagen a la grilla.'''
        #inicializo la grilla
        if self.grid is None:
            self.im_shape = im.shape
            self._create_grid()

        #la lleno
        col, row = self._filcol()
        #sumo el booleano de bordes para que cierre bien la cuenta
        if self._imRGB:
            self.grid[col * self._im_shape_bordes[0] + int(self.bordes) : 
                        (col + 1) * self._im_shape_bordes[0],
                    row * self._im_shape_bordes[1] + int(self.bordes) : 
                        (row + 1) * self._im_shape_bordes[1], :]= im
        else:        
            self.grid[col * self._im_shape_bordes[0] + int(self.bordes) : 
                        (col + 1) * self._im_shape_bordes[0],
                    row * self._im_shape_bordes[1] + int(self.bordes) : 
                        (row + 1) * self._im_shape_bordes[1]]= im
        
        #avanzo el contador para la siguiente imagen
        self.ind += 1
        
    def show(self, **kw):
        return plt.imshow(self.grid, cmap=kw.pop('cmap', 'viridis'), **kw)

#Grid Testing
# cant = 33
# shape = (21,25)
# g = Grid(cant, trasponer=False, bordes=True, fill=100)
# for i in range(cant):
#     g.insert_image(np.ones(shape)*i)

# plt.matshow(g.grid)
# plt.grid()

# #%%

# cant = 17
# shape = (11,9)
# g = Grid(cant, trasponer=False, bordes=True, fill=np.nan)
# colores = [(0,0,0), (1,0,0), (0,1,0), (0,0,1), (0,1,1), (1,0,1), (1,1,0), 
#            (1,1,1), (.5,.5,.5), (1,.5,0), (1,0,.5), (.5,1,0), (.5,0,1),
#            (0,.5,1), (0,1,.5), (.5,0,0), (0,.5,0), (0,0,.5)]
# imagenes = []
# for c in colores:
#     liso = np.ones((*shape,3))
#     for i in range(3):
#         liso[:,:,i] *= c[i]
#     imagenes.append(liso)

# for i in range(cant):
#     g.insert_image(imagenes[i])

# plt.imshow(g.grid)
# plt.grid()

if pil_image is not None:
    _PIL_INTERPOLATION_METHODS = {
        'nearest': pil_image.NEAREST,
        'bilinear': pil_image.BILINEAR,
        'bicubic': pil_image.BICUBIC,
    }
    # These methods were only introduced in version 3.4.0 (2016).
    if hasattr(pil_image, 'HAMMING'):
        _PIL_INTERPOLATION_METHODS['hamming'] = pil_image.HAMMING
    if hasattr(pil_image, 'BOX'):
        _PIL_INTERPOLATION_METHODS['box'] = pil_image.BOX
    # This method is new in version 1.1.3 (2013).
    if hasattr(pil_image, 'LANCZOS'):
        _PIL_INTERPOLATION_METHODS['lanczos'] = pil_image.LANCZOS


def load_img(path, grayscale=False, color_mode='rgb', target_size=None,
             interpolation='nearest'):
    """Loads an image into PIL format.
    # Arguments
        path: Path to image file.
        grayscale: DEPRECATED use `color_mode="grayscale"`.
        color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
            The desired image format.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.
    # Returns
        A PIL Image instance.
    # Raises
        ImportError: if PIL is not available.
        ValueError: if interpolation method is not supported.
    """
    if grayscale is True:
        import warnings
        warnings.warn('grayscale is deprecated. Please use '
                      'color_mode = "grayscale"')
        color_mode = 'grayscale'
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `load_img` requires PIL.')
    img = pil_image.open(path)
    if color_mode == 'grayscale':
        if img.mode != 'L':
            img = img.convert('L')
    elif color_mode == 'rgba':
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
    elif color_mode == 'rgb':
        if img.mode != 'RGB':
            img = img.convert('RGB')
    else:
        raise ValueError('color_mode must be "grayscale", "rgb", or "rgba"')
    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            if interpolation not in _PIL_INTERPOLATION_METHODS:
                raise ValueError(
                    'Invalid interpolation method {} specified. Supported '
                    'methods are {}'.format(
                        interpolation,
                        ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
            resample = _PIL_INTERPOLATION_METHODS[interpolation]
            img = img.resize(width_height_tuple, resample)
    return img


def smooth(x, size, window=np.hamming):
    """
    Smooths data using a <size> sized window of the type given in <window>

    Parameters
    ----------
    x : array-like
        The data.
    size : int
        The size of the window.
    window : callable, optional
        A window type to use. Default is numpy.hamming, but any callable can 
        be passed. The default is np.hamming.

    Returns
    -------
    array-like
        Smoothed data of the same size as the input array, as long as the passed
        window has some behaviour to handle edges. This is the default behaviour.
    """
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    
    s=np.r_[x[size-1:0:-1], x, x[-2:-size-1:-1]]
    w = window(size)
    y = np.convolve(w/w.sum(), s, mode='same')
    
    size_diff = y.size - x.size
    lower = int(np.floor(size_diff/2))
    upper = int(np.ceil(size_diff/2))
    
    return y[lower:-upper]

def calc_mode(x, use='kde', bins=None, center=True):
    """
    Estimates the mode of the data given in x. To do so, it uses a kernel 
    density approximation (if use='kde') or creates an histogram (if use='hist')
    of the data and finds the bin with the highest count. The mode is then the 
    value of the bin that realizes the maximum.
    
    This function is useful when the data in question is continuos and you can
    only approximate the subjacent probability density.
    
    If all the datapoints are nan, simply return nan. Else, skip all nan values.

    Parameters
    ----------
    x : array-like
        Data over which the mode shall be calculated.
    use : 'kde' or 'hist'
        Whether to use a kernel density estimator or a histogram (respetively) 
        to estimate the mode. In the former case, bins and center are ignored.
        The default is 'kde'.
    bins : None, int, str or array-like, optional
        The bins argument passed to np.histogram. If bins=None, it uses whatever
        the default of that function, otherwise, the value gets passed to that
        function. The default is None.
    center : Bool, optional
        If center=True, it calculates the center of the bin that realzies the
        maximum. Otherwise, it uses the lower bound of the bin. The default is 
        True.

    Returns
    -------
    mode : number

    """
    
    # handle nans
    if all(np.isnan(x)):
        return np.nan
    
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    
    # if after filtering, only one element remains, return that as the mode
    if x.size == 1:
        return x[0]
    
    if use=='kde':
        # following https://rmflight.github.io/post/finding-modes-using-kernel-density-estimates/
        
        # estimate the kernel density
        kernel = stats.gaussian_kde(x)
        # use the calculated density and evaluate it the pdf in the given values
        # here we could evaluate the pdf in np.linspace(x.min, x.max, 1000), instead
        # this approach gives us a value of x as mode, rather than just the maximum of the estimated pdf
        height = kernel.pdf(x)
        # find the maximal value of the mdf, which approximates the mode
        mode = x[np.argmax(height)]
    
    elif use=='hist':
    
        if bins is None:
            counts, bin_edges = np.histogram(x)
        else:
            counts, bin_edges = np.histogram(x, bins=bins)
        
        max_inx = np.argmax(counts)
        
        mode = bin_edges[max_inx]
        if center:
            mode += (bin_edges[max_inx + 1] - bin_edges[max_inx + 1]) / 2
        
    else:
        raise ValueError(f"'use' was to be one of 'kde' or 'hist', not {use}.")
    
    return mode


'''
Defines a function colorline that draws a (multi-)colored 2D line with coordinates x and y.
The color is taken from optional data in z, and creates a LineCollection.

z can be:
- empty, in which case a default coloring will be used based on the position along the input arrays
- a single number, for a uniform color [this can also be accomplished with the usual plt.plot]
- an array of the length of at least the same length as x, to color according to this data
- an array of a smaller length, in which case the colors are repeated along the curve

The function colorline returns the LineCollection created, which can be modified afterwards.

See also: plt.streamplot

See https://nbviewer.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
for examples
'''


# Data manipulation:

def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    return segments

# Interface to LineCollection:

def colorline(x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0, ax=None):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify values to be mapped to colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''
    
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
           
    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])
        
    z = np.asarray(z)
    
    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    
    ax = plt.gca() if ax is None else ax
    ax.add_collection(lc)
    
    return lc
        
    
def clear_frame(ax=None, hidespine=True): 
    """Hides axis frame, ticks and so on"""
    # Taken from a post by Tony S Yu
    if ax is None: 
        ax = plt.gca() 
    ax.xaxis.set_visible(False) 
    ax.yaxis.set_visible(False) 
    if hidespine:
        for spine in ax.spines.values(): 
            spine.set_visible(False) 

