"""
# Here you can use markdown if you wish
Source code is in docs/source/documentation_template.py
"""
class ExampleClass(object):
    """

    Use rst syntax in class and method docstrings

    This is *italic* and **bold**

    Sections need blank lines before and after to function

    Note:
        Something important
    Warning:
        Example
    See Also:
        Make sure you check this out for more section options: https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
    """
    def templateMethod(self, x, y, z):
        """
        Args:
            path (str): The path of the file to wrap
            field_storage (FileStorage): The `FileStorage` instance to wrap
            temporary (bool): Whether or not to delete the file when the `File` instance is destructed

        Returns:
            BufferedFileStorage: A buffered writable file descriptor

        Raises:
            [ErrorType]: [ErrorDescription]
        """

    def Sum(self, x, y, optional):
        """Add x and y values.


        Args:
            x (int): x value
            y (int): y value

            optional (int, optional): Optional value, defaults to 0
        Raises:
            IntegerOverflowException: Sum of x and y is too big
        Returns:
            int: Sum of x and y

        """