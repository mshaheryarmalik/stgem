"""
# Here you can use markdown if you wish
Source code is in docs/source/documentation_template.py
"""
class ExampleClass(object):
    """Use Google style in class and method docstrings

    Note:
        Use \:meta private: and \:meta public: in class and method docstrings to hide/unhide them.
    Warning:
        This is *italic* and **bold**.
    See Also:
        Check out following for more section options: https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html.
    """
    def exampleMethod(self, path, field_storage, temporary):
        """Method summary

        Args:
            path (str): The path of the file to wrap
            field_storage (FileStorage): The `FileStorage` instance to wrap
            temporary (bool): Whether to delete the file when the `File` instance is destructed

        Returns:
            BufferedFileStorage: A buffered writable file descriptor

        Raises:
            OutOfBoundsException: Out of bounds
        """

    def exampleSum(self, x, y, optional):
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