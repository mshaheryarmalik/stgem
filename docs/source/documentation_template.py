"""
# Here you can use markdown if you wish
Source code is in docs/source/documentation_template.py
"""
class ExampleClass(object):
    """

    :Use rst syntax in class and method docstrings:

    This is *italic* and **bold**

    :Sections need blank lines before and after to function:

    .. note:: Something important
    .. warning:: Example
    .. seealso:: awdawd
    """
    def templateMethod(self, x, y, z):
        """[Summary]

        :param [ParamName]: [ParamDescription], defaults to [DefaultParamVal]
        :type [ParamName]: [ParamType](, optional)
        :raises [ErrorType]: [ErrorDescription]
        :return: [ReturnDescription]
        :rtype: [ReturnType]
        """

    def Sum(self, x, y, optional):
        """Add x and y values.

        :param x: x value
        :type x: int
        :param y: y value
        :type y: int
        :param optional: Optional value, defaults to 0
        :type optional: int, optional
        :raises IntegerOverflowException: Sum of x and y is too big
        :return: Sum of x and y
        :rtype: int
        """