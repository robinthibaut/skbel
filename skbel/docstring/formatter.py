"""this script format docstrings in python files following the Sphinx format"""
# In general, a typical Sphinx docstring has the following format:
#
# """[Summary]
#
# :param [ParamName]: [ParamDescription], defaults to [DefaultParamVal]
# :type [ParamName]: [ParamType](, optional)
# ...
# :raises [ErrorType]: [ErrorDescription]
# ...
# :return: [ReturnDescription]
# :rtype: [ReturnType]
# """

import re

# The following regexes are used to parse the docstring and extract the
# information. The first one is used to extract the summary, the second one
# extracts the parameters, the third one extracts the exceptions, the fourth
# one extracts the return value and the fifth one extracts the return type.
# The regexes are not perfect, but they work for the most common cases.
SUMMARY_REGEX = re.compile(r'"""(.*)"""', re.DOTALL)
PARAM_REGEX = re.compile(r':param\s+(\w+):\s*(.*)', re.DOTALL)
EXCEPTION_REGEX = re.compile(r':raises\s+(\w+):\s*(.*)', re.DOTALL)
RETURN_REGEX = re.compile(r':return:\s*(.*)', re.DOTALL)
RETURNTYPE_REGEX = re.compile(r':rtype:\s*(.*)', re.DOTALL)

# The following regexes are used to format the docstring. The first one is
# used to format the summary, the second one is used to format the parameters,
# the third one is used to format the exceptions, the fourth one is used to
# format the return value and the fifth one is used to format the return type.
SUMMARY_FORMAT = '"""{}"""'
PARAM_FORMAT = ':param {}: {}'
EXCEPTION_FORMAT = ':raises {}: {}'
RETURN_FORMAT = ':return: {}'
RETURNTYPE_FORMAT = ':rtype: {}'


# This function loads the file and reformats the docstring.
def format_docstring(filename: str) -> None:
    """Formats the docstring of a file.

    :param filename: The name of the file to format.
    """
    with open(filename, 'r') as f:
        content = f.read()
    new_content = format_docstring_string(content)
    if new_content != content:
        with open(filename, 'w') as f:
            f.write(new_content)


# This function reformats the docstring.
def format_docstring_string(content: str) -> str:
    """Formats the docstring of a string.

    :param content: The string to format.
    :return: The formatted string.
    """
    # Extract the summary.
    summary = SUMMARY_REGEX.search(content)
    if summary is None:
        return content
    summary = summary.group(1).strip()
    # Extract the parameters.
    params = PARAM_REGEX.findall(content)
    # Extract the exceptions.
    exceptions = EXCEPTION_REGEX.findall(content)
    # Extract the return value.
    return_value = RETURN_REGEX.search(content)
    if return_value is not None:
        return_value = return_value.group(1).strip()
    # Extract the return type.
    return_type = RETURNTYPE_REGEX.search(content)
    if return_type is not None:
        return_type = return_type.group(1).strip()
    # Format the summary.
    new_summary = SUMMARY_FORMAT.format(summary)
    # Format the parameters.
    new_params = [
        PARAM_FORMAT.format(param, description)
        for param, description in params
    ]
    # Format the exceptions.
    new_exceptions = [
        EXCEPTION_FORMAT.format(exception, description)
        for exception, description in exceptions
    ]
    # Format the return value.
    new_return_value = RETURN_FORMAT.format(return_value) if return_value else ''
    # Format the return type.
    new_return_type = RETURNTYPE_FORMAT.format(return_type) if return_type else ''
    # Join everything together.
    new_content = ' '.join([
        new_summary,
        ''.join(new_params),
        ' '.join(new_exceptions),
        new_return_value,
        new_return_type,
    ])
    # Replace the old docstring with the new one.
    return SUMMARY_REGEX.sub(new_content, content)


if __name__ == '__main__':
    filename = '/Users/robin/PycharmProjects/SKBEL/skbel/tmaps/transport_map_ds.py'
    format_docstring(filename)
