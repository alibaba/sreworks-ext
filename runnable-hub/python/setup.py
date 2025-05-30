from setuptools import setup, find_packages

setup(
    name='runnable_hub',
    version='0.3',
    packages=find_packages(),
    install_requires=[
        "pydantic",
        "aiohttp",
        "jinja2"
    ],
    author='jiongen.zje',
    author_email='jiongen.zje@alibaba-inc.com',
    description='Runnable Hub is a middleware for large models, enabling management and execution of complex inference tasks with a flexible architecture and support for multiple task types.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/alibaba/sreworks-ext/tree/main/runnable-hub',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.12',
)