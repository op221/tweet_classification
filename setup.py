from distutils.core import setup


def readme():
    """Import the README.md Markdown file and try to convert it to RST format."""
    try:
        import pypandoc
        return pypandoc.convert('README.md', 'rst')
    except(IOError, ImportError):
        with open('README.md') as readme_file:
            return readme_file.read()


setup(
    name='tweet',
    version='0.1',
    description='Classification of Disaster Tweets',
    long_description=readme(),
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    url='https://github.com/op221/tweet_classification',
    author='Jason Kim',
    author_email='oceanpark221@gmail.com',
    license='JSK',
    packages=['tweet'],
)
