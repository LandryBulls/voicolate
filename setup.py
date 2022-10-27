import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name='voicolate',
    version='0.0.1',
    author='Landry Bulls',
    author_email='landry.s.bulls.gr@dartmouth.edu',
    description='Voice isolation for conversation studies with multiple microphones.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/LandryBulls/voicolate',
    project_urls = {
        "Bug Tracker": "https://github.com/LandryBulls/voicolate/issues"
    },
    license='MIT',
    packages=['voicolate'],
    install_requires=requirements
)