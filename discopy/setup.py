"""
Setup discopy package.
"""

if __name__ == '__main__':  # pragma: no cover
    from setuptools import setup, find_packages

    setup(name='discopy',
          version="THESIS",
          package_dir={'discopy': 'discopy'},
          description='Distributional Compositional Python',
          long_description=open("README.md", "r").read(),
          long_description_content_type="text/markdown",
          url='https://github.com/toumix/discopy',
          author='Alexis Toumi',
          author_email='alexis.toumi@cs.ox.ac.uk',
          python_requires='>=3',
          )
