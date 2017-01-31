from setuptools import setup

setup(name='NNFlow',
      version='0.1',
      description='NNFlow framework to convert ROOT files to Tensorflow models',
      url='https://github.com/kit-cn-cms/NNFlow',
      author='KIT CN CMS team: Maximilian Welsch, Marco A. Harrendorf',
      author_email='flyingcircus@example.com',
      packages=['NNFlow'],
      install_requires=[
          'matplotlib',
          'root-numpy',
          'tensorflow==0.11.0*',
      ],
      zip_safe=True)
