from setuptools import setup

setup(name='stocks',
      packages=['stocks'],
      version='0.0.1dev1',
      entry_points={
            'console_scripts': ['stocks-cmd=stocks.cmd:main']
      }
      )