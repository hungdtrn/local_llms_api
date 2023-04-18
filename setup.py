from distutils.core import setup

setup(name='local_llms_api',
      version='0.1.0',
      description='API Wrapper for local llms',
      author='Hung Tran',
      author_email='hung.dtrn@gmail.com',
      package_dir={"": "src"},
      python_requires=">=3.7",
     )